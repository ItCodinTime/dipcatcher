"""Baseline-diff security scanner reports — only *new* findings fail the gate.

The repo pins its scanners (bandit, pip-audit) and runs them in
``.github/workflows/security.yml``. A raw "any finding fails" gate would make
every pre-existing, risk-accepted finding block unrelated PRs, so findings are
fingerprinted and checked into ``security/baselines/``:

- A finding *not* in the baseline is new → exit 1 (the gate fails).
- A baseline entry no longer reported is resolved → printed, exit 0;
  regenerate the baseline with ``--update`` to keep it honest.
- Fingerprints exclude volatile fields (line numbers, timestamps) so
  refactors do not create phantom findings; a snippet change re-flags.

Stdlib-only so the CI job needs no extra dependencies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

BASELINE_SCHEMA = "security-baseline.v1"


def _fingerprint(parts: list[str]) -> str:
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def _bandit_findings(report: dict[str, Any]) -> dict[str, str]:
    """fingerprint -> summary for each bandit result."""
    out: dict[str, str] = {}
    for result in report.get("results", []):
        if not isinstance(result, dict):
            continue
        code = str(result.get("code") or "").strip()
        code_digest = hashlib.sha256(code.encode("utf-8")).hexdigest()
        fp = _fingerprint(
            [
                "bandit",
                str(result.get("filename")),
                str(result.get("test_id")),
                str(result.get("issue_severity")),
                str(result.get("issue_confidence")),
                code_digest,
            ]
        )
        out[fp] = (
            f"{result.get('filename')}:{result.get('line_number')} "
            f"{result.get('test_id')} {result.get('issue_severity')}/"
            f"{result.get('issue_confidence')} {result.get('issue_text', '')[:100]}"
        )
    return out


def _pip_audit_findings(report: dict[str, Any]) -> dict[str, str]:
    """fingerprint -> summary for each pip-audit dependency vuln."""
    out: dict[str, str] = {}
    for dep in report.get("dependencies", []):
        if not isinstance(dep, dict):
            continue
        name = dep.get("name")
        for vuln in dep.get("vulns", []):
            if not isinstance(vuln, dict):
                continue
            fix_versions = ",".join(sorted(str(v) for v in vuln.get("fix_versions", [])))
            fp = _fingerprint(["pip-audit", str(name), str(vuln.get("id")), fix_versions])
            out[fp] = f"{name}=={dep.get('version')} {vuln.get('id')} fix={fix_versions or 'none'}"
    return out


_FINDINGS = {
    "bandit": _bandit_findings,
    "pip-audit": _pip_audit_findings,
}


def load_baseline(path: Path) -> dict[str, str]:
    """fingerprint -> recorded summary. Missing baseline = empty (everything new)."""
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema") != BASELINE_SCHEMA:
        raise ValueError(f"baseline {path} has schema {data.get('schema')!r}")
    out: dict[str, str] = {}
    for entry in data.get("findings", []):
        if isinstance(entry, dict) and isinstance(entry.get("fingerprint"), str):
            out[entry["fingerprint"]] = str(entry.get("summary", ""))
    return out


def write_baseline(path: Path, tool: str, findings: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": BASELINE_SCHEMA,
        "tool": tool,
        "findings": [
            {"fingerprint": fp, "summary": summary} for fp, summary in sorted(findings.items())
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool", required=True, choices=sorted(_FINDINGS))
    parser.add_argument("--report", required=True, type=Path, help="scanner JSON output")
    parser.add_argument("--baseline", required=True, type=Path, help="checked-in baseline JSON")
    parser.add_argument(
        "--update",
        action="store_true",
        help="rewrite the baseline from the report instead of diffing",
    )
    args = parser.parse_args(argv)

    report = json.loads(args.report.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        print(f"security-baseline: {args.report} is not a JSON object", file=sys.stderr)
        return 2
    findings = _FINDINGS[args.tool](report)

    if args.update:
        write_baseline(args.baseline, args.tool, findings)
        print(f"security-baseline: wrote {len(findings)} fingerprints to {args.baseline}")
        return 0

    try:
        baseline = load_baseline(args.baseline)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"security-baseline: cannot load {args.baseline}: {exc}", file=sys.stderr)
        return 2

    new = sorted(set(findings) - set(baseline))
    resolved = sorted(set(baseline) - set(findings))
    for fp in resolved:
        print(f"security-baseline: resolved since baseline: {baseline[fp]}")
    if new:
        print(f"security-baseline: {len(new)} NEW {args.tool} finding(s) not in baseline:")
        for fp in new:
            print(f"  {findings[fp]}")
        print(
            "Triage each finding; if it is a deliberate, risk-accepted pattern, "
            "regenerate with --update and commit the new baseline."
        )
        return 1
    print(
        f"security-baseline: clean — {len(findings)} {args.tool} finding(s), "
        f"all present in baseline; {len(resolved)} resolved"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
