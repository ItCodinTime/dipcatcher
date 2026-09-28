#!/usr/bin/env python3
"""Verify a release artifact directory produced by `.github/workflows/release.yml`.

Fail-closed checks (always):

1. ``SHA256SUMS`` exists and lists at least one wheel, one sdist, and
   ``sbom.cdx.json``.
2. Every listed digest matches the on-disk file bytes.
3. No listed path is missing.

Optional Sigstore identity verification (``--sigstore``) shells out to the
``sigstore`` CLI when ``.sigstore.json`` bundles sit next to the
distributions. That path needs either network access or ``--offline`` with a
cached/baked trust root; unit tests cover checksums only.

SLSA provenance lives in GitHub artifact attestations and is verified with
``gh attestation verify``, not this script.

Examples::

    uv run python scripts/verify_release_artifacts.py dist
    uv run python scripts/verify_release_artifacts.py dist --sigstore \\
      --cert-identity \\
      'https://github.com/artificial-hedge/dipcatcher/.github/workflows/release.yml@refs/tags/v0.1.0'
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

SUMS_NAME = "SHA256SUMS"
SBOM_NAME = "sbom.cdx.json"
OIDC_ISSUER = "https://token.actions.githubusercontent.com"


class ReleaseVerificationError(Exception):
    """Raised when a release artifact directory fails verification."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_sha256sums(sums_path: Path) -> dict[str, str]:
    """Parse GNU ``sha256sum`` output into ``{relative_name: hex_digest}``."""
    if not sums_path.is_file():
        raise ReleaseVerificationError(f"missing checksum file: {sums_path}")
    entries: dict[str, str] = {}
    for line_no, raw in enumerate(sums_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            raise ReleaseVerificationError(f"{sums_path}:{line_no}: malformed line")
        digest, name = parts[0].lower(), parts[1]
        if name.startswith("*") or name.startswith(" "):
            name = name[1:]
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ReleaseVerificationError(f"{sums_path}:{line_no}: invalid sha256 digest")
        if name in entries:
            raise ReleaseVerificationError(f"{sums_path}:{line_no}: duplicate entry {name!r}")
        entries[name] = digest
    if not entries:
        raise ReleaseVerificationError(f"{sums_path}: no checksum entries")
    return entries


def _require_layout(entries: dict[str, str]) -> None:
    wheels = [n for n in entries if n.endswith(".whl")]
    sdists = [n for n in entries if n.endswith(".tar.gz")]
    if not wheels:
        raise ReleaseVerificationError("SHA256SUMS lists no wheel (*.whl)")
    if not sdists:
        raise ReleaseVerificationError("SHA256SUMS lists no sdist (*.tar.gz)")
    if SBOM_NAME not in entries:
        raise ReleaseVerificationError(f"SHA256SUMS missing {SBOM_NAME}")


def verify_checksums(dist_dir: Path) -> dict[str, str]:
    """Verify every ``SHA256SUMS`` entry against files under ``dist_dir``.

    Returns the parsed checksum map on success. Raises
    :class:`ReleaseVerificationError` on any mismatch, missing file, or
    layout failure (including a tampered artifact).
    """
    root = Path(dist_dir)
    if not root.is_dir():
        raise ReleaseVerificationError(f"not a directory: {root}")
    entries = parse_sha256sums(root / SUMS_NAME)
    _require_layout(entries)
    for name, expected in sorted(entries.items()):
        path = (root / name).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError as exc:
            raise ReleaseVerificationError(f"path escapes dist dir: {name}") from exc
        if not path.is_file():
            raise ReleaseVerificationError(f"missing artifact listed in {SUMS_NAME}: {name}")
        actual = _sha256_file(path)
        if actual != expected:
            raise ReleaseVerificationError(
                f"checksum mismatch for {name}: expected {expected}, got {actual}"
            )
    sbom_path = root / SBOM_NAME
    try:
        doc = json.loads(sbom_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ReleaseVerificationError(f"{SBOM_NAME}: invalid JSON: {exc}") from exc
    if doc.get("bomFormat") != "CycloneDX":
        raise ReleaseVerificationError(f"{SBOM_NAME}: bomFormat is not CycloneDX")
    if not (doc.get("components") or []):
        raise ReleaseVerificationError(f"{SBOM_NAME}: no components")
    return entries


def verify_sigstore_bundles(
    dist_dir: Path,
    *,
    cert_identity: str,
    offline: bool = False,
    sigstore_bin: str = "sigstore",
) -> None:
    """Verify ``*.sigstore.json`` bundles next to wheel/sdist files."""
    root = Path(dist_dir)
    dists = sorted(root.glob("*.whl")) + sorted(root.glob("*.tar.gz"))
    if not dists:
        raise ReleaseVerificationError("no wheel/sdist files for Sigstore verify")
    for path in dists:
        bundle = Path(str(path) + ".sigstore.json")
        if not bundle.is_file():
            bundle = path.with_suffix(path.suffix + ".sigstore.json")
        if not bundle.is_file():
            # sigstore 4.x writes "<filename>.sigstore.json"
            candidates = list(root.glob(path.name + ".sigstore.json"))
            if not candidates:
                raise ReleaseVerificationError(f"missing Sigstore bundle for {path.name}")
            bundle = candidates[0]
        cmd = [
            sigstore_bin,
            "verify",
            "identity",
            "--cert-identity",
            cert_identity,
            "--cert-oidc-issuer",
            OIDC_ISSUER,
            "--bundle",
            str(bundle),
            str(path),
        ]
        if offline:
            cmd.insert(3, "--offline")
        try:
            completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
        except FileNotFoundError as exc:
            raise ReleaseVerificationError(
                f"{sigstore_bin!r} not found; install sigstore or omit --sigstore"
            ) from exc
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip()
            raise ReleaseVerificationError(
                f"Sigstore verify failed for {path.name}: {detail or completed.returncode}"
            )


def verify_release_artifacts(
    dist_dir: Path | str,
    *,
    require_sigstore: bool = False,
    cert_identity: str | None = None,
    offline: bool = False,
    sigstore_bin: str = "sigstore",
) -> dict[str, str]:
    """Verify checksums (always) and optionally Sigstore bundles."""
    entries = verify_checksums(Path(dist_dir))
    if require_sigstore:
        if not cert_identity:
            raise ReleaseVerificationError("--sigstore requires --cert-identity")
        verify_sigstore_bundles(
            Path(dist_dir),
            cert_identity=cert_identity,
            offline=offline,
            sigstore_bin=sigstore_bin,
        )
    return entries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dist_dir",
        type=Path,
        help="Directory containing SHA256SUMS, wheel, sdist, and sbom.cdx.json",
    )
    parser.add_argument(
        "--sigstore",
        action="store_true",
        help="Also verify *.sigstore.json bundles via the sigstore CLI",
    )
    parser.add_argument(
        "--cert-identity",
        default=None,
        help="Expected Sigstore certificate identity (workflow URL @ ref)",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Pass --offline to sigstore verify (bundle + cached trust root)",
    )
    parser.add_argument(
        "--sigstore-bin",
        default="sigstore",
        help="sigstore executable name or path (default: sigstore)",
    )
    args = parser.parse_args(argv)
    try:
        entries = verify_release_artifacts(
            args.dist_dir,
            require_sigstore=args.sigstore,
            cert_identity=args.cert_identity,
            offline=args.offline,
            sigstore_bin=args.sigstore_bin,
        )
    except ReleaseVerificationError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print(f"OK: verified {len(entries)} artifact(s) under {args.dist_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
