"""Render the reference-to-code table from the packaged catalog (no network)."""

from pathlib import Path

from quant_fund.research.research100 import load_catalog

root = Path(__file__).resolve().parents[1]
lines = [
    "# Research 100: source-to-code map",
    "",
    "Selected for relevance to Dipcatcher across prediction, allocation, execution, and validation. "
    "IDs are stable identifiers, not an objective ranking of expected performance. "
    "There are 96 existing components and four newly added components. "
    "Every entry is a component mapping; zero original empirical studies are marked as reproduced.",
    "",
    "Source review means a matching primary paper, author manuscript, institutional record, or publisher record "
    "was located on 2026-09-24. It does not mean the full text was audited for all 100 methods. "
    "The book entry R072 covers CPCV specifically.",
    "",
    "See [usage and evidence](RESEARCH100.md) for runnable commands, scope, and remaining validation.",
    "",
    "| ID | Reference | Component | Status | Linked test suites |",
    "| --- | --- | --- | --- | --- |",
]
for r in load_catalog():
    module, symbol = r["entrypoint"].split(":")
    path = "https://github.com/artificial-hedge/dipcatcher/blob/main/src/" + module.replace(".", "/") + ".py"
    tests = ", ".join(f"[{Path(t).name}](https://github.com/artificial-hedge/dipcatcher/blob/main/{t})" for t in r["test_paths"])
    title = r["title"].replace("|", "/")
    lines.append(
        f"| {r['id']} | [{title}]({r['source_url']}) — {r['authors']} | [{symbol}]({path}) | {r['implementation_status']} | {tests} |"
    )
lines.extend(["", "## Implementation boundaries", ""])
for r in load_catalog():
    lines.append(f"- **{r['id']}**: {r['implementation_notes']}")
(root / "docs" / "RESEARCH100_CATALOG.md").write_text("\n".join(lines) + "\n")
