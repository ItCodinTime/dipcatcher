"""Markdown research reports. Synthetic runs are labeled."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def write_report(path: Path, title: str, sections: dict[str, Any], *, synthetic: bool) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", ""]
    if synthetic:
        lines += ["> SYNTHETIC DATA. Not evidence of live profitability.", ""]
    for name, body in sections.items():
        lines.append(f"## {name}")
        if isinstance(body, dict):
            for k, v in body.items():
                if isinstance(v, float):
                    lines.append(f"- {k}: {v:.6g}")
                else:
                    lines.append(f"- {k}: {v}")
        else:
            lines.append(str(body))
        lines.append("")
    path.write_text("\n".join(lines))
    return path


def latest_report_dir(root: Path) -> Path:
    d = root / "metadata" / "reports"
    d.mkdir(parents=True, exist_ok=True)
    return d
