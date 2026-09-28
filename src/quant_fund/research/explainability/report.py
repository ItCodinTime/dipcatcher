"""Explainability report assembly and rendering.

``build_report`` orchestrates permutation attribution, top-k partial
dependence, and time-blocked drift into an :class:`ExplainabilityReport`.
``to_markdown`` / ``to_html`` render it (HTML is self-contained: charts are
base64-embedded PNGs drawn with matplotlib's Agg canvas — no external assets,
no CDN, no JavaScript). ``write_report`` persists ``explainability.{md,html,
json}`` atomically with sha256 sidecars, matching repo artifact conventions.

Honesty contract: scores are proper losses only; reports carry an explicit
``synthetic`` label and ``research_only`` claim and never headline P&L/NAV/
ratio metrics.
"""

from __future__ import annotations

import base64
import html
import io
import json
import math
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from quant_fund.research.explainability.attribution import (
    AttributionResult,
    _validate_xy,
    permutation_attribution,
    shap_attribution,
)
from quant_fund.research.explainability.drift import (
    AttributionDrift,
    attribution_drift,
)
from quant_fund.research.explainability.partial_dependence import (
    PartialDependenceCurve,
    partial_dependence_top_k,
)
from quant_fund.research.explainability.scoring import (
    PredictFn,
    ProperScoreSpec,
    model_predict_fn,
    resolve_score,
)
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes, hash_file

EXPLAINABILITY_REPORT_SCHEMA = "explainability_report.v1"


@dataclass(frozen=True)
class ExplainabilityReport:
    """Self-contained explainability summary for one fitted model head."""

    model_name: str
    model_family: str
    scoring: str
    baseline_score: float
    attribution: AttributionResult
    pd_curves: tuple[PartialDependenceCurve, ...]
    drift: AttributionDrift | None
    feature_names: tuple[str, ...]
    n_rows: int
    seed: int
    synthetic: bool
    generated_at: str
    dataset_sha256: str
    shap: AttributionResult | None = None
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        drift_dict: dict[str, Any] | None = None
        if self.drift is not None:
            drift_dict = {
                "n_blocks": self.drift.n_blocks,
                "block_rows": [b.n_rows for b in self.drift.blocks],
                "consecutive_spearman": list(self.drift.consecutive_spearman),
                "consecutive_js": list(self.drift.consecutive_js),
                "consecutive_topk_overlap": list(self.drift.consecutive_topk_overlap),
                "per_feature_max_share_change": dict(self.drift.per_feature_max_share_change),
                "min_spearman": self.drift.min_spearman,
                "max_js": self.drift.max_js,
                "mean_topk_overlap": self.drift.mean_topk_overlap,
                "drift_flagged": self.drift.drift_flagged,
                "thresholds": dict(self.drift.thresholds),
                "warnings": list(self.drift.warnings),
                "per_block_importance": [
                    {
                        "block_index": b.block_index,
                        "n_rows": b.n_rows,
                        "baseline_score": b.result.baseline_score,
                        "importances": {
                            a.feature: a.importance_mean for a in b.result.attributions
                        },
                    }
                    for b in self.drift.blocks
                ],
            }
        return cast(
            "dict[str, Any]",
            _json_safe(
                {
                    "schema": EXPLAINABILITY_REPORT_SCHEMA,
                    "model": {"name": self.model_name, "family": self.model_family},
                    "scoring": self.scoring,
                    "baseline_score": self.baseline_score,
                    "synthetic": bool(self.synthetic),
                    "generated_at": self.generated_at,
                    "seed": self.seed,
                    "n_rows": self.n_rows,
                    "n_features": len(self.feature_names),
                    "dataset_sha256": self.dataset_sha256,
                    "feature_names": list(self.feature_names),
                    "importances": [
                        {
                            "feature": a.feature,
                            "importance_mean": a.importance_mean,
                            "importance_std": a.importance_std,
                            "rank": a.rank,
                        }
                        for a in self.attribution.attributions
                    ],
                    "partial_dependence": [
                        {
                            "feature": c.feature,
                            "grid": list(c.grid),
                            "mean_curve": list(c.mean_curve),
                        }
                        for c in self.pd_curves
                    ],
                    "drift": drift_dict,
                    "shap": (
                        [
                            {
                                "feature": a.feature,
                                "mean_abs_shap": a.importance_mean,
                                "rank": a.rank,
                            }
                            for a in self.shap.attributions
                        ]
                        if self.shap is not None
                        else None
                    ),
                    "warnings": list(self.warnings),
                    "claim": "research_only",
                },
            ),
        )

    def to_markdown(self) -> str:
        return _render_markdown(self)

    def to_html(self) -> str:
        return _render_html(self)


def _json_safe(value: Any) -> Any:
    """Represent undefined diagnostics as JSON null, never nonstandard NaN."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _dataset_digest(x: NDArray[np.float64], y: NDArray[np.float64]) -> str:
    """Portable little-endian content hash of eval features and labels."""
    xx = np.ascontiguousarray(np.asarray(x, dtype="<f8"))
    yy = np.ascontiguousarray(np.asarray(y, dtype="<f8").ravel())
    shape = np.asarray(xx.shape, dtype="<i8")
    return hash_bytes(xx.tobytes() + yy.tobytes() + shape.tobytes())


def _resolve_names(model: Any, explicit: list[str] | None, n_features: int) -> list[str]:
    if explicit is not None:
        names = [str(v) for v in explicit]
        if len(names) != n_features:
            raise ValueError(f"feature_names length {len(names)} != x columns {n_features}")
        return names
    if model is not None:
        metadata = getattr(model, "metadata", None)
        if callable(metadata):
            try:
                meta = metadata()
                features = getattr(meta, "features", None)
                if isinstance(features, list) and len(features) == n_features:
                    return [str(v) for v in features]
            except (TypeError, ValueError, AttributeError):
                pass
    return [f"x{j}" for j in range(n_features)]


def build_report(
    model: Any | None,
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    *,
    predict: PredictFn | None = None,
    times: NDArray[Any] | None = None,
    feature_names: list[str] | None = None,
    scoring: ProperScoreSpec | str | None = None,
    top_k: int = 8,
    n_blocks: int = 4,
    n_repeats: int = 5,
    seed: int = 42,
    grid_points: int = 21,
    synthetic: bool = False,
    model_name: str | None = None,
    model_family: str | None = None,
    generated_at: str | None = None,
    include_shap: bool = False,
    drift: bool = True,
    max_rows: int | None = None,
    pd_top_k: int | None = None,
) -> ExplainabilityReport:
    """Build an explainability report for a fitted ForecastModel head.

    ``model`` is any object with ``predict(x)`` (``fit``/``metadata`` per the
    ForecastModel protocol are used when present for name/feature metadata);
    alternatively pass a bare ``predict`` callable. ``times`` enables
    time-ordered drift blocks. ``scoring`` is a proper loss — see
    :mod:`scoring` for built-ins.
    """
    if predict is None:
        if model is None:
            raise ValueError("pass a fitted model or a predict callable")
        predict_fn = model_predict_fn(model)
    else:
        predict_fn = predict
    xx, yy = _validate_xy(x, y)
    n_rows, n_features = xx.shape
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    names = _resolve_names(model, feature_names, n_features)
    spec = resolve_score(scoring)

    name = model_name
    family = model_family
    if model is not None and (name is None or family is None):
        metadata = getattr(model, "metadata", None)
        if callable(metadata):
            try:
                meta = metadata()
                name = name or str(getattr(meta, "name", "unknown"))
                family = family or str(getattr(meta, "family", "unknown"))
            except (TypeError, ValueError, AttributeError):
                pass
    name = name or ("predict_callable" if model is None else type(model).__name__)
    family = family or "unknown"

    attribution = permutation_attribution(
        predict_fn,
        xx,
        yy,
        names,
        scoring=spec,
        n_repeats=n_repeats,
        seed=seed,
        max_rows=max_rows,
    )
    pd_k = int(pd_top_k) if pd_top_k is not None else min(int(top_k), n_features)
    top_names = [a.feature for a in attribution.top(pd_k)]
    curves = partial_dependence_top_k(predict_fn, xx, names, top_names, grid_points=grid_points)
    drift_result: AttributionDrift | None = None
    if drift and n_rows >= int(n_blocks) >= 2:
        drift_result = attribution_drift(
            predict_fn,
            xx,
            yy,
            times=times,
            feature_names=names,
            scoring=spec,
            n_blocks=n_blocks,
            n_repeats=n_repeats,
            seed=seed,
            top_k=min(int(top_k), n_features),
        )
    shap_result: AttributionResult | None = None
    warnings = list(attribution.warnings)
    if include_shap:
        try:
            shap_result = shap_attribution(predict_fn, xx, yy, names, scoring=spec, seed=seed)
        except ImportError as exc:
            warnings.append(str(exc))
    stamp = generated_at or datetime.now(UTC).isoformat()
    return ExplainabilityReport(
        model_name=name,
        model_family=family,
        scoring=spec.name,
        baseline_score=attribution.baseline_score,
        attribution=attribution,
        pd_curves=tuple(curves),
        drift=drift_result,
        feature_names=tuple(names),
        n_rows=int(n_rows),
        seed=int(seed),
        synthetic=bool(synthetic),
        generated_at=stamp,
        dataset_sha256=_dataset_digest(xx, yy),
        shap=shap_result,
        warnings=tuple(sorted(set(warnings))),
    )


def _fmt(value: float | None, digits: int = 5) -> str:
    if value is None or not math.isfinite(float(value)):
        return "n/a"
    return f"{float(value):.{digits}g}"


def _render_markdown(report: ExplainabilityReport) -> str:
    lines = [
        f"# Explainability report — {report.model_name}",
        "",
        f"- family: {report.model_family}",
        f"- scoring (proper loss, lower is better): {report.scoring}",
        f"- baseline score: {_fmt(report.baseline_score)}",
        f"- eval rows: {report.n_rows} | features: {len(report.feature_names)} | seed: {report.seed}",
        f"- dataset_sha256: {report.dataset_sha256}",
        f"- generated_at: {report.generated_at}",
        f"- attribution method: {report.attribution.method} (n_repeats={report.attribution.n_repeats})",
        "- claim: research_only",
    ]
    if report.synthetic:
        lines.append("- label: **SYNTHETIC DATA — correctness evidence only, not market evidence**")
    lines += ["", "## Permutation importances", ""]
    lines.append("| rank | feature | importance_mean | importance_std |")
    lines.append("|---:|---|---:|---:|")
    for a in report.attribution.attributions:
        lines.append(
            f"| {a.rank} | {a.feature} | {_fmt(a.importance_mean)} | {_fmt(a.importance_std)} |"
        )
    if report.shap is not None:
        lines += ["", "## SHAP importances (supplementary — mean |value|)", ""]
        lines.append("| rank | feature | mean_abs_shap |")
        lines.append("|---:|---|---:|")
        for a in report.shap.attributions:
            lines.append(f"| {a.rank} | {a.feature} | {_fmt(a.importance_mean)} |")
    if report.pd_curves:
        lines += ["", "## Partial dependence (top features)", ""]
        lines.append("Grid = empirical quantiles; values = mean prediction per grid point.")
        for curve in report.pd_curves:
            lines.append("")
            lines.append(f"### {curve.feature}")
            lines.append("")
            lines.append("| grid | mean_pred |")
            lines.append("|---:|---:|")
            for g, v in zip(curve.grid, curve.mean_curve, strict=True):
                lines.append(f"| {_fmt(g)} | {_fmt(v)} |")
    if report.drift is not None:
        drift = report.drift
        lines += ["", "## Attribution drift over time", ""]
        lines.append(f"- blocks: {drift.n_blocks} | flagged: **{drift.drift_flagged}**")
        lines.append(
            f"- min consecutive Spearman: {_fmt(drift.min_spearman)} "
            f"(floor {_fmt(drift.thresholds.get('spearman_floor'))})"
        )
        lines.append(
            f"- max consecutive JS divergence: {_fmt(drift.max_js)} "
            f"(ceiling {_fmt(drift.thresholds.get('js_ceiling'))})"
        )
        lines.append(
            f"- mean top-k overlap: {_fmt(drift.mean_topk_overlap)} "
            f"(floor {_fmt(drift.thresholds.get('topk_floor'))})"
        )
        lines += [
            "",
            "| block pair | spearman | js_div | topk_overlap |",
            "|---|---:|---:|---:|",
        ]
        for i, (rho, js, ov) in enumerate(
            zip(
                drift.consecutive_spearman,
                drift.consecutive_js,
                drift.consecutive_topk_overlap,
                strict=True,
            )
        ):
            lines.append(f"| {i}→{i + 1} | {_fmt(rho)} | {_fmt(js)} | {_fmt(ov)} |")
        lines += ["", "### Per-block top-3 features", ""]
        for block in drift.blocks:
            top3 = ", ".join(
                f"{a.feature}({_fmt(a.importance_mean, 3)})" for a in block.result.top(3)
            )
            lines.append(f"- block {block.block_index} ({block.n_rows} rows): {top3}")
        lines += ["", "### Max share change per feature", ""]
        lines.append("| feature | max share change |")
        lines.append("|---|---:|")
        for feat, change in sorted(
            drift.per_feature_max_share_change.items(), key=lambda kv: (-kv[1], kv[0])
        ):
            lines.append(f"| {feat} | {_fmt(change, 4)} |")
    if report.warnings:
        lines += ["", "## Warnings", ""]
        lines.extend(f"- {w}" for w in report.warnings)
    lines += [
        "",
        "---",
        "",
        "> Research-only explainability artifact. Importance = increase in a "
        "proper score when the feature is permuted; it is not a causal claim "
        "and not evidence of live profitability.",
        "",
    ]
    return "\n".join(lines)


def _fig_png_b64(fig: Any) -> str:
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    canvas = FigureCanvasAgg(fig)
    buf = io.BytesIO()
    cast(Callable[[io.BytesIO], None], canvas.print_png)(buf)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _importance_figure(report: ExplainabilityReport) -> str | None:
    from matplotlib.figure import Figure

    attrs = report.attribution.attributions
    if not attrs:
        return None
    fig = Figure(figsize=(7.0, max(2.5, 0.35 * len(attrs) + 1.2)), dpi=110)
    ax = fig.subplots()
    names = [a.feature for a in reversed(attrs)]
    means = [a.importance_mean for a in reversed(attrs)]
    stds = [a.importance_std for a in reversed(attrs)]
    ax.barh(np.arange(len(names)), means, xerr=stds, color="#2b6cb0", alpha=0.85)
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.axvline(0.0, color="#444", linewidth=0.8)
    ax.set_xlabel(f"Δ {report.scoring} (permuted − baseline)")
    ax.set_title("Permutation importance", fontsize=10)
    fig.tight_layout()
    return _fig_png_b64(fig)


def _pd_figure(report: ExplainabilityReport) -> str | None:
    from matplotlib.figure import Figure

    curves = report.pd_curves
    if not curves:
        return None
    ncols = min(4, len(curves))
    nrows = int(math.ceil(len(curves) / ncols))
    fig = Figure(figsize=(2.6 * ncols, 2.2 * nrows), dpi=110)
    axes = fig.subplots(nrows, ncols, squeeze=False)
    for idx, curve in enumerate(curves):
        ax = axes[idx // ncols][idx % ncols]
        ax.plot(
            curve.grid, curve.mean_curve, marker="o", markersize=3, linewidth=1.2, color="#2b6cb0"
        )
        ax.set_title(curve.feature, fontsize=8)
        ax.tick_params(labelsize=6)
        ax.grid(alpha=0.25)
    for idx in range(len(curves), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")
    fig.suptitle("Partial dependence (mean prediction)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _fig_png_b64(fig)


def _drift_figure(report: ExplainabilityReport) -> str | None:
    from matplotlib.figure import Figure

    drift = report.drift
    if drift is None or not drift.blocks:
        return None
    names = list(report.feature_names)
    shares = np.array(
        [_share_vector(block.result, names) for block in drift.blocks],
        dtype=float,
    )
    top = report.attribution.top(min(10, len(names)))
    row_idx = [names.index(a.feature) for a in top]
    sub = shares[:, row_idx].T
    fig = Figure(
        figsize=(max(4.0, 1.0 * len(drift.blocks) + 2.5), 0.45 * len(row_idx) + 1.6), dpi=110
    )
    ax = fig.subplots()
    image = ax.imshow(sub, aspect="auto", cmap="viridis", vmin=0.0)
    ax.set_xticks(np.arange(len(drift.blocks)))
    ax.set_xticklabels([f"b{b.block_index}" for b in drift.blocks], fontsize=8)
    ax.set_yticks(np.arange(len(row_idx)))
    ax.set_yticklabels([names[j] for j in row_idx], fontsize=8)
    ax.set_xlabel("time block")
    ax.set_title("Normalized importance share per block", fontsize=10)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="share")
    fig.tight_layout()
    return _fig_png_b64(fig)


def _share_vector(result: AttributionResult, names: list[str]) -> list[float]:
    vec = result.importance_vector(names)
    positive = np.clip(vec, 0.0, None) + 1e-12
    total = float(positive.sum())
    if total <= 0.0 or not np.isfinite(total):
        return [1.0 / len(names)] * len(names)
    return [float(v) for v in (positive / total).tolist()]


def _render_html(report: ExplainabilityReport) -> str:
    esc = html.escape
    charts = {
        "importance": _importance_figure(report),
        "pd": _pd_figure(report),
        "drift": _drift_figure(report),
    }
    banner = (
        '<div class="synthetic">SYNTHETIC DATA — correctness evidence only, '
        "not market evidence.</div>"
        if report.synthetic
        else ""
    )
    warnings = "".join(f"<li>{esc(w)}</li>" for w in report.warnings)
    warnings_html = f"<h2>Warnings</h2><ul>{warnings}</ul>" if warnings else ""

    rows = "".join(
        f"<tr><td>{a.rank}</td><td>{esc(a.feature)}</td>"
        f"<td class='num'>{_fmt(a.importance_mean)}</td>"
        f"<td class='num'>{_fmt(a.importance_std)}</td></tr>"
        for a in report.attribution.attributions
    )
    drift_rows = ""
    drift_summary = "<p>Drift not computed (disabled or too few rows).</p>"
    if report.drift is not None:
        d = report.drift
        drift_rows = "".join(
            f"<tr><td>{i}&rarr;{i + 1}</td><td class='num'>{_fmt(rho)}</td>"
            f"<td class='num'>{_fmt(js)}</td><td class='num'>{_fmt(ov)}</td></tr>"
            for i, (rho, js, ov) in enumerate(
                zip(
                    d.consecutive_spearman,
                    d.consecutive_js,
                    d.consecutive_topk_overlap,
                    strict=True,
                )
            )
        )
        flag = "yes" if d.drift_flagged else "no"
        drift_summary = (
            f"<p><b>drift flagged: {flag}</b> — min Spearman {_fmt(d.min_spearman)} "
            f"(floor {_fmt(d.thresholds.get('spearman_floor'))}), max JS "
            f"{_fmt(d.max_js)} (ceiling {_fmt(d.thresholds.get('js_ceiling'))}), "
            f"mean top-k overlap {_fmt(d.mean_topk_overlap)} "
            f"(floor {_fmt(d.thresholds.get('topk_floor'))})</p>"
        )
    pd_note = ""
    if report.shap is not None:
        shap_rows = "".join(
            f"<tr><td>{a.rank}</td><td>{esc(a.feature)}</td>"
            f"<td class='num'>{_fmt(a.importance_mean)}</td></tr>"
            for a in report.shap.attributions
        )
        pd_note = (
            "<h2>SHAP importances (supplementary)</h2>"
            "<table><tr><th>rank</th><th>feature</th><th>mean_abs_shap</th></tr>"
            f"{shap_rows}</table>"
        )

    def img(key: str) -> str:
        data = charts[key]
        return f'<img src="data:image/png;base64,{data}" alt="{key} chart"/>' if data else ""

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"/>
<title>Explainability — {esc(report.model_name)}</title>
<style>
body {{ font-family: -apple-system, 'Segoe UI', sans-serif; margin: 2rem auto; max-width: 960px; color: #1a202c; }}
h1 {{ font-size: 1.5rem; }} h2 {{ font-size: 1.15rem; margin-top: 2rem; }}
table {{ border-collapse: collapse; font-size: 0.85rem; }}
th, td {{ border: 1px solid #cbd5e0; padding: 3px 8px; text-align: left; }}
th {{ background: #edf2f7; }}
td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
.meta {{ color: #4a5568; font-size: 0.85rem; }}
.synthetic {{ background: #fefcbf; border: 1px solid #d69e2e; padding: 0.6rem; margin: 1rem 0; font-weight: 600; }}
img {{ max-width: 100%; height: auto; border: 1px solid #e2e8f0; }}
.footer {{ margin-top: 2.5rem; font-size: 0.8rem; color: #718096; border-top: 1px solid #e2e8f0; padding-top: 0.8rem; }}
</style></head><body>
<h1>Explainability report — {esc(report.model_name)}</h1>
{banner}
<p class="meta">
family: {esc(report.model_family)} · scoring (proper loss, lower=better): {esc(report.scoring)}
· baseline {_fmt(report.baseline_score)} · rows {report.n_rows} · features {len(report.feature_names)}
· seed {report.seed} · method {esc(report.attribution.method)}<br/>
dataset_sha256: <code>{esc(report.dataset_sha256)}</code> · generated_at {esc(report.generated_at)}
· claim: research_only
</p>
<h2>Permutation importances</h2>
{img("importance")}
<table><tr><th>rank</th><th>feature</th><th>importance_mean</th><th>importance_std</th></tr>
{rows}</table>
{pd_note}
<h2>Partial dependence</h2>
{img("pd")}
<h2>Attribution drift over time</h2>
{drift_summary}
{img("drift")}
<table><tr><th>block pair</th><th>spearman</th><th>js_div</th><th>topk_overlap</th></tr>
{drift_rows}</table>
{warnings_html}
<div class="footer">Research-only explainability artifact. Importance = increase in a proper
score when the feature is permuted; not a causal claim and not evidence of live profitability.</div>
</body></html>
"""


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as tmp:
            temporary = Path(tmp.name)
            tmp.write(content)
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _write_sidecar(path: Path) -> None:
    _atomic_write(
        path.with_name(f"{path.name}.sha256"),
        (hash_file(path) + "\n").encode("ascii"),
    )


def write_report(report: ExplainabilityReport, out_dir: Path) -> dict[str, Path]:
    """Persist Markdown, self-contained HTML, and machine JSON + sha256 sidecars."""
    out_dir = Path(out_dir)
    paths = {
        "markdown": out_dir / "explainability.md",
        "html": out_dir / "explainability.html",
        "json": out_dir / "explainability.json",
    }
    payload = (
        json.dumps(
            json.loads(canonical_json_bytes(report.to_dict())),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    _atomic_write(paths["markdown"], report.to_markdown().encode("utf-8"))
    _atomic_write(paths["html"], report.to_html().encode("utf-8"))
    _atomic_write(paths["json"], payload)
    for path in paths.values():
        _write_sidecar(path)
    return paths
