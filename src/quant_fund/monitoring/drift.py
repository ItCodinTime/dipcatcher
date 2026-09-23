"""Feature / prediction drift. PSI is optional, not the only metric."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from quant_fund.models.base import load_joblib_artifact


def drift_report(
    reference: NDArray[np.float64],
    current: NDArray[np.float64],
    *,
    psi_threshold: float = 0.25,
    mean_shift_threshold: float | None = None,
    bins: int = 10,
) -> dict[str, float | bool | str]:
    """Return an auditable drift decision with explicit sample sufficiency.

    PSI is only meaningful when both windows support the requested bins. A
    short or empty window therefore produces ``insufficient_data`` rather
    than a false all-clear. Mean shift is optional and remains a diagnostic.
    """
    r = np.asarray(reference, dtype=float)
    c = np.asarray(current, dtype=float)
    n_reference = int(np.isfinite(r).sum())
    n_current = int(np.isfinite(c).sum())
    value = psi(r, c, bins=bins)
    shift = mean_shift(r, c)
    sufficient = bool(np.isfinite(value))
    psi_alert = bool(sufficient and value >= float(psi_threshold))
    mean_alert = bool(
        mean_shift_threshold is not None
        and np.isfinite(shift)
        and abs(shift) >= float(mean_shift_threshold)
    )
    return {
        "status": "alert"
        if psi_alert or mean_alert
        else ("ok" if sufficient else "insufficient_data"),
        "psi": float(value),
        "mean_shift": float(shift),
        "psi_threshold": float(psi_threshold),
        "n_reference": float(n_reference),
        "n_current": float(n_current),
        "sufficient_data": sufficient,
        "alert": psi_alert or mean_alert,
    }


def psi(reference: NDArray[np.float64], current: NDArray[np.float64], bins: int = 10) -> float:
    if int(bins) < 2:
        raise ValueError("bins must be >= 2")
    r = np.asarray(reference, dtype=float)
    c = np.asarray(current, dtype=float)
    r = r[np.isfinite(r)]
    c = c[np.isfinite(c)]
    if r.size < bins or c.size < bins:
        return float("nan")
    edges = np.quantile(r, np.linspace(0, 1, bins + 1))
    # Keep current-window outliers in the outer bins instead of dropping them.
    edges[0] = min(edges[0], float(np.min(c)))
    edges[-1] = max(edges[-1], float(np.max(c)))
    if np.any(np.diff(edges) <= 0.0):
        edges = np.linspace(
            float(min(np.min(r), np.min(c))), float(max(np.max(r), np.max(c))), bins + 1
        )
    edges[0] -= 1e-12
    edges[-1] += 1e-12
    pr, _ = np.histogram(r, bins=edges)
    pc, _ = np.histogram(c, bins=edges)
    pr = np.clip(pr / pr.sum(), 1e-6, 1.0)
    pc = np.clip(pc / pc.sum(), 1e-6, 1.0)
    return float(np.sum((pc - pr) * np.log(pc / pr)))


def mean_shift(reference: NDArray[np.float64], current: NDArray[np.float64]) -> float:
    r = np.asarray(reference, dtype=float)
    c = np.asarray(current, dtype=float)
    r = r[np.isfinite(r)]
    c = c[np.isfinite(c)]
    if r.size == 0 or c.size == 0:
        return float("nan")
    return float(np.mean(c) - np.mean(r))


def model_health_report(
    reference_features: Mapping[str, NDArray[np.float64]],
    current_features: Mapping[str, NDArray[np.float64]],
    *,
    artifact_paths: Mapping[str, str | Path] | None = None,
    calibration_windows: tuple[
        NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
    ]
    | None = None,
    calibration_alert_delta: float = 0.05,
    psi_threshold: float = 0.25,
    min_features: int = 1,
) -> dict[str, Any]:
    """Return one fail-closed artifact and feature-health report.

    This is an operational diagnostic only: it does not promote or halt a
    model. Missing features, invalid artifacts, and insufficient samples are
    explicit failures rather than being treated as a clean report.
    """
    if min_features < 1:
        raise ValueError("min_features must be positive")
    feature_names = sorted(set(reference_features) | set(current_features))
    feature_reports: dict[str, Any] = {}
    errors: list[str] = []
    for name in feature_names:
        if name not in reference_features or name not in current_features:
            errors.append(f"missing_feature:{name}")
            continue
        report = drift_report(
            reference_features[name], current_features[name], psi_threshold=psi_threshold
        )
        feature_reports[name] = report
        if report["status"] == "insufficient_data":
            errors.append(f"insufficient_feature_data:{name}")
    artifact_reports: dict[str, Any] = {}
    for name, raw_path in (artifact_paths or {}).items():
        path = Path(raw_path)
        try:
            load_joblib_artifact(path)
            artifact_reports[name] = {"status": "ok", "path": str(path)}
        except (OSError, ValueError, TypeError) as exc:
            artifact_reports[name] = {"status": "invalid", "path": str(path)}
            errors.append(f"invalid_artifact:{name}:{type(exc).__name__}")
    calibration: dict[str, Any] | None = None
    if calibration_windows is not None:
        reference_prob, reference_label, current_prob, current_label = calibration_windows
        rp = np.asarray(reference_prob, dtype=float)
        ry = np.asarray(reference_label, dtype=float)
        cp = np.asarray(current_prob, dtype=float)
        cy = np.asarray(current_label, dtype=float)
        if rp.size != ry.size or cp.size != cy.size:
            errors.append("calibration_window_length_mismatch")
        else:
            rmask = np.isfinite(rp) & np.isfinite(ry)
            cmask = np.isfinite(cp) & np.isfinite(cy)
            if not rmask.any() or not cmask.any():
                errors.append("insufficient_calibration_data")
            else:
                reference_brier = float(np.mean((rp[rmask] - ry[rmask]) ** 2))
                current_brier = float(np.mean((cp[cmask] - cy[cmask]) ** 2))
                delta = current_brier - reference_brier
                calibration = {
                    "reference_brier": reference_brier,
                    "current_brier": current_brier,
                    "delta": delta,
                    "alert": bool(delta >= calibration_alert_delta),
                }
    alerts = [name for name, report in feature_reports.items() if report.get("alert")]
    if calibration is not None and calibration.get("alert"):
        alerts.append("calibration")
    if len(feature_reports) < min_features:
        errors.append("insufficient_feature_count")
    status = "invalid" if errors else ("alert" if alerts else "ok")
    return {
        "status": status,
        "features": feature_reports,
        "artifacts": artifact_reports,
        "calibration": calibration,
        "alerts": alerts,
        "errors": errors,
        "research_only": True,
        "live_pnl_claim": False,
    }
