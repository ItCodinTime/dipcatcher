"""Cached ranker, policy, and probability-calibrator artifacts.

Split out of the original module. Import the parent path; it re-exports these names.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from quant_fund.config.models import AppConfig
from quant_fund.models.base import JoblibMixin, load_joblib_artifact
from quant_fund.models.calibration import ProbabilityCalibrator
from quant_fund.utils.hashing import hash_file

from .state import _RANKER_CACHE, _RL_POLICY_CACHE


def _ranker_artifact_path(config: AppConfig) -> Path:
    return Path(config.data.root) / "metadata" / "ranker_ridge.joblib"


def _joblib_artifact_digest(path: Path) -> str:
    """SHA-256 of joblib artifact bytes. Mtime is not identity."""
    return hash_file(path)


def _load_ranker_cached(config: AppConfig):
    """Load the strongest available supervised ranker artifact.

    Candidate artifacts are searched in priority order; cache identity is
    artifact bytes, not mtime, so an in-place replacement that preserves mtime
    (Windows timestamp resolution, ``os.utime``, or a same-size rewrite) must
    not reuse a stale ranker for ``forecast_asof`` / ``optimize_asof``. Missing
    artifacts return None so the momentum heuristic remains the explicit
    no-model path.
    """
    root = Path(config.data.root) / "metadata"
    rank_path = next(
        (
            candidate
            for candidate in (
                root / "ranker_auto.joblib",
                root / "ranker_ensemble.joblib",
                root / "ranker_neural.joblib",
                root / "ranker_ridge.joblib",
                root / "ranker_elasticnet.joblib",
                root / "ranker_lambdarank.joblib",
                root / "ranker_xendcg.joblib",
                root / "ranker_xgboost.joblib",
                root / "ranker_lightgbm.joblib",
            )
            if candidate.exists()
        ),
        _ranker_artifact_path(config),
    )
    if not rank_path.exists():
        return None
    digest = _joblib_artifact_digest(rank_path)
    key = (str(rank_path.resolve()), digest)
    cached = _RANKER_CACHE.get(key)
    if cached is not None:
        return cached
    # The artifact name is a routing hint only: the serialized object may be
    # any ranking implementation. JoblibMixin.load retains checksum
    # verification while avoiding a Ridge-only type assertion.
    model = JoblibMixin.load(rank_path)
    _RANKER_CACHE[key] = model
    if len(_RANKER_CACHE) > 8:
        oldest = next(iter(_RANKER_CACHE))
        _RANKER_CACHE.pop(oldest, None)
    return model


def _load_rl_cached(config: AppConfig):
    """Load the strongest available persisted RL policy and feature contract."""
    root = Path(config.data.root) / "metadata"
    path = next(
        (
            candidate
            for candidate in (
                root / "rl_auto.joblib",
                root / "rl_policy_gradient.joblib",
                root / "rl_quantile_thompson.joblib",
                root / "rl_linucb.joblib",
                root / "rl_thompson.joblib",
            )
            if candidate.exists()
        ),
        None,
    )
    if path is None:
        return None
    key = (str(path.resolve()), path.stat().st_mtime)
    cached = _RL_POLICY_CACHE.get(key)
    if cached is not None:
        return cached
    artifact = load_joblib_artifact(path)
    if not isinstance(artifact, dict) or "policy" not in artifact or "features" not in artifact:
        raise ValueError("RL artifact is malformed: expected policy and features")
    policy = artifact["policy"]
    features = artifact["features"]
    if (
        not (hasattr(policy, "scores") or hasattr(policy, "predict"))
        or not isinstance(features, list)
        or not features
    ):
        raise ValueError("RL artifact has an invalid policy or feature contract")
    artifact_name = str(artifact.get("policy_name", path.stem.removeprefix("rl_"))).upper()
    loaded = (policy, [str(feature) for feature in features], f"RL_{artifact_name}")
    _RL_POLICY_CACHE[key] = loaded
    if len(_RL_POLICY_CACHE) > 8:
        oldest = next(iter(_RL_POLICY_CACHE))
        _RL_POLICY_CACHE.pop(oldest, None)
    return loaded


def _load_probability_calibrator(
    config: AppConfig, *, asof: datetime | None = None
) -> ProbabilityCalibrator:
    """Load an explicitly requested calibrator and validate its score identity."""
    name = str(config.fusion.probability_calibrator)
    path = Path(config.data.root) / "metadata" / f"calibrator_{name}.joblib"
    if name == "auto":
        path = Path(config.data.root) / "metadata" / "calibrator_auto.joblib"
    if not path.is_file():
        raise ValueError(f"probability calibration is enabled but artifact is missing: {path}")
    calibrator = JoblibMixin.load(path)
    if not isinstance(calibrator, ProbabilityCalibrator) or not calibrator.fitted:
        raise ValueError("probability calibrator artifact is invalid or unfitted")
    if calibrator.score_feature != "cs_pct_mom_20":
        raise ValueError("probability calibrator score identity mismatch: expected cs_pct_mom_20")
    expected_label = config.fusion.probability_calibration_label
    if expected_label is not None and calibrator.label != expected_label:
        raise ValueError("probability calibrator label identity mismatch")
    expected_horizon = config.fusion.probability_calibration_horizon
    if expected_horizon is not None and getattr(calibrator, "horizon", None) != expected_horizon:
        raise ValueError("probability calibrator horizon identity mismatch")
    for window_field in ("fit_start", "fit_end", "oos_start", "oos_end"):
        value = getattr(calibrator, window_field, None)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"probability calibrator {window_field} is missing")
    max_age = config.fusion.probability_calibration_max_age_days
    if asof is not None and max_age is not None:
        try:
            fit_end = datetime.fromisoformat(str(calibrator.fit_end).replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("probability calibrator fit_end is not parseable") from exc
        left = fit_end.date()
        right = asof.date()
        if (right - left).days < 0 or (right - left).days > max_age:
            raise ValueError("probability calibrator is stale for forecast asof")
    return calibrator


def _paper_challenger_stamp(
    config: AppConfig,
    x: NDArray[np.float64],
    ridge_scores: NDArray[np.float64],
    dates: NDArray[np.float64] | None = None,
    ids: NDArray[np.float64] | None = None,
) -> tuple[str, dict[str, float | str]]:
    """List fitted paper rankers. Spearman vs ridge only. Never blended into alpha."""
    import joblib
    from scipy.stats import spearmanr

    from quant_fund.pipeline.train import _predict_ranker

    present: list[str] = []
    extra: dict[str, float | str] = {"paper_challengers_blend": 0.0}
    root = Path(config.data.root) / "metadata"
    for name in config.train.paper_rankers:
        path = root / f"ranker_{name}.joblib"
        if not path.is_file():
            continue
        try:
            model = joblib.load(path)
            pred = np.asarray(_predict_ranker(model, name, x, dates, ids), dtype=float)
        except (TypeError, ValueError, OSError, AttributeError):
            continue
        present.append(name)
        if pred.shape[0] == ridge_scores.shape[0] and pred.size >= 4:
            rho = spearmanr(ridge_scores, pred, nan_policy="omit").correlation
            if rho is not None and np.isfinite(rho):
                extra[f"paper_{name}_spearman_vs_ridge"] = float(rho)
    extra["paper_challengers"] = ",".join(present) if present else "none"
    return f"paper_challengers={extra['paper_challengers']}", extra
