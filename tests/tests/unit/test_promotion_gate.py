from quant_fund.config.models import PromotionConfig
from quant_fund.registry.mlflow_store import promotion_decision


def test_promotion_rejects_coercion_shaped_fold_and_stability_evidence() -> None:
    base = {
        "evidence_complete": True,
        "data_source": "file",
        "mean_ic": 0.5,
        "net_spread": 0.5,
        "turnover": 0.1,
    }
    for fold_count in (True, "5", 5.5, float("nan"), float("inf")):
        result = promotion_decision(
            {**base, "n_folds": fold_count, "fold_ic_stability": 1.0},
            PromotionConfig(min_folds=2),
            leakage_ok=True,
        )
        assert result["promote"] is False
        assert "insufficient_folds" in result["reasons"]

    result = promotion_decision(
        {**base, "n_folds": 5, "fold_ic_stability": float("inf")},
        PromotionConfig(min_folds=2),
        leakage_ok=True,
    )
    assert result["promote"] is False
    assert "fold_ic_unstable" in result["reasons"]


def test_promotion_malformed_metric_strings_fail_closed_without_raising() -> None:
    metrics = {
        "evidence_complete": True,
        "data_source": "file",
        "mean_ic": "not-a-number",
        "net_spread": "also-not-a-number",
        "turnover": object(),
        "n_folds": 5,
    }
    result = promotion_decision(metrics, PromotionConfig(), leakage_ok=True)
    assert result["promote"] is False
    assert any(reason.startswith("missing_metrics:") for reason in result["reasons"])


def test_promotion_gate_fails_closed_on_missing_evidence() -> None:
    result = promotion_decision({}, PromotionConfig(), leakage_ok=True)
    assert result["promote"] is False
    assert "evidence_incomplete" in result["reasons"]
    assert any(reason.startswith("missing_metrics:") for reason in result["reasons"])


def test_synthetic_evidence_can_never_promote() -> None:
    metrics = {
        "evidence_complete": True,
        "data_source": "SYNTHETIC",
        "mean_ic": 1.0,
        "net_spread": 1.0,
        "turnover": 0.1,
    }
    result = promotion_decision(metrics, PromotionConfig(), leakage_ok=True)
    assert result["promote"] is False
    assert "synthetic_evidence_not_promotable" in result["reasons"]


def test_promotion_insufficient_folds_fails_closed() -> None:
    metrics = {
        "evidence_complete": True,
        "data_source": "file",
        "mean_ic": 0.5,
        "net_spread": 0.5,
        "turnover": 0.1,
        "n_folds": 1,
    }
    result = promotion_decision(metrics, PromotionConfig(min_folds=2), leakage_ok=True)
    assert result["promote"] is False
    assert "insufficient_folds" in result["reasons"]


def test_promotion_fold_ic_unstable_fails_closed() -> None:
    metrics = {
        "evidence_complete": True,
        "data_source": "file",
        "mean_ic": 0.5,
        "net_spread": 0.5,
        "turnover": 0.1,
        "n_folds": 5,
        "fold_ic_stability": 0.1,
    }
    result = promotion_decision(
        metrics, PromotionConfig(min_folds=2, min_fold_ic_stability=0.5), leakage_ok=True
    )
    assert result["promote"] is False
    assert "fold_ic_unstable" in result["reasons"]


def test_promotion_nonfinite_metrics_fail_closed() -> None:
    metrics = {
        "evidence_complete": True,
        "data_source": "file",
        "mean_ic": float("nan"),
        "net_spread": 0.5,
        "turnover": 0.1,
        "n_folds": 5,
    }
    result = promotion_decision(metrics, PromotionConfig(), leakage_ok=True)
    assert result["promote"] is False
    assert any(r.startswith("missing_metrics:") for r in result["reasons"])


def test_promotion_perfect_non_synthetic_can_promote() -> None:
    metrics = {
        "evidence_complete": True,
        "data_source": "file",
        "mean_ic": 0.5,
        "net_spread": 0.5,
        "turnover": 0.1,
        "n_folds": 5,
        "fold_ic_stability": 1.0,
        "run_id": "run-xyz",
    }
    result = promotion_decision(metrics, PromotionConfig(min_folds=2), leakage_ok=True)
    assert result["promote"] is True
    assert result["reasons"] == []


def test_promotion_leakage_failed_fails_closed() -> None:
    metrics = {
        "evidence_complete": True,
        "data_source": "file",
        "mean_ic": 0.5,
        "net_spread": 0.5,
        "turnover": 0.1,
        "n_folds": 5,
    }
    result = promotion_decision(metrics, PromotionConfig(), leakage_ok=False)
    assert result["promote"] is False
    assert "leakage_failed" in result["reasons"]


def test_promotion_rejects_truthy_string_leakage_flags() -> None:
    metrics = {
        "evidence_complete": True,
        "data_source": "file",
        "mean_ic": 0.5,
        "net_spread": 0.5,
        "turnover": 0.1,
        "n_folds": 5,
    }
    for flag in ("false", "true", 1, object()):
        result = promotion_decision(metrics, PromotionConfig(), leakage_ok=flag)  # type: ignore[arg-type]
        assert result["promote"] is False
        assert "leakage_failed" in result["reasons"]


def test_promotion_threshold_edges_fail_closed() -> None:
    base = {
        "evidence_complete": True,
        "data_source": "file",
        "mean_ic": 0.5,
        "net_spread": 0.5,
        "turnover": 0.1,
        "n_folds": 5,
    }
    cfg = PromotionConfig(min_mean_ic=0.6, min_cost_adjusted_spread=0.0, max_turnover=5.0)
    r = promotion_decision(base, cfg, leakage_ok=True)
    assert r["promote"] is False
    assert "ic_below_min" in r["reasons"]

    cfg2 = PromotionConfig(min_mean_ic=0.0, min_cost_adjusted_spread=0.9, max_turnover=5.0)
    r2 = promotion_decision(base, cfg2, leakage_ok=True)
    assert r2["promote"] is False
    assert "spread_below_min" in r2["reasons"]

    cfg3 = PromotionConfig(min_mean_ic=0.0, min_cost_adjusted_spread=0.0, max_turnover=0.05)
    r3 = promotion_decision(base, cfg3, leakage_ok=True)
    assert r3["promote"] is False
    assert "turnover_high" in r3["reasons"]


def test_promotion_n_ic_folds_alias_and_bool_metric_fail_closed() -> None:
    # n_ic_folds used when n_folds absent
    metrics = {
        "evidence_complete": True,
        "data_source": "file",
        "mean_ic": 0.5,
        "net_spread": 0.5,
        "turnover": 0.1,
        "n_ic_folds": 1,  # below default min_folds=2
    }
    result = promotion_decision(metrics, PromotionConfig(), leakage_ok=True)
    assert result["promote"] is False
    assert "insufficient_folds" in result["reasons"]

    # bool is not finite numeric evidence
    poisoned = {
        "evidence_complete": True,
        "data_source": "file",
        "mean_ic": True,  # bool masquerading as 1
        "net_spread": 0.5,
        "turnover": 0.1,
        "n_folds": 5,
    }
    result2 = promotion_decision(poisoned, PromotionConfig(), leakage_ok=True)
    assert result2["promote"] is False
    assert any(r.startswith("missing_metrics:") and "mean_ic" in r for r in result2["reasons"])


def test_promotion_fold_stability_nan_and_garbage_fail_closed() -> None:
    metrics = {
        "evidence_complete": True,
        "data_source": "file",
        "mean_ic": 0.5,
        "net_spread": 0.5,
        "turnover": 0.1,
        "n_folds": 5,
        "fold_ic_stability": float("nan"),
    }
    result = promotion_decision(
        metrics, PromotionConfig(min_folds=2, min_fold_ic_stability=0.5), leakage_ok=True
    )
    assert result["promote"] is False
    assert "fold_ic_unstable" in result["reasons"]

    metrics2 = dict(metrics)
    metrics2["fold_ic_stability"] = "not-a-number"
    result2 = promotion_decision(
        metrics2, PromotionConfig(min_folds=2, min_fold_ic_stability=0.5), leakage_ok=True
    )
    assert result2["promote"] is False
    assert "fold_ic_unstable" in result2["reasons"]


def test_promotion_synthetic_flag_alone_blocks() -> None:
    metrics = {
        "evidence_complete": True,
        "synthetic": True,  # no SYNTHETIC data_source string
        "data_source": "file",
        "mean_ic": 0.5,
        "net_spread": 0.5,
        "turnover": 0.1,
        "n_folds": 5,
        "fold_ic_stability": 1.0,
    }
    result = promotion_decision(metrics, PromotionConfig(min_folds=2), leakage_ok=True)
    assert result["promote"] is False
    assert "synthetic_evidence_not_promotable" in result["reasons"]


def test_promotion_rejects_malformed_synthetic_flag() -> None:
    metrics = {
        "evidence_complete": True,
        "data_source": "file",
        "synthetic": "false",
        "mean_ic": 0.5,
        "net_spread": 0.5,
        "turnover": 0.1,
        "n_folds": 5,
    }
    result = promotion_decision(metrics, PromotionConfig(), leakage_ok=True)
    assert result["promote"] is False
    assert "synthetic_flag_invalid" in result["reasons"]
