"""Impossible-fit screening: statistical leakage canaries (warnings, never errors)."""

from __future__ import annotations

import json

from quant_fund.research.impossible_fit import (
    IMPOSSIBLE_FIT_MIN_OBS,
    impossible_fit_flags,
    impossible_fit_scan,
)
from quant_fund.research.receipt_v2 import verify_receipt_payload


class TestImpossibleFitFlags:
    def test_clean_metrics_flag_nothing(self) -> None:
        metrics = {"mean_pinball": 0.0123, "rank_ic_mean": 0.031, "n_obs": 500}
        assert impossible_fit_flags(metrics) == []

    def test_exact_zero_pinball_flags(self) -> None:
        metrics = {"mean_pinball": 0.0, "n_obs": 300}
        assert "exact_zero:mean_pinball" in impossible_fit_flags(metrics)

    def test_exact_zero_crps_and_qlike(self) -> None:
        assert impossible_fit_flags({"crps": 0.0}, n_obs=200) == ["exact_zero:crps"]
        assert impossible_fit_flags({"qlike_mean": 0.0, "n": 100}) == ["exact_zero:qlike_mean"]

    def test_near_perfect_ic_flags(self) -> None:
        metrics = {"oos_rank_ic": 0.99999, "n_obs": 400}
        assert impossible_fit_flags(metrics) == ["near_perfect_correlation:oos_rank_ic"]

    def test_perfect_negative_ic_flags(self) -> None:
        flags = impossible_fit_flags({"spearman_ic": -1.0, "n": 60})
        assert flags == ["near_perfect_correlation:spearman_ic"]

    def test_realistic_ic_does_not_flag(self) -> None:
        assert impossible_fit_flags({"rank_ic": 0.999, "n_obs": 400}) == []

    def test_perfect_auc_flags(self) -> None:
        assert impossible_fit_flags({"roc_auc": 1.0, "n_obs": 100}) == ["perfect_auc:roc_auc"]

    def test_zero_violations_needs_high_n(self) -> None:
        assert impossible_fit_flags({"n_violations": 0, "n_obs": 30}) == []
        flags = impossible_fit_flags({"n_violations": 0, "n_obs": 250})
        assert flags == ["zero_violations_high_n:n_violations"]

    def test_small_n_suppresses_exact_zero(self) -> None:
        # 12-row shard scoring exactly zero is a degenerate shard, not a leak.
        assert impossible_fit_flags({"pinball": 0.0, "n_obs": 12}) == []

    def test_low_n_still_flags_when_n_unknown(self) -> None:
        # No n detected: conservative default still fires on exact zero.
        assert impossible_fit_flags({"pinball": 0.0}) == ["exact_zero:pinball"]

    def test_fragile_tokens_do_not_match_substrings(self) -> None:
        # `ic` must not match inside `price`, `metrics`, `static`; a price of
        # 1.0 (a pegged asset) is not a correlation of 1.0.
        metrics = {"price": 1.0, "metric_scale": 1.0, "n_obs": 500}
        assert impossible_fit_flags(metrics) == []

    def test_bool_is_not_a_number(self) -> None:
        assert impossible_fit_flags({"pinball": True, "n": 500}) == []

    def test_nan_ic_does_not_flag(self) -> None:
        assert impossible_fit_flags({"ic": float("nan"), "n": 500}) == []


class TestImpossibleFitScan:
    def test_walks_nested_leaderboard(self) -> None:
        payload = {
            "kind": "distribution_fleet_eval",
            "payload": {
                "leaderboard": [
                    {"head": "ridge", "mean_pinball": 0.011, "n": 300},
                    {"head": "oracle", "mean_pinball": 0.0, "n": 300},
                ]
            },
        }
        flags = impossible_fit_scan(payload)
        assert flags == ["payload.leaderboard[1]:exact_zero:mean_pinball"]

    def test_root_level_metrics(self) -> None:
        flags = impossible_fit_scan({"rank_ic": 1.0, "n": 100})
        assert flags == ["$:near_perfect_correlation:rank_ic"]

    def test_strings_and_none_ignored(self) -> None:
        payload = {"a": {"b": [{"pinball": "0.0"}, None, {"c": []}]}}
        assert impossible_fit_scan(payload) == []

    def test_deterministic_order(self) -> None:
        payload = {"x": {"pinball": 0.0, "n": 100}, "y": [{"brier": 0.0, "n": 100}]}
        assert impossible_fit_scan(payload) == impossible_fit_scan(payload)


class TestVerifyReceiptWarnings:
    def test_warnings_field_present_and_independent_of_validity(self) -> None:
        result = verify_receipt_payload({"unrelated": True})
        assert result["valid"] is False  # unsealed
        assert result["warnings"] == []

    def test_suspicious_scores_warn_without_invalidating(self, tmp_path) -> None:
        import hashlib

        from quant_fund.utils.hashing import canonical_json_bytes

        body = {"kind": "demo", "scores": {"pinball": 0.0, "n": 400}}
        sealed = dict(body)
        sealed["receipt_sha256"] = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
        result = verify_receipt_payload(json.loads(json.dumps(sealed)))
        assert result["valid"] is True
        assert result["warnings"] == ["scores:exact_zero:pinball"]

    def test_honest_scores_warn_nothing(self) -> None:
        import hashlib

        from quant_fund.utils.hashing import canonical_json_bytes

        body = {"scores": {"pinball": 0.02, "n": 400}}
        sealed = dict(body)
        sealed["receipt_sha256"] = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
        result = verify_receipt_payload(json.loads(json.dumps(sealed)))
        assert result["valid"] is True
        assert result["warnings"] == []


def test_min_obs_constant_is_sane() -> None:
    assert IMPOSSIBLE_FIT_MIN_OBS >= 20
