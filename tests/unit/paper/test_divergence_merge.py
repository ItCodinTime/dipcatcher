"""_merge_divergence_summary resume-weighting contracts.

A step where the paper loop halts on nonpositive NAV still emits a
divergence sample, so the cumulative step counter alone cannot recover how
many samples a prior run's mean covered. The receipt carries the exact
count (``n_divergence_samples``); legacy receipts fall back to ``n_steps``.
"""

from __future__ import annotations

import pytest

from quant_fund.paper.loop import _merge_divergence_summary


def test_prior_mean_weighted_by_recorded_sample_count() -> None:
    # Prior run: 11 samples (10 steps + the ruin bar's sample), mean 0.2.
    prior = {
        "mean_l1": 0.2,
        "max_l1": 0.4,
        "n_steps": 10,
        "n_divergence_samples": 11,
    }
    mean, maximum = _merge_divergence_summary(prior, [0.5, 0.6], total_steps=12)
    assert mean == pytest.approx((0.2 * 11 + 0.5 + 0.6) / 13)
    assert maximum == pytest.approx(0.6)


def test_legacy_receipt_falls_back_to_n_steps() -> None:
    prior = {"mean_l1": 0.2, "max_l1": 0.4, "n_steps": 10}
    mean, _ = _merge_divergence_summary(prior, [0.5, 0.6], total_steps=12)
    assert mean == pytest.approx((0.2 * 10 + 0.5 + 0.6) / 12)


def test_break_iteration_does_not_underweight_prior_mean() -> None:
    # total_steps - len(current) would yield 9 here (the resumed run's ruin
    # bar emitted a sample without advancing step); n_steps pins 10.
    prior = {"mean_l1": 0.2, "max_l1": 0.4, "n_steps": 10}
    mean, _ = _merge_divergence_summary(prior, [0.5, 0.6, 0.7], total_steps=12)
    assert mean == pytest.approx((0.2 * 10 + 0.5 + 0.6 + 0.7) / 13)


def test_no_prior_uses_current_only() -> None:
    mean, maximum = _merge_divergence_summary(None, [0.1, 0.3], total_steps=2)
    assert mean == pytest.approx(0.2)
    assert maximum == pytest.approx(0.3)


def test_all_nan_current_inherits_prior() -> None:
    prior = {"mean_l1": 0.2, "max_l1": 0.4, "n_steps": 10}
    mean, maximum = _merge_divergence_summary(prior, [float("nan"), float("nan")], total_steps=12)
    assert mean == pytest.approx(0.2)
    assert maximum == pytest.approx(0.4)


def test_promotion_receipt_roundtrips_sample_count() -> None:
    from quant_fund.paper.ledger import (
        promotion_dry_run,
        validate_promotion_dry_run_receipt,
    )

    receipt = promotion_dry_run(
        run_id="rt",
        mean_l1=0.1,
        max_l1=0.2,
        n_steps=5,
        n_divergence_samples=6,
        champion_nav=1.0,
        shadow_gross=1.0,
        max_mean_l1=0.25,
        min_steps=5,
        data_source="SYNTHETIC",
    )
    assert receipt["n_divergence_samples"] == 6
    assert validate_promotion_dry_run_receipt(receipt) == []


def test_promotion_receipt_rejects_bad_sample_count() -> None:
    import pytest as _pytest

    from quant_fund.paper.ledger import promotion_dry_run

    with _pytest.raises(ValueError, match="n_divergence_samples"):
        promotion_dry_run(
            mean_l1=0.1,
            max_l1=0.2,
            n_steps=5,
            n_divergence_samples=-1,
            champion_nav=1.0,
            shadow_gross=1.0,
            max_mean_l1=0.25,
            min_steps=5,
            data_source="SYNTHETIC",
        )
