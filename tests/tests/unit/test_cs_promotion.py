"""The CS promotion rule rejects a merely less-negative IC."""

from __future__ import annotations

from quant_fund.hedge_lab.promotion import clears_cs_promotion, positive_mean_ic
from quant_fund.research.sota_protocol import promotion_decision


def test_negative_ic_does_not_clear_even_when_it_beats_ridge() -> None:
    # 20-day lambdarank pattern: IC still negative, every relative gate green.
    assert positive_mean_ic(-0.008) is False
    assert (
        clears_cs_promotion(
            name="lambdarank",
            mean_ic=-0.008,
            dm_preferred="lambdarank",
            dm_p=0.009,
            stepm_rejected=["lambdarank"],
            rc_p=0.018,
            spa_p=0.019,
        )
        is False
    )


def test_dm_winner_must_be_the_stepm_rejection() -> None:
    assert (
        clears_cs_promotion(
            name="gbrt",
            mean_ic=0.02,
            dm_preferred="gbrt",
            dm_p=0.01,
            stepm_rejected=["lambdarank"],
            rc_p=0.01,
            spa_p=0.01,
        )
        is False
    )


def test_positive_ic_clears_when_every_gate_matches() -> None:
    assert (
        clears_cs_promotion(
            name="gbrt",
            mean_ic=0.02,
            dm_preferred="gbrt",
            dm_p=0.01,
            stepm_rejected=["gbrt"],
            rc_p=0.01,
            spa_p=0.01,
        )
        is True
    )


def test_both_windows_are_required() -> None:
    from quant_fund.hedge_lab.promotion import names_clearing_both

    assert names_clearing_both(["tsmom"], []) == []
    assert names_clearing_both(["tsmom"], ["nautica"]) == []
    assert names_clearing_both(["tsmom", "gbrt"], ["gbrt"]) == ["gbrt"]


def test_book_overlay_rejects_a_reality_check_without_positive_excess() -> None:
    from quant_fund.hedge_lab.promotion import clears_book_overlay

    assert clears_book_overlay(excess_means=[0.001, -0.002], rc_p=0.01, spa_p=0.01) is False
    assert clears_book_overlay(excess_means=[0.001], rc_p=0.01, spa_p=0.2) is False
    assert clears_book_overlay(excess_means=[0.001, 0.002], rc_p=0.01, spa_p=0.01) is True


def test_kronos_blend_stays_zero_when_ic_is_negative() -> None:
    decision = promotion_decision(
        data_source="file",
        g1={
            "ridge": {"mean_ic": -0.06},
            "kronos_mini": {
                "mean_ic": -0.01,
                "diebold_mariano_crps": {"preferred": "robinhood_plus"},
            },
        },
        calibration={"pass": True},
    )
    assert decision["blend_weight"] == 0.0
    assert decision["sizes_book"] is False
    assert decision["ic_win_vs_public_ridge"] is False
