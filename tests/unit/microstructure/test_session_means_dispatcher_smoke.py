"""Smoke: dispatcher invokes all known session soft-verify helpers."""

from __future__ import annotations

from quant_fund.research.catalog import (
    NORTHSET_SESSION_MEANS_HONESTY_HELPERS,
    northset_session_book_snaps_honesty_errors,
    northset_session_book_snaps_n_session_consistency_errors,
    northset_session_book_vpin_mean_honesty_errors,
    northset_session_close_depth_honesty_errors,
    northset_session_close_imbalance_honesty_errors,
    northset_session_close_micro_bps_honesty_errors,
    northset_session_close_mid_honesty_errors,
    northset_session_close_mid_micro_pair_honesty_errors,
    northset_session_close_spread_bps_honesty_errors,
    northset_session_imbalance_mean_honesty_errors,
    northset_session_imbalance_std_honesty_errors,
    northset_session_ofi_abs_dominates_sum_honesty_errors,
    northset_session_ofi_abs_sum_honesty_errors,
    northset_session_ofi_sum_ic_honesty_errors,
    northset_session_ofi_sum_mean_honesty_errors,
    northset_session_spread_bps_mean_honesty_errors,
)

_KNOWN = (
    northset_session_imbalance_mean_honesty_errors,
    northset_session_close_micro_bps_honesty_errors,
    northset_session_close_depth_honesty_errors,
    northset_session_imbalance_std_honesty_errors,
    northset_session_close_imbalance_honesty_errors,
    northset_session_close_mid_honesty_errors,
    northset_session_close_mid_micro_pair_honesty_errors,
    northset_session_close_spread_bps_honesty_errors,
    northset_session_spread_bps_mean_honesty_errors,
    northset_session_book_snaps_honesty_errors,
    northset_session_ofi_sum_mean_honesty_errors,
    northset_session_ofi_abs_sum_honesty_errors,
    northset_session_ofi_abs_dominates_sum_honesty_errors,
    northset_session_book_vpin_mean_honesty_errors,
    northset_session_book_snaps_n_session_consistency_errors,
    northset_session_ofi_sum_ic_honesty_errors,
)


def test_dispatcher_helper_tuple_matches_known_set() -> None:
    names = {fn.__name__ for fn in NORTHSET_SESSION_MEANS_HONESTY_HELPERS}
    expected = {fn.__name__ for fn in _KNOWN}
    assert names == expected
    assert len(NORTHSET_SESSION_MEANS_HONESTY_HELPERS) == len(_KNOWN)


def test_dispatcher_invokes_each_helper(monkeypatch) -> None:
    called: list[str] = []

    def _wrap(fn):
        def _inner(blob: object) -> list[str]:
            called.append(fn.__name__)
            return fn(blob)

        _inner.__name__ = fn.__name__
        return _inner

    wrapped = tuple(_wrap(fn) for fn in _KNOWN)
    monkeypatch.setattr(
        "quant_fund.research.catalog.NORTHSET_SESSION_MEANS_HONESTY_HELPERS",
        wrapped,
    )
    # Re-import path: dispatcher reads the module-level tuple at call time
    from quant_fund.research import catalog as cat

    assert cat.northset_session_means_honesty_errors({}) == []
    assert called == [fn.__name__ for fn in _KNOWN]


def test_doctor_mentions_session_means_dispatcher() -> None:
    from typer.testing import CliRunner

    from quant_fund.cli.main import app

    result = CliRunner().invoke(app, ["doctor"])
    # doctor may exit 1 if dirs unhealthy; still check stdout pointer
    assert "northset_session_means_honesty_errors" in result.output
    assert "research_only" in result.output.lower() or "no Sharpe" in result.output
