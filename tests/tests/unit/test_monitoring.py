import numpy as np
import pytest

from quant_fund.config.models import KillSwitchConfig
from quant_fund.monitoring.drift import drift_report, mean_shift, psi
from quant_fund.monitoring.kill_switch import (
    CANCEL_OPEN_ORDERS,
    FLATTEN_OPTIONAL,
    HALT_NEW_ORDERS,
    KillSwitch,
)
from quant_fund.schemas.errors import KillSwitchActive


def test_drift_report_alerts_and_reports_sample_sufficiency() -> None:
    reference = np.linspace(0.0, 1.0, 100)
    current = reference + 2.0
    report = drift_report(reference, current, psi_threshold=0.25, mean_shift_threshold=1.0)
    assert report["status"] == "alert"
    assert report["alert"] is True
    assert report["sufficient_data"] is True

    short = drift_report(np.array([1.0]), np.array([2.0]))
    assert short["status"] == "insufficient_data"
    assert np.isnan(float(short["psi"]))


def test_psi_rejects_invalid_bin_count() -> None:
    with pytest.raises((ValueError, TypeError)):
        psi(np.arange(20.0), np.arange(20.0), bins=0)


@pytest.mark.parametrize("state", [HALT_NEW_ORDERS, CANCEL_OPEN_ORDERS, FLATTEN_OPTIONAL])
def test_kill_switch_blocks_new_orders(state: str) -> None:
    switch = KillSwitch(KillSwitchConfig(state=state))
    with pytest.raises(KillSwitchActive):
        switch.assert_new_orders_allowed()


def test_flatten_requires_human_authorization() -> None:
    switch = KillSwitch(KillSwitchConfig(state=FLATTEN_OPTIONAL, allow_auto_flatten=True))
    assert switch.may_flatten(False) is False
    assert switch.may_flatten(True) is False


def test_psi_identical_distributions_near_zero() -> None:
    """Identical samples → PSI ≈ 0 (clip floor may leave tiny residual)."""
    rng = np.random.default_rng(0)
    reference = rng.normal(size=200)
    value = psi(reference, reference.copy(), bins=10)
    assert np.isfinite(value)
    assert value == pytest.approx(0.0, abs=1e-10)


def test_mean_shift_identical_and_known_offset() -> None:
    reference = np.linspace(0.0, 1.0, 50)
    assert mean_shift(reference, reference.copy()) == pytest.approx(0.0, abs=1e-12)
    assert mean_shift(reference, reference + 2.5) == pytest.approx(2.5, abs=1e-12)
    assert mean_shift(reference, reference - 1.0) == pytest.approx(-1.0, abs=1e-12)


def test_psi_known_bin_shift_positive() -> None:
    """Mass shifted into higher bins → strictly positive PSI."""
    reference = np.linspace(0.0, 1.0, 200)
    # Push current mass toward the upper end of the reference range.
    current = np.concatenate([np.linspace(0.6, 1.0, 160), np.linspace(0.0, 0.2, 40)])
    value = psi(reference, current, bins=10)
    assert value > 0.05
    identical = psi(reference, reference, bins=10)
    assert value > identical


def test_psi_empty_and_short_windows_nan() -> None:
    assert np.isnan(psi(np.array([]), np.array([1.0, 2.0, 3.0]), bins=2))
    assert np.isnan(psi(np.arange(5.0), np.arange(5.0), bins=10))  # size < bins
    assert np.isnan(psi(np.arange(20.0), np.array([np.nan, np.nan]), bins=5))


def test_mean_shift_all_nan_is_nan() -> None:
    assert np.isnan(mean_shift(np.array([np.nan, np.nan]), np.array([1.0, 2.0])))


def test_drift_report_mean_shift_only_alert() -> None:
    reference = np.linspace(0.0, 1.0, 100)
    # Small distributional change but large mean offset.
    current = reference + 3.0
    report = drift_report(
        reference,
        current,
        psi_threshold=1e9,  # disable PSI alert
        mean_shift_threshold=1.0,
        bins=10,
    )
    assert report["alert"] is True
    assert report["status"] == "alert"
    assert float(report["mean_shift"]) == pytest.approx(3.0, abs=1e-9)
