"""Kill-switch state-machine extremes: ENABLED↔halt/cancel/flatten, fail-closed."""

from __future__ import annotations

import pytest

from quant_fund.config.models import KillSwitchConfig
from quant_fund.monitoring.kill_switch import (
    BLOCKING_STATES,
    CANCEL_OPEN_ORDERS,
    ENABLED,
    FLATTEN_OPTIONAL,
    HALT_NEW_ORDERS,
    VALID_STATES,
    KillSwitch,
)
from quant_fund.schemas.errors import KillSwitchActive


def test_enabled_allows_new_orders() -> None:
    switch = KillSwitch(KillSwitchConfig(state=ENABLED))
    switch.assert_new_orders_allowed()  # does not raise


@pytest.mark.parametrize("state", sorted(BLOCKING_STATES))
def test_blocking_states_raise_kill_switch_active(state: str) -> None:
    switch = KillSwitch(KillSwitchConfig(state=state))
    with pytest.raises(KillSwitchActive, match=state):
        switch.assert_new_orders_allowed()


@pytest.mark.parametrize(
    "bad",
    ["", "enabled", "HALT", "UNKNOWN", "DISABLED", " flatten ", "None"],
)
def test_invalid_state_fail_closed_blocks_orders(bad: str) -> None:
    switch = KillSwitch(KillSwitchConfig(state=bad))
    with pytest.raises(KillSwitchActive):
        switch.assert_new_orders_allowed()
    assert switch.may_flatten(True) is False
    assert switch.may_cancel_open_orders() is False


def test_set_state_transitions_round_trip() -> None:
    switch = KillSwitch(KillSwitchConfig(state=ENABLED))
    switch.assert_new_orders_allowed()

    switch.set_state(HALT_NEW_ORDERS)
    with pytest.raises(KillSwitchActive):
        switch.assert_new_orders_allowed()
    assert switch.may_flatten(True) is False
    assert switch.may_cancel_open_orders() is False

    switch.set_state(CANCEL_OPEN_ORDERS)
    with pytest.raises(KillSwitchActive):
        switch.assert_new_orders_allowed()
    assert switch.may_cancel_open_orders() is True
    assert switch.may_flatten(True) is False

    switch.set_state(FLATTEN_OPTIONAL)
    with pytest.raises(KillSwitchActive):
        switch.assert_new_orders_allowed()
    assert switch.may_cancel_open_orders() is True
    assert switch.may_flatten(human_authorized=True) is True

    switch.set_state(ENABLED)
    switch.assert_new_orders_allowed()
    assert switch.may_flatten(True) is False


def test_set_state_invalid_raises_and_preserves_prior() -> None:
    switch = KillSwitch(KillSwitchConfig(state=ENABLED))
    with pytest.raises(KillSwitchActive, match="invalid"):
        switch.set_state("NOT_A_STATE")
    assert switch.state == ENABLED
    switch.assert_new_orders_allowed()


def test_flatten_requires_human_and_reject_auto_flag() -> None:
    # Happy path: FLATTEN_OPTIONAL + human + auto flag false.
    ok = KillSwitch(KillSwitchConfig(state=FLATTEN_OPTIONAL, allow_auto_flatten=False))
    assert ok.may_flatten(False) is False
    assert ok.may_flatten(True) is True

    # allow_auto_flatten=True is never trusted — still refuse flatten.
    blocked = KillSwitch(KillSwitchConfig(state=FLATTEN_OPTIONAL, allow_auto_flatten=True))
    assert blocked.may_flatten(False) is False
    assert blocked.may_flatten(True) is False

    # Wrong state: never flatten even with human auth.
    for state in (ENABLED, HALT_NEW_ORDERS, CANCEL_OPEN_ORDERS):
        sw = KillSwitch(KillSwitchConfig(state=state, allow_auto_flatten=False))
        assert sw.may_flatten(True) is False


def test_valid_states_cover_documented_machine() -> None:
    assert (
        frozenset({ENABLED, HALT_NEW_ORDERS, CANCEL_OPEN_ORDERS, FLATTEN_OPTIONAL}) == VALID_STATES
    )
    assert VALID_STATES - {ENABLED} == BLOCKING_STATES
