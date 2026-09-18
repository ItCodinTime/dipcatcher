"""Kill switch. Flatten requires explicit human authorization."""

from __future__ import annotations

from quant_fund.config.models import KillSwitchConfig
from quant_fund.schemas.errors import KillSwitchActive

ENABLED = "ENABLED"
HALT_NEW_ORDERS = "HALT_NEW_ORDERS"
CANCEL_OPEN_ORDERS = "CANCEL_OPEN_ORDERS"
FLATTEN_OPTIONAL = "FLATTEN_OPTIONAL"

VALID_STATES = frozenset({ENABLED, HALT_NEW_ORDERS, CANCEL_OPEN_ORDERS, FLATTEN_OPTIONAL})
# Any non-ENABLED state (including unknown) blocks new orders — fail closed.
BLOCKING_STATES = frozenset({HALT_NEW_ORDERS, CANCEL_OPEN_ORDERS, FLATTEN_OPTIONAL})


class KillSwitch:
    def __init__(self, config: KillSwitchConfig) -> None:
        self.state = str(config.state)
        self.allow_auto_flatten = bool(config.allow_auto_flatten)
        # Fail closed: unknown config state is treated as blocking (do not coerce).
        if self.state not in VALID_STATES:
            # Keep invalid state visible; assert_new_orders_allowed will raise.
            pass

    def set_state(self, state: str) -> None:
        """Transition kill-switch state. Invalid names raise KillSwitchActive (fail closed)."""
        if state not in VALID_STATES:
            raise KillSwitchActive(f"invalid kill switch state={state}")
        self.state = state

    def assert_new_orders_allowed(self) -> None:
        # Only ENABLED permits new orders. Unknown / halt / flatten all raise.
        if self.state != ENABLED:
            raise KillSwitchActive(f"kill switch state={self.state}")

    def may_cancel_open_orders(self) -> bool:
        return self.state in {CANCEL_OPEN_ORDERS, FLATTEN_OPTIONAL}

    def may_flatten(self, human_authorized: bool) -> bool:
        if self.state != FLATTEN_OPTIONAL:
            return False
        if self.allow_auto_flatten:
            return False  # still require human; auto flatten never on exception
        return bool(human_authorized)
