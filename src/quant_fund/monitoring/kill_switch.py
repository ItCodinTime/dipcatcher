"""Kill switch. Flatten requires explicit human authorization."""

from __future__ import annotations

from quant_fund.config.models import KillSwitchConfig
from quant_fund.schemas.errors import KillSwitchActive

ENABLED = "ENABLED"
HALT_NEW_ORDERS = "HALT_NEW_ORDERS"
CANCEL_OPEN_ORDERS = "CANCEL_OPEN_ORDERS"
FLATTEN_OPTIONAL = "FLATTEN_OPTIONAL"


class KillSwitch:
    def __init__(self, config: KillSwitchConfig) -> None:
        self.state = config.state
        self.allow_auto_flatten = config.allow_auto_flatten

    def assert_new_orders_allowed(self) -> None:
        if self.state in {HALT_NEW_ORDERS, CANCEL_OPEN_ORDERS, FLATTEN_OPTIONAL}:
            raise KillSwitchActive(f"kill switch state={self.state}")

    def may_flatten(self, human_authorized: bool) -> bool:
        if self.state != FLATTEN_OPTIONAL:
            return False
        if self.allow_auto_flatten:
            return False  # still require human; auto flatten never on exception
        return human_authorized
