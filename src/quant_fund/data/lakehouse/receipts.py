"""Optional research-receipt fields for a lake snapshot.

Attaching these does not change scorecard metrics. The default research
provenance builder does not call this; callers opt in by copying the dict
onto ``provenance``. ``verify-research`` accepts a missing id and rejects a
malformed one.
"""

from __future__ import annotations

import re

from quant_fund.data.lakehouse.store import SNAPSHOT_SCHEMA
from quant_fund.schemas.errors import DataContractError

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def snapshot_receipt_fields(snapshot_id: str) -> dict[str, str]:
    """Return provenance fields a research receipt may cite."""
    if not isinstance(snapshot_id, str) or _SHA256.fullmatch(snapshot_id) is None:
        raise DataContractError("data_snapshot_id must be a lowercase sha256")
    return {
        "data_snapshot_id": snapshot_id,
        "data_snapshot_schema": SNAPSHOT_SCHEMA,
    }
