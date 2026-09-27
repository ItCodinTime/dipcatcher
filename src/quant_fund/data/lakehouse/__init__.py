"""Content-addressed, lineage-tracked market-data lake.

Research storage only. This package does not place orders or claim live P&L.
"""

from quant_fund.data.lakehouse.lineage import format_dag, record_lineage, verify_lineage
from quant_fund.data.lakehouse.migrate import import_files
from quant_fund.data.lakehouse.panels import lagged_return_panel_bytes
from quant_fund.data.lakehouse.quality import QualityThresholds, quality_report, quality_report_path
from quant_fund.data.lakehouse.query import asof_bars, universe_asof
from quant_fund.data.lakehouse.receipts import snapshot_receipt_fields
from quant_fund.data.lakehouse.store import load_snapshot, write_partitioned_bars

__all__ = [
    "QualityThresholds",
    "asof_bars",
    "format_dag",
    "import_files",
    "lagged_return_panel_bytes",
    "load_snapshot",
    "quality_report",
    "quality_report_path",
    "record_lineage",
    "snapshot_receipt_fields",
    "universe_asof",
    "verify_lineage",
    "write_partitioned_bars",
]
