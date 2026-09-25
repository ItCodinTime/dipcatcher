"""fx-1 training corpus construction from dipcatcher artifacts."""

from fx1.data.corpus import SFTExample, build_corpus, build_full_corpus
from fx1.data.ledger import CorpusLedger, LedgerEntry
from fx1.data.ledgers import ledger_corpus, ledger_examples
from fx1.data.notebooks import notebook_examples
from fx1.data.quality import QualityReport, SplitManifest, dedup_and_filter, frozen_split
from fx1.data.receipts import ReceiptRecord, load_receipts
from fx1.data.traces import TraceRecorder, TraceStep, Trajectory

__all__ = [
    "QualityReport",
    "CorpusLedger",
    "LedgerEntry",
    "ReceiptRecord",
    "SFTExample",
    "SplitManifest",
    "TraceRecorder",
    "TraceStep",
    "Trajectory",
    "build_corpus",
    "build_full_corpus",
    "dedup_and_filter",
    "frozen_split",
    "ledger_corpus",
    "ledger_examples",
    "load_receipts",
    "notebook_examples",
    "sources",
]


def __getattr__(name: str):  # lazy: keep base import light
    if name == "sources":
        from fx1.data import sources

        return sources
    raise AttributeError(name)

