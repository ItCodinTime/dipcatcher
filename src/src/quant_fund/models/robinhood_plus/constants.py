"""Stable names and column contracts for robinhood+ research diagnostics."""

ENGINE_NAME = "robinhood_plus"
MODEL_VERSION = "robinhood_plus.numpy.v1"
STATUS_OK = "ok"
STATUS_FALLBACK = "fallback"
STATUS_INSUFFICIENT = "insufficient_history"
PRICE_COLS = ("open", "high", "low", "close")
VOLUME_COL = "volume"
AMOUNT_COL = "amount"
KLINE_COLS = (*PRICE_COLS, VOLUME_COL, AMOUNT_COL)
