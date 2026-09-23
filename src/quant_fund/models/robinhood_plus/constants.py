"""robinhood+ identity, Kronos attribution, and K-line column contracts.

robinhood+ is Dipcatcher's rebrand of the Kronos two-stage K-line foundation
model (Shi et al., 2025, arXiv:2508.02739; MIT). It is an internal engine
name and is not affiliated with Robinhood Markets, Inc.
"""

from __future__ import annotations

ENGINE_NAME = "robinhood_plus"
ENGINE_DISPLAY = "robinhood+"
ENGINE_VERSION = "v1"
FAMILY = "kline_foundation"
MODEL_VERSION = f"{ENGINE_NAME}.{ENGINE_VERSION}"

# Kronos paper / upstream (MIT). Required attribution for the derived engine.
KRONOS_PAPER = "Shi et al., 2025. Kronos: A Foundation Model for the Language of Financial Markets. arXiv:2508.02739"
KRONOS_REPO = "https://github.com/shiyu-coder/Kronos"
KRONOS_LICENSE = "MIT"

AFFILIATION_DISCLAIMER = (
    "robinhood+ is a Dipcatcher internal engine name derived from Kronos. "
    "It is not affiliated with, endorsed by, or a product of Robinhood Markets, Inc."
)

PRICE_COLS = (
    "open_split_adjusted",
    "high_split_adjusted",
    "low_split_adjusted",
    "close_split_adjusted",
)
VOLUME_COL = "volume"
AMOUNT_COL = "amount"
KLINE_FEATURE_NAMES = ("open", "high", "low", "close", "volume", "amount")
PRICE_SPACE = "split_adjusted"

# Official Kronos zoo (Hugging Face). Torch backend only; never downloaded in CI.
# Revisions are pinned to the upstream head commits captured 2026-02 — a moving
# ``main`` would silently mutate the weights under a frozen evaluation.
VARIANT_HUB = {
    "mini": {
        "tokenizer": "NeoQuasar/Kronos-Tokenizer-2k",
        "tokenizer_rev": "26966d0035065a0cae0ebad7af8ece35bc1fb51c",
        "model": "NeoQuasar/Kronos-mini",
        "model_rev": "f4e68697d9d5aed55cef5c96aabc3376bcad9f81",
        "max_context": 2048,
        "params": "4.1M",
    },
    "small": {
        "tokenizer": "NeoQuasar/Kronos-Tokenizer-base",
        "tokenizer_rev": "0e0117387f39004a9016484a186a908917e22426",
        "model": "NeoQuasar/Kronos-small",
        "model_rev": "901c26c1332695a2a8f243eb2f37243a37bea320",
        "max_context": 512,
        "params": "24.7M",
    },
    "base": {
        "tokenizer": "NeoQuasar/Kronos-Tokenizer-base",
        "tokenizer_rev": "0e0117387f39004a9016484a186a908917e22426",
        "model": "NeoQuasar/Kronos-base",
        "model_rev": "2b554741eca47781b64468546e77fef3e85130e6",
        "max_context": 512,
        "params": "102.3M",
    },
}

DEFAULT_S1_BITS = 5
DEFAULT_S2_BITS = 5
DEFAULT_LOOKBACK = 64
DEFAULT_PRED_LEN = 20
DEFAULT_SAMPLE_COUNT = 8
DEFAULT_MAX_CONTEXT = 512
DEFAULT_CLIP = 5.0
NORM_EPS = 1e-5

STATUS_OK = "ok"
STATUS_INSUFFICIENT_HISTORY = "insufficient_history"
STATUS_MISSING_OHLC = "missing_ohlc"
STATUS_NONFINITE = "nonfinite_kline"
STATUS_DISABLED = "disabled"
STATUS_TORCH_UNAVAILABLE = "torch_unavailable"
