"""fx-1 — the quant LLM fine-tuned from Kimi K3 open weights.

fx-1 (always lowercase) *is the model*: a Kimi K3 fine-tune trained
exclusively on gate-passed, receipt-bound dipcatcher research behavior.
This package is the fx-1 project: corpus construction (``fx1.data``),
evaluation (``fx1.eval``), training (``fx1.train``), serving (``fx1.serve``),
the flagship bench (``fx1.bench``), the reward model (``fx1.reward``), and
the harness bridge (``fx1.harness``) through which dipcatcher builds,
evaluates, and verifies the model. See ``docs/FX1.md``.

``fx1.forecast`` is a separate inference/evaluation harness for an external
price/return model that is not implemented in this package. See
``docs/fx1_harness.md``.
"""

__version__ = "0.4.0"

BASE_MODEL = "moonshotai/Kimi-K3"
