"""Portfolio-construction research pack: causal weight engines and honest evaluation.

Engines (:func:`fit_weights`) fit on a trailing returns window only —
strictly causal, no lookahead — and pass through a shared constraint layer
(long-only, leverage cap, min/max weight). Evaluation
(:func:`run_walk_forward`) reports realized-vol-vs-target tracking error,
weight turnover/stability, and equal-risk-contribution residuals — never
P&L, Sharpe, or NAV, per the repo honesty contract. Receipts are
lightweight hash-bound JSON documents (:func:`build_receipt` /
:func:`verify_allocation_receipt`), additive to — never inside — the
sealed research-receipt machinery.
"""

from quant_fund.research.allocation.constraints import (
    AllocationConstraints,
    DegenerateCovarianceError,
    InfeasibleConstraintsError,
    apply_constraints,
    validate_covariance,
    weights_satisfy,
)
from quant_fund.research.allocation.engines import (
    ENGINE_NAMES,
    estimate_covariance,
    estimate_mean,
    fit_weights,
    inverse_volatility_weights,
    kelly_weights,
    portfolio_vol,
    risk_contributions,
    risk_parity_weights,
    volatility_target_scale,
)
from quant_fund.research.allocation.evaluation import (
    AllocationEvaluation,
    compare_engines,
    run_walk_forward,
)
from quant_fund.research.allocation.receipt import (
    ALLOCATION_RECEIPT_SCHEMA,
    build_receipt,
    verify_allocation_receipt,
    write_receipt,
)

__all__ = [
    "ALLOCATION_RECEIPT_SCHEMA",
    "ENGINE_NAMES",
    "AllocationConstraints",
    "AllocationEvaluation",
    "DegenerateCovarianceError",
    "InfeasibleConstraintsError",
    "apply_constraints",
    "build_receipt",
    "compare_engines",
    "estimate_covariance",
    "estimate_mean",
    "fit_weights",
    "inverse_volatility_weights",
    "kelly_weights",
    "portfolio_vol",
    "risk_contributions",
    "risk_parity_weights",
    "run_walk_forward",
    "validate_covariance",
    "verify_allocation_receipt",
    "volatility_target_scale",
    "weights_satisfy",
    "write_receipt",
]
