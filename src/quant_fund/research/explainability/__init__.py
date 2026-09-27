"""Model explainability reports for dipcatcher forecast heads.

Research-only tooling. Attribution is always measured as degradation of a
proper score (pinball / CRPS / Brier) — never accuracy, Sharpe, or P&L — per
the repo honesty contract. Reports may be attached to existing research
receipts as an optional, additive sidecar that never modifies sealed files.
"""

from quant_fund.research.explainability.attribution import (
    AttributionResult,
    FeatureAttribution,
    permutation_attribution,
    shap_attribution,
)
from quant_fund.research.explainability.drift import (
    AttributionDrift,
    BlockAttribution,
    attribution_drift,
)
from quant_fund.research.explainability.partial_dependence import (
    PartialDependenceCurve,
    partial_dependence_1d,
    partial_dependence_top_k,
)
from quant_fund.research.explainability.receipt import (
    EXPLAINABILITY_SIDECAR_SCHEMA,
    attach_explainability_report,
    attach_explainability_sidecar,
    explainability_artifact_entry,
    explainability_sidecar_path,
    verify_explainability_sidecar,
)
from quant_fund.research.explainability.report import (
    ExplainabilityReport,
    build_report,
    write_report,
)
from quant_fund.research.explainability.scoring import (
    ProperScoreSpec,
    brier,
    crps_quantiles,
    pinball,
    resolve_score,
)

__all__ = [
    "EXPLAINABILITY_SIDECAR_SCHEMA",
    "AttributionDrift",
    "AttributionResult",
    "BlockAttribution",
    "ExplainabilityReport",
    "FeatureAttribution",
    "PartialDependenceCurve",
    "ProperScoreSpec",
    "attach_explainability_report",
    "attach_explainability_sidecar",
    "attribution_drift",
    "brier",
    "build_report",
    "crps_quantiles",
    "explainability_artifact_entry",
    "explainability_sidecar_path",
    "partial_dependence_1d",
    "partial_dependence_top_k",
    "permutation_attribution",
    "pinball",
    "resolve_score",
    "shap_attribution",
    "verify_explainability_sidecar",
    "write_report",
]
