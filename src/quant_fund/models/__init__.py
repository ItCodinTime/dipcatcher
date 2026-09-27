"""Forecast-model package.

Heavy estimators (sklearn and optional torch backends) load on first use.
Importing the package does not import them.
"""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "AdaptiveLassoRanker",
    "ClassicRanker",
    "ClassicShortRanker",
    "CombinationRanker",
    "CompositeRanker",
    "DoubleSelectionRanker",
    "ENGINE_DISPLAY",
    "ENGINE_NAME",
    "ElasticNetRanker",
    "FamaMacBethRanker",
    "FamaMacBethRidgeRanker",
    "FNWRanker",
    "ForecastModel",
    "GBRTRanker",
    "GXThreePassRanker",
    "ICWeightedCombinationRanker",
    "IPCARanker",
    "KraussRanker",
    "MSFECombinationRanker",
    "MODEL_VERSION",
    "ModelMeta",
    "NauticaRanker",
    "NeuralRanker",
    "PCRRanker",
    "PLSRanker",
    "PolicyGradientRanker",
    "PrincipalPortfolioRanker",
    "RPPCARanker",
    "RandomFourierRanker",
    "ReversalRanker",
    "RidgeRanker",
    "RobinhoodPlusEngine",
    "RobinhoodPlusPredictor",
    "SDFElasticNetRanker",
    "SDFRidgeRanker",
    "ThreePassFilterRanker",
    "TSMOMRanker",
    "VMERanker",
]

_EXPORTS: dict[str, str] = {
    "AdaptiveLassoRanker": "quant_fund.models.cs_papers",
    "ClassicRanker": "quant_fund.models.cs_papers",
    "ClassicShortRanker": "quant_fund.models.cs_papers",
    "CombinationRanker": "quant_fund.models.cs_papers",
    "CompositeRanker": "quant_fund.models.ranking",
    "DoubleSelectionRanker": "quant_fund.models.cs_papers",
    "ENGINE_DISPLAY": "quant_fund.models.robinhood_plus",
    "ENGINE_NAME": "quant_fund.models.robinhood_plus",
    "ElasticNetRanker": "quant_fund.models.ranking",
    "FamaMacBethRanker": "quant_fund.models.cs_papers",
    "FamaMacBethRidgeRanker": "quant_fund.models.cs_papers",
    "FNWRanker": "quant_fund.models.cs_papers",
    "ForecastModel": "quant_fund.models.base",
    "GBRTRanker": "quant_fund.models.cs_papers",
    "GXThreePassRanker": "quant_fund.models.cs_papers",
    "ICWeightedCombinationRanker": "quant_fund.models.cs_papers",
    "IPCARanker": "quant_fund.models.asset_pricing",
    "KraussRanker": "quant_fund.models.cs_papers",
    "MSFECombinationRanker": "quant_fund.models.cs_papers",
    "MODEL_VERSION": "quant_fund.models.robinhood_plus",
    "ModelMeta": "quant_fund.models.base",
    "NauticaRanker": "quant_fund.lightspeed.ranker",
    "NeuralRanker": "quant_fund.models.ranking",
    "PCRRanker": "quant_fund.models.cs_papers",
    "PLSRanker": "quant_fund.models.cs_papers",
    "PolicyGradientRanker": "quant_fund.models.deep_rl",
    "PrincipalPortfolioRanker": "quant_fund.models.cs_papers",
    "RPPCARanker": "quant_fund.models.cs_papers",
    "RandomFourierRanker": "quant_fund.models.asset_pricing",
    "ReversalRanker": "quant_fund.models.cs_papers",
    "RidgeRanker": "quant_fund.models.ranking",
    "RobinhoodPlusEngine": "quant_fund.models.robinhood_plus",
    "RobinhoodPlusPredictor": "quant_fund.models.robinhood_plus",
    "SDFElasticNetRanker": "quant_fund.models.cs_papers",
    "SDFRidgeRanker": "quant_fund.models.asset_pricing",
    "ThreePassFilterRanker": "quant_fund.models.cs_papers",
    "TSMOMRanker": "quant_fund.models.cs_papers",
    "VMERanker": "quant_fund.models.cs_papers",
}


def __getattr__(name: str) -> object:
    """Resolve a public model on first access.

    ``NauticaRanker`` stays behind this hook: ``lightspeed.ranker`` subclasses
    ``ClassicRanker``, so importing it while this package is still initializing
    recurses.
    """
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals()))
