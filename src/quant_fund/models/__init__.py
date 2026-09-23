from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quant_fund.lightspeed.ranker import NauticaRanker

from quant_fund.models.asset_pricing import IPCARanker, RandomFourierRanker, SDFRidgeRanker
from quant_fund.models.base import ForecastModel, ModelMeta
from quant_fund.models.cs_papers import (
    AdaptiveLassoRanker,
    ClassicRanker,
    ClassicShortRanker,
    CombinationRanker,
    DoubleSelectionRanker,
    FamaMacBethRanker,
    FamaMacBethRidgeRanker,
    FNWRanker,
    GBRTRanker,
    GXThreePassRanker,
    ICWeightedCombinationRanker,
    KraussRanker,
    MSFECombinationRanker,
    PCRRanker,
    PLSRanker,
    PrincipalPortfolioRanker,
    ReversalRanker,
    RPPCARanker,
    SDFElasticNetRanker,
    ThreePassFilterRanker,
    TSMOMRanker,
    VMERanker,
)
from quant_fund.models.ranking import CompositeRanker, ElasticNetRanker, RidgeRanker
from quant_fund.models.robinhood_plus import (
    ENGINE_DISPLAY,
    ENGINE_NAME,
    MODEL_VERSION,
    RobinhoodPlusEngine,
    RobinhoodPlusPredictor,
)

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
    "NauticaRanker",
    "MODEL_VERSION",
    "ModelMeta",
    "PCRRanker",
    "PLSRanker",
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


def __getattr__(name: str):
    """Load the Lightspeed adapter lazily to avoid a package cycle.

    ``lightspeed.ranker`` subclasses ``ClassicRanker`` from ``cs_papers``.
    Importing it eagerly here makes ``import quant_fund.lightspeed`` recurse
    through this package while ``cs_papers`` is still initializing.
    """
    if name == "NauticaRanker":
        from quant_fund.lightspeed.ranker import NauticaRanker

        return NauticaRanker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
