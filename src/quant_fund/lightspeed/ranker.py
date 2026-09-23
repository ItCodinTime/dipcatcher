"""CS analog of frozen nautica 63-day momentum. Challenger only."""

from __future__ import annotations

from quant_fund.models.cs_papers import ClassicRanker

NAUTICA_SIGNS: dict[str, float] = {
    "cs_z_mom_60": 1.0,
}


class NauticaRanker(ClassicRanker):
    """A priori +1 on ``cs_z_mom_60``. No estimated slopes, no OOS peek.

    Public-feature stand-in for Lightspeed nautica-momentum-v1 (63-day
    total-return score, mom_blend=1). Challenger in the paper catalog;
    champion remains public ridge.
    """

    def __init__(self) -> None:
        super().__init__(
            signs=dict(NAUTICA_SIGNS),
            name="nautica",
            paper="Lightspeed nautica-momentum-v1; Jegadeesh–Titman (1993) 63d mom",
        )
