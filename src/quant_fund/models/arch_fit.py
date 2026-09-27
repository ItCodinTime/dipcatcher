"""arch-package fits shared by research and the paper quantile lanes.

Research imports this module directly so it does not pull in the paper loop
or the simulated broker.
"""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np


def arch_fit(
    rets_pct: np.ndarray,
    vol: Literal["GARCH", "ARCH", "EGARCH", "FIGARCH", "APARCH", "HARCH"],
    dist: Literal[
        "normal",
        "gaussian",
        "t",
        "studentst",
        "skewstudent",
        "skewt",
        "ged",
        "generalized error",
    ],
    o: int,
):
    """Shared arch fit helper (mirrors scripts/sota_eval_kronos._arch_fit)."""
    from arch import arch_model

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = arch_model(
            rets_pct, mean="Constant", vol=vol, p=1, o=o, q=1, dist=dist, rescale=False
        )
        return model.fit(disp="off", show_warning=False, options={"maxiter": 300})
