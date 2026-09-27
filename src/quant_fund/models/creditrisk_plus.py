"""CreditRisk+ portfolio loss distribution (CSFB 1997).

CreditRisk+ approximates each obligor's default count as Poisson with mean equal
to its default probability and aggregates integer-unit exposures.  With
fixed default rates the portfolio loss (in exposure units) is compound Poisson
with probability generating function ``G(z) = exp(sum_i pd_i (z^{v_i} - 1))``.
Its loss distribution follows the Panjer-type recursion

    p_0 = exp(-sum_i pd_i),
    p_n = (1/n) sum_{k=1}^{n} k a_k p_{n-k},   a_k = sum_{i: v_i = k} pd_i.

Reference: Credit Suisse First Boston (1997), *CreditRisk+: A Credit Risk
Management Framework*; H. Panjer (1981).  Fail-closed on invalid inputs.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def creditrisk_plus_distribution(
    pd: Array, exposure_units: Array, max_units: int | None = None
) -> dict[str, Array | float]:
    """Portfolio loss distribution (in exposure units) via the Panjer recursion."""
    pdv = np.asarray(pd, dtype=float).ravel()
    ev = np.asarray(exposure_units, dtype=int).ravel()
    if pdv.size != ev.size or pdv.size < 1:
        raise ValueError("pd and exposure_units must be aligned and non-empty")
    if not np.isfinite(pdv).all() or ((pdv < 0) | (pdv >= 1)).any():
        raise ValueError("pd must be finite in [0, 1)")
    if (ev <= 0).any():
        raise ValueError("exposure_units must be positive integers")
    total_units = int(ev.max() * pdv.size) if max_units is None else int(max_units)
    total_units = max(total_units, int(ev.max()) + 1)
    # a_k = sum of pd over names with exposure band k.
    a = np.zeros(total_units + 1)
    for p, v in zip(pdv, ev, strict=True):
        if v <= total_units:
            a[v] += p
    p_arr = np.zeros(total_units + 1)
    p_arr[0] = np.exp(-float(pdv.sum()))
    for n in range(1, total_units + 1):
        ks = np.arange(1, n + 1)
        # p_{n-k} for k=1..n is p_arr[n-1], p_arr[n-2], ..., p_arr[0].
        p_arr[n] = float(np.sum(ks * a[1 : n + 1] * p_arr[n - 1 :: -1])) / n
    # numerically stabilise / renormalise the truncated tail
    p_arr = np.clip(p_arr, 0.0, None)
    units = np.arange(total_units + 1, dtype=float)
    mean = float(np.sum(pdv * ev))
    var = float(np.sum(pdv * ev**2))
    return {"pmf": p_arr, "units": units, "expected_loss": mean, "loss_var": var}
