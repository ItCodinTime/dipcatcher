"""Wave 35: feasible MV weights respect gross/net bounds (Hypothesis).

Unit fixtures already cover fixed cases in test_optimizer_edges;
this property suite randomizes alpha / PSD sigma / caps when the solve
is feasible and finite. Research-only — not a live P&L claim.
"""

from __future__ import annotations

import numpy as np
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from quant_fund.config.models import AppConfig
from quant_fund.portfolio.optimizer import optimize_mean_variance


@st.composite
def feasible_mv_instances(draw: st.DrawFn):
    n = draw(st.integers(min_value=2, max_value=5))
    alpha = np.asarray(
        draw(
            st.lists(
                st.floats(-0.08, 0.08, allow_nan=False, allow_infinity=False),
                min_size=n,
                max_size=n,
            )
        ),
        dtype=float,
    )
    # Diagonal PSD cov keeps solves cheap and always repairable.
    vols = np.asarray(
        draw(
            st.lists(
                st.floats(0.01, 0.08, allow_nan=False, allow_infinity=False),
                min_size=n,
                max_size=n,
            )
        ),
        dtype=float,
    )
    sig = np.diag(vols**2)
    gross = draw(st.floats(0.2, 1.5, allow_nan=False, allow_infinity=False))
    net = draw(st.floats(0.05, min(0.5, gross), allow_nan=False, allow_infinity=False))
    name = draw(st.floats(0.05, max(0.05, gross), allow_nan=False, allow_infinity=False))
    # Keep sleeve caps consistent with AppConfig validators / effective gross.
    long_max = gross
    short_max = gross
    w_prev = np.zeros(n)
    return (
        alpha,
        sig,
        w_prev,
        float(gross),
        float(net),
        float(name),
        float(long_max),
        float(short_max),
    )


@given(feasible_mv_instances())
@settings(max_examples=35, deadline=None)
def test_feasible_weights_respect_gross_net_bounds(inst) -> None:
    alpha, sig, w_prev, gross, net, name, long_max, short_max = inst
    cfg = AppConfig()
    cfg.optimizer.lambda_tc = 0.0
    cfg.optimizer.lambda_risk = 1.0
    cfg.optimizer.lambda_turnover = 0.0
    cfg.constraints.gross_leverage = gross
    cfg.constraints.net_exposure = net
    cfg.constraints.name_max = name
    cfg.constraints.name_min = -name
    cfg.constraints.long_max = long_max
    cfg.constraints.short_max = short_max
    cfg.constraints.turnover_limit = max(gross, 1.0)
    cfg.constraints.cash_buffer = 0.0
    w, diag = optimize_mean_variance(alpha, sig, w_prev, cfg)
    assume(diag.feasible)
    assume(w is not None and np.all(np.isfinite(w)))
    tol = 1e-5
    assert float(np.sum(np.abs(w))) <= gross + tol
    assert abs(float(np.sum(w))) <= net + tol
    assert np.all(w <= name + tol)
    assert np.all(w >= -name - tol)
    if diag.gross is not None:
        assert diag.gross <= gross + tol
    if diag.net is not None:
        assert abs(diag.net) <= net + tol
