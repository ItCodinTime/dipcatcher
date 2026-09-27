"""Sensitivity, flat-optimum search, walk-forward, and adversarial radius.

Diagnostics for the differentiable research book. Callers label the price
path. ``run_small_case`` is the fixed SYNTHETIC CI path: it is not market
evidence and not a live P&L claim. Out-of-sample numbers are whatever the
hard book produces. Nothing here retunes the DGP after seeing them.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np

from quant_fund.diffbacktest.jax_core import (
    _forward,
    _libs,
    _objective_from_net,
    _returns,
    _static_ints,
    objective_gradients,
)
from quant_fund.diffbacktest.numpy_core import (
    drawdown,
    objective_value,
    path_sigma,
    perturb_prices,
    sharpe,
    simulate,
    sum_pnl,
    synthetic_prices,
    terminal_pnl,
    warmup_start,
)
from quant_fund.diffbacktest.spec import (
    BOXES,
    DATA_SOURCE_SYNTHETIC,
    LIVE_PNL_CLAIM,
    OBJECTIVES,
    RESEARCH_ONLY,
    SMALL_CASE_ADAM_STEPS,
    SMALL_CASE_BETA,
    SMALL_CASE_BISECTIONS,
    SMALL_CASE_LAMBDA,
    SMALL_CASE_MU,
    SMALL_CASE_N,
    SMALL_CASE_PGD_STEPS,
    SMALL_CASE_RHO_MAX,
    SMALL_CASE_SEED,
    SMALL_CASE_SIGMA,
    SMALL_CASE_T,
    SMALL_CASE_TEST,
    SMALL_CASE_TRAIN,
    StrategyParams,
    active_parameters,
    free_parameters,
    harden,
    limitations,
    pack,
    project_box,
    unpack,
    validate_params,
)

# Fixed before any out-of-sample number was computed. Not a searched rate.
_ADAM_LR = 0.05
_ADAM_Z_CLIP = 6.0
_RETURN_CLIP = (-0.8, 2.0)

# Same free-parameter space the optimizer moves. Order is the tie break.
_GRIDS: dict[str, dict[str, tuple[float, ...]]] = {
    "tsmom": {
        "target_vol": (0.15, 0.35, 0.6),
        "rebalance_band": (0.0, 0.02),
        "max_gross": (0.5, 1.0),
    },
    "topk": {"top_k": (1.0, 2.0), "rebalance_band": (0.0, 0.02)},
    "antonacci": {"lookback": (12.0, 21.0), "rebalance_band": (0.0, 0.02)},
    "risk_parity": {
        "vol_lookback": (8.0, 12.0, 20.0),
        "rebalance_band": (0.0, 0.02),
        "max_gross": (0.5, 1.0),
    },
    "momentum": {
        "fraction": (0.34, 0.67),
        "rebalance_band": (0.0, 0.02),
        "gross_limit": (0.5, 1.0),
    },
    "reversal": {
        "fraction": (0.34, 0.67),
        "rebalance_band": (0.0, 0.02),
        "gross_limit": (0.5, 1.0),
    },
    "equal_weight": {"gross_limit": (0.5, 1.0), "rebalance_band": (0.0, 0.02)},
}

_SMALL_CASE_STRATEGIES = ("tsmom", "momentum", "risk_parity")


def _label(data_source: str) -> dict[str, Any]:
    return {
        "research_only": RESEARCH_ONLY,
        "live_pnl_claim": LIVE_PNL_CLAIM,
        "data_source": data_source,
    }


def _scaled_gradient(names: tuple[str, ...], gradient: np.ndarray) -> np.ndarray:
    width = np.array([BOXES[name][1] - BOXES[name][0] for name in names], dtype=np.float64)
    return np.asarray(gradient, dtype=np.float64) * width


def sensitivity_map(
    prices: np.ndarray,
    strategy: str,
    params: StrategyParams | None = None,
    *,
    beta: float = 8.0,
    data_source: str = "unspecified",
) -> dict[str, Any]:
    """Smooth gradients of each objective, raw and rescaled by the box width."""
    book = params or StrategyParams()
    validate_params(strategy, book)
    names = active_parameters(strategy)
    objectives: dict[str, Any] = {}
    for objective in OBJECTIVES:
        grad = objective_gradients(
            prices, strategy, book, objective, mode="smooth", beta=float(beta)
        )
        scaled = _scaled_gradient(names, grad.parameter_gradient)
        objectives[objective] = {
            "value": float(grad.value),
            "parameter_gradient": np.asarray(grad.parameter_gradient, dtype=np.float64),
            "scaled_gradient": scaled,
            "gradient_norm": float(np.linalg.norm(grad.parameter_gradient)),
            "scaled_gradient_norm": float(np.linalg.norm(scaled)),
            "price_gradient_norm": float(np.linalg.norm(grad.price_gradient)),
        }
    return {
        "strategy": strategy,
        "parameter_names": names,
        "beta": float(beta),
        "mode": "smooth",
        "objectives": objectives,
        **_label(data_source),
    }


def _grid_points(strategy: str) -> list[dict[str, float]]:
    free = free_parameters(strategy)
    grid = _GRIDS[strategy]
    missing = [name for name in free if name not in grid]
    if missing:
        raise ValueError(f"{strategy} grid is missing {missing}")
    axes = [grid[name] for name in free]
    points: list[dict[str, float]] = []

    def rec(i: int, acc: dict[str, float]) -> None:
        if i == len(free):
            points.append(dict(acc))
            return
        for value in axes[i]:
            acc[free[i]] = float(value)
            rec(i + 1, acc)
        acc.pop(free[i], None)

    rec(0, {})
    return points


def grid_search(
    prices: np.ndarray,
    strategy: str,
    params: StrategyParams | None = None,
    *,
    data_source: str = "unspecified",
) -> dict[str, Any]:
    """Maximize hard-book sample ratio on ``net[warmup:]``. Train prices only.

    Tie break is the grid order in ``_GRIDS`` (first maximum wins). A grid
    with no finite score returns the input parameters and status ``degenerate``.
    """
    book = params or StrategyParams()
    validate_params(strategy, book)
    best_score = float("-inf")
    best_params = book
    best_point: dict[str, float] | None = None
    n_finite = 0
    for point in _grid_points(strategy):
        trial = project_box(harden(book.with_updates(**point)), free_parameters(strategy))
        try:
            validate_params(strategy, trial)
            sim = simulate(prices, strategy, trial)
        except ValueError:
            continue
        start = warmup_start(strategy, trial)
        score = objective_value(
            sim.net, "sharpe", periods_per_year=trial.periods_per_year, score_start=start
        )
        if not np.isfinite(score):
            continue
        n_finite += 1
        if score > best_score:
            best_score = float(score)
            best_params = trial
            best_point = point
    if best_point is None:
        return {
            "strategy": strategy,
            "status": "degenerate",
            "params": book,
            "hardened": harden(book),
            "train_hard_sharpe": float("nan"),
            "n_finite": 0,
            "n_candidates": len(_grid_points(strategy)),
            **_label(data_source),
        }
    return {
        "strategy": strategy,
        "status": "ok",
        "params": best_params,
        "hardened": best_params,
        "point": best_point,
        "train_hard_sharpe": best_score,
        "n_finite": n_finite,
        "n_candidates": len(_grid_points(strategy)),
        **_label(data_source),
    }


def _logit_box(theta: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    width = np.maximum(hi - lo, 1e-12)
    prob = np.clip((theta - lo) / width, 1e-4, 1.0 - 1e-4)
    return cast(np.ndarray, np.log(prob) - np.log(1.0 - prob))


def optimize_flat(
    prices: np.ndarray,
    strategy: str,
    params: StrategyParams | None = None,
    *,
    lam: float = SMALL_CASE_LAMBDA,
    steps: int = SMALL_CASE_ADAM_STEPS,
    beta: float = SMALL_CASE_BETA,
    lr: float = _ADAM_LR,
    data_source: str = "unspecified",
) -> dict[str, Any]:
    """Ascend ``sample ratio − lam * ||gradient||`` inside the free-parameter box.

    The gradient inside the penalty is the smooth-mode gradient with respect
    to every active parameter, including costs. Adam steps only the free
    parameters (the same names the grid is allowed to change). The best
    iterate is kept, so a diverging step is not reported as the result.
    """
    if lam < 0 or steps < 1 or lr <= 0 or beta <= 0:
        raise ValueError("lam >= 0, steps >= 1, lr > 0, beta > 0 required")
    book = params or StrategyParams()
    validate_params(strategy, book)
    names = active_parameters(strategy)
    free = free_parameters(strategy)
    free_idx = np.array([names.index(name) for name in free], dtype=np.int32)
    lo = np.array([BOXES[name][0] for name in free], dtype=np.float64)
    hi = np.array([BOXES[name][1] for name in free], dtype=np.float64)
    theta0 = pack(book, names)
    score_start = warmup_start(strategy, book)
    if prices.shape[0] < score_start + 2:
        raise ValueError("need at least two scored bars after warmup")

    jax, jnp = _libs()
    lookback_i, skip_i, vol_i, topk_i, long_only_i, require_i, long_short_i = _static_ints(book)
    periods = float(book.periods_per_year)
    delay = int(book.delay)
    every = int(book.rebalance_every)
    px = jnp.asarray(np.asarray(prices, dtype=np.float64))
    base = jnp.asarray(theta0)
    f_idx = jnp.asarray(free_idx)
    lo_j = jnp.asarray(lo)
    hi_j = jnp.asarray(hi)
    lam_j = jnp.asarray(float(lam))

    def smooth_sharpe(theta: Any) -> Any:
        _w, _r, _t, _c, net, _nav = _forward(
            px,
            theta,
            strategy=strategy,
            mode="smooth",
            beta=float(beta),
            delay=delay,
            every=every,
            periods=periods,
            initial_nav=1.0,
            names=names,
            lookback_i=lookback_i,
            skip_i=skip_i,
            vol_i=vol_i,
            topk_i=topk_i,
            long_only_i=long_only_i,
            require_i=require_i,
            long_short_i=long_short_i,
        )
        return _objective_from_net(
            net,
            "sharpe",
            initial_nav=1.0,
            periods=periods,
            score_start=int(score_start),
            beta=float(beta),
            smooth_drawdown=True,
        )

    def loss(z: Any) -> Any:
        prob = jax.nn.sigmoid(z)
        free_theta = lo_j + (hi_j - lo_j) * prob
        theta = base.at[f_idx].set(free_theta)
        value, grad = jax.value_and_grad(smooth_sharpe)(theta)
        gnorm = jnp.sqrt(jnp.sum(jnp.square(grad)) + 1e-12)
        return -(value - lam_j * gnorm)

    loss_grad = jax.jit(jax.value_and_grad(loss))
    z = jnp.asarray(_logit_box(theta0[free_idx], lo, hi))

    def eval_J(point: Any) -> tuple[float, Any]:
        value, g = loss_grad(point)
        return float(-value), g

    J, g = eval_J(z)
    best_z = z
    best_J = J
    trace = [J]
    status = "ok"
    m = jnp.zeros_like(z)
    v = jnp.zeros_like(z)
    b1, b2 = 0.9, 0.999
    for t in range(1, int(steps) + 1):
        if not np.isfinite(J) or not np.all(np.isfinite(np.asarray(g))):
            status = "nan_step"
            break
        m = b1 * m + (1.0 - b1) * g
        v = b2 * v + (1.0 - b2) * jnp.square(g)
        mhat = m / (1.0 - b1**t)
        vhat = v / (1.0 - b2**t)
        z = jnp.clip(z - float(lr) * mhat / (jnp.sqrt(vhat) + 1e-8), -_ADAM_Z_CLIP, _ADAM_Z_CLIP)
        J, g = eval_J(z)
        trace.append(J)
        if np.isfinite(J) and best_J < J:
            best_J = J
            best_z = z

    prob = jax.nn.sigmoid(best_z)
    free_theta = np.asarray(lo_j + (hi_j - lo_j) * prob, dtype=np.float64)
    theta = np.array(theta0, copy=True)
    theta[free_idx] = free_theta
    fit = project_box(unpack(theta, names, book), free)
    # Report the objective at the parameters we return. Recompute so a clip
    # into the box cannot desync the logged value from the vector.
    grad = objective_gradients(
        prices, strategy, fit, "sharpe", mode="smooth", beta=float(beta), score_start=score_start
    )
    gnorm = float(np.linalg.norm(grad.parameter_gradient))
    end_J = float(grad.value) - float(lam) * gnorm
    start_grad = objective_gradients(
        prices, strategy, book, "sharpe", mode="smooth", beta=float(beta), score_start=score_start
    )
    start_gnorm = float(np.linalg.norm(start_grad.parameter_gradient))
    start_J = float(start_grad.value) - float(lam) * start_gnorm
    return {
        "strategy": strategy,
        "status": status,
        "lambda": float(lam),
        "steps": int(steps),
        "beta": float(beta),
        "learning_rate": float(lr),
        "score_start": int(score_start),
        "free_parameters": free,
        "start_objective": start_J,
        "end_objective": end_J,
        "start_sharpe": float(start_grad.value),
        "end_sharpe": float(grad.value),
        "start_grad_norm": start_gnorm,
        "end_grad_norm": gnorm,
        "improved": bool(end_J >= start_J - 1e-8),
        "objective_trace": [float(x) for x in trace],
        "params": fit,
        "hardened": harden(fit),
        **_label(data_source),
    }


def _hard_train_sharpe(prices: np.ndarray, strategy: str, params: StrategyParams) -> float:
    book = harden(params)
    sim = simulate(prices, strategy, book)
    return objective_value(
        sim.net,
        "sharpe",
        periods_per_year=book.periods_per_year,
        score_start=warmup_start(strategy, book),
    )


def _oos_metrics(
    prices: np.ndarray,
    strategy: str,
    params: StrategyParams,
    cursor: int,
    test_bars: int,
) -> dict[str, float]:
    """Hard book on the causal prefix. Score only the held-out bars."""
    book = harden(params)
    prefix = np.asarray(prices[: cursor + test_bars], dtype=np.float64)
    sim = simulate(prefix, strategy, book)
    sl = np.asarray(sim.net[cursor : cursor + test_bars], dtype=np.float64)
    return {
        "oos_sharpe": sharpe(sl, book.periods_per_year),
        "oos_terminal_pnl": terminal_pnl(sl),
        "oos_sum_pnl": sum_pnl(sl),
        "oos_max_drawdown": drawdown(sl),
    }


def _smooth_grad_norm(
    prices: np.ndarray,
    strategy: str,
    params: StrategyParams,
    *,
    beta: float,
    score_start: int,
) -> float:
    grad = objective_gradients(
        prices,
        strategy,
        params,
        "sharpe",
        mode="smooth",
        beta=float(beta),
        score_start=int(score_start),
    )
    return float(np.linalg.norm(grad.parameter_gradient))


def walk_forward_compare(
    prices: np.ndarray,
    strategy: str,
    params: StrategyParams | None = None,
    *,
    train_bars: int,
    test_bars: int,
    lam: float = SMALL_CASE_LAMBDA,
    adam_steps: int = SMALL_CASE_ADAM_STEPS,
    beta: float = SMALL_CASE_BETA,
    data_source: str = "unspecified",
) -> dict[str, Any]:
    """Expanding walk-forward. Selectors see ``prices[:cursor]`` only.

    Both selectors are scored out of sample with the hard book on the causal
    prefix. Neither is chosen after seeing the test fold. Pooled diagnostics
    concatenate the out-of-sample net returns and compound across the fold
    boundary (no capital reset).
    """
    if train_bars < 8 or test_bars < 2:
        raise ValueError("train_bars >= 8 and test_bars >= 2 required")
    book = params or StrategyParams()
    validate_params(strategy, book)
    px = np.asarray(prices, dtype=np.float64)
    folds: list[dict[str, Any]] = []
    grid_chunks: list[np.ndarray] = []
    flat_chunks: list[np.ndarray] = []
    cursor = int(train_bars)
    while cursor + int(test_bars) <= px.shape[0]:
        train = np.array(px[:cursor], copy=True)
        grid = grid_search(train, strategy, book, data_source=data_source)
        flat = optimize_flat(
            train,
            strategy,
            book,
            lam=lam,
            steps=adam_steps,
            beta=beta,
            data_source=data_source,
        )
        grid_book = grid["hardened"]
        flat_book = flat["hardened"]
        grid_oos = _oos_metrics(px, strategy, grid_book, cursor, int(test_bars))
        flat_oos = _oos_metrics(px, strategy, flat_book, cursor, int(test_bars))
        # Keep the raw net slices for the pool by re-simulating once.
        grid_net = simulate(px[: cursor + test_bars], strategy, harden(grid_book)).net
        flat_net = simulate(px[: cursor + test_bars], strategy, harden(flat_book)).net
        grid_chunks.append(np.asarray(grid_net[cursor : cursor + test_bars], dtype=np.float64))
        flat_chunks.append(np.asarray(flat_net[cursor : cursor + test_bars], dtype=np.float64))
        score_start = warmup_start(strategy, book)
        folds.append(
            {
                "cursor": cursor,
                "train_rows": int(train.shape[0]),
                "test_rows": int(test_bars),
                "grid": {
                    "status": grid["status"],
                    "train_hard_sharpe": _hard_train_sharpe(train, strategy, grid_book),
                    "train_grad_norm": _smooth_grad_norm(
                        train, strategy, grid_book, beta=beta, score_start=score_start
                    ),
                    **grid_oos,
                },
                "flat": {
                    "status": flat["status"],
                    "improved": flat["improved"],
                    "train_objective": flat["end_objective"],
                    "train_hard_sharpe": _hard_train_sharpe(train, strategy, flat_book),
                    "train_grad_norm": flat["end_grad_norm"],
                    "start_grad_norm": flat["start_grad_norm"],
                    **flat_oos,
                },
            }
        )
        cursor += int(test_bars)
    if not folds:
        raise ValueError("price path is shorter than train_bars + test_bars")

    def _pool(chunks: list[np.ndarray]) -> dict[str, float | int]:
        cat = np.concatenate(chunks)
        return {
            "n_bars": int(cat.size),
            "oos_sharpe": sharpe(cat, book.periods_per_year),
            "oos_terminal_pnl": terminal_pnl(cat),
            "oos_sum_pnl": sum_pnl(cat),
            "oos_max_drawdown": drawdown(cat),
        }

    return {
        "strategy": strategy,
        "train_bars": int(train_bars),
        "test_bars": int(test_bars),
        "n_folds": len(folds),
        "folds": folds,
        "pooled_grid": _pool(grid_chunks),
        "pooled_flat": _pool(flat_chunks),
        "limitations": limitations(),
        **_label(data_source),
    }


def _pgd_eps(
    prices: np.ndarray,
    strategy: str,
    params: StrategyParams,
    sigma: np.ndarray,
    rho: float,
    steps: int,
    beta: float,
) -> np.ndarray:
    """L-infinity PGD on the smooth terminal P&L. Minimizes; projects into [-rho, rho]."""
    jax, jnp = _libs()
    book = harden(params)
    names = active_parameters(strategy)
    theta = jnp.asarray(pack(book, names))
    lookback_i, skip_i, vol_i, topk_i, long_only_i, require_i, long_short_i = _static_ints(book)
    periods = float(book.periods_per_year)
    delay = int(book.delay)
    every = int(book.rebalance_every)
    px = jnp.asarray(np.asarray(prices, dtype=np.float64))
    sig = jnp.asarray(np.asarray(sigma, dtype=np.float64).reshape(1, -1))
    lo, hi = _RETURN_CLIP

    def pnl(eps: Any) -> Any:
        raw = _returns(px)
        shocked = jnp.clip(raw + eps * sig, lo, hi)
        shocked = shocked.at[0].set(0.0)
        growth = jnp.cumprod(1.0 + shocked, axis=0)
        path = px[0] * growth
        _w, _r, _t, _c, net, _nav = _forward(
            path,
            theta,
            strategy=strategy,
            mode="smooth",
            beta=float(beta),
            delay=delay,
            every=every,
            periods=periods,
            initial_nav=1.0,
            names=names,
            lookback_i=lookback_i,
            skip_i=skip_i,
            vol_i=vol_i,
            topk_i=topk_i,
            long_only_i=long_only_i,
            require_i=require_i,
            long_short_i=long_short_i,
        )
        return _objective_from_net(
            net,
            "pnl",
            initial_nav=1.0,
            periods=periods,
            score_start=0,
            beta=float(beta),
            smooth_drawdown=True,
        )

    grad_fn = jax.jit(jax.grad(pnl))
    eps = jnp.full_like(px, -0.25 * float(rho))
    eps = eps.at[0].set(0.0)
    for _ in range(int(steps)):
        g = grad_fn(eps)
        eps = jnp.clip(eps - float(rho) * jnp.sign(g), -float(rho), float(rho))
        eps = eps.at[0].set(0.0)
    return np.asarray(eps, dtype=np.float64)


def _certify(
    prices: np.ndarray,
    strategy: str,
    params: StrategyParams,
    eps: np.ndarray,
    sigma: np.ndarray,
) -> float:
    path = perturb_prices(prices, eps, sigma, return_clip=_RETURN_CLIP)
    sim = simulate(path, strategy, harden(params))
    return float(sim.terminal_pnl)


def adversarial_radius(
    prices: np.ndarray,
    strategy: str,
    params: StrategyParams | None = None,
    *,
    rho_max: float = SMALL_CASE_RHO_MAX,
    pgd_steps: int = SMALL_CASE_PGD_STEPS,
    bisections: int = SMALL_CASE_BISECTIONS,
    beta: float = SMALL_CASE_BETA,
    data_source: str = "unspecified",
) -> dict[str, Any]:
    """Smallest probed L-infinity ball, in volatility units, that destroys terminal P&L.

    Destruction means the hard book's compounded P&L is ≤ 0. The search only
    lowers the upper bound when NumPy certifies the attacked path. If the
    unperturbed book is already non-positive, the radius is 0. If projected
    gradient never certifies a hit inside ``rho_max``, the radius is ``None``
    (the attack failed; that is not a proof of robustness).
    """
    if rho_max <= 0 or pgd_steps < 1 or bisections < 1 or beta <= 0:
        raise ValueError("rho_max > 0, pgd_steps >= 1, bisections >= 1, beta > 0 required")
    book = harden(params or StrategyParams())
    validate_params(strategy, book)
    px = np.asarray(prices, dtype=np.float64)
    base = float(simulate(px, strategy, book).terminal_pnl)
    sigma = path_sigma(px)
    zeros = np.zeros_like(px)
    common = {
        "strategy": strategy,
        "rho_max": float(rho_max),
        "beta": float(beta),
        "base_terminal_pnl": base,
        "sigma": sigma,
        "note": (
            "Radius is the smallest volatility-scaled L-infinity ball on which "
            "projected gradient found a destroying path. It is an upper bound, "
            "not a certificate."
        ),
        **_label(data_source),
    }
    if base <= 0.0:
        return {
            **common,
            "status": "already_nonpositive",
            "radius": 0.0,
            "certified": True,
            "attacked_terminal_pnl": base,
            "eps": zeros,
            "eps_linf": 0.0,
        }

    def probe(rho: float) -> tuple[np.ndarray, float, bool]:
        eps = _pgd_eps(px, strategy, book, sigma, rho, pgd_steps, beta)
        pnl = _certify(px, strategy, book, eps, sigma)
        return eps, pnl, bool(pnl <= 0.0)

    eps, pnl, ok = probe(float(rho_max))
    if not ok:
        return {
            **common,
            "status": "not_found",
            "radius": None,
            "certified": False,
            "attacked_terminal_pnl": pnl,
            "eps": eps,
            "eps_linf": float(np.max(np.abs(eps))),
        }
    lo = 0.0
    hi = float(rho_max)
    best_eps = eps
    best_pnl = pnl
    for _ in range(int(bisections)):
        mid = 0.5 * (lo + hi)
        eps, pnl, ok = probe(mid)
        if ok:
            hi = mid
            best_eps = eps
            best_pnl = pnl
        else:
            lo = mid
    return {
        **common,
        "status": "certified",
        "radius": float(hi),
        "certified": True,
        "attacked_terminal_pnl": float(best_pnl),
        "eps": best_eps,
        "eps_linf": float(np.max(np.abs(best_eps))),
        "search_lo": float(lo),
    }


def run_small_case() -> dict[str, Any]:
    """Fixed SYNTHETIC case used by CI. Constants live in ``spec`` and are not retuned."""
    prices = synthetic_prices(
        SMALL_CASE_T,
        SMALL_CASE_N,
        SMALL_CASE_SEED,
        mu=SMALL_CASE_MU,
        sigma=SMALL_CASE_SIGMA,
    )
    book = StrategyParams()
    reports: list[dict[str, Any]] = []
    for strategy in _SMALL_CASE_STRATEGIES:
        sens = sensitivity_map(
            prices, strategy, book, beta=SMALL_CASE_BETA, data_source=DATA_SOURCE_SYNTHETIC
        )
        folds = walk_forward_compare(
            prices,
            strategy,
            book,
            train_bars=SMALL_CASE_TRAIN,
            test_bars=SMALL_CASE_TEST,
            lam=SMALL_CASE_LAMBDA,
            adam_steps=SMALL_CASE_ADAM_STEPS,
            beta=SMALL_CASE_BETA,
            data_source=DATA_SOURCE_SYNTHETIC,
        )
        radius = adversarial_radius(
            prices,
            strategy,
            book,
            rho_max=SMALL_CASE_RHO_MAX,
            pgd_steps=SMALL_CASE_PGD_STEPS,
            bisections=SMALL_CASE_BISECTIONS,
            beta=SMALL_CASE_BETA,
            data_source=DATA_SOURCE_SYNTHETIC,
        )
        reports.append(
            {
                "strategy": strategy,
                "sensitivity": {
                    name: {
                        "value": sens["objectives"][name]["value"],
                        "gradient_norm": sens["objectives"][name]["gradient_norm"],
                        "price_gradient_norm": sens["objectives"][name]["price_gradient_norm"],
                    }
                    for name in OBJECTIVES
                },
                "walk_forward": {
                    "n_folds": folds["n_folds"],
                    "train_rows": [fold["train_rows"] for fold in folds["folds"]],
                    "pooled_grid": folds["pooled_grid"],
                    "pooled_flat": folds["pooled_flat"],
                    "folds": folds["folds"],
                },
                "adversarial": {
                    "status": radius["status"],
                    "radius": radius["radius"],
                    "base_terminal_pnl": radius["base_terminal_pnl"],
                    "attacked_terminal_pnl": radius["attacked_terminal_pnl"],
                    "certified": radius["certified"],
                    "eps_linf": radius["eps_linf"],
                },
            }
        )
    return {
        "case": "small",
        "seed": SMALL_CASE_SEED,
        "T": SMALL_CASE_T,
        "N": SMALL_CASE_N,
        "train_bars": SMALL_CASE_TRAIN,
        "test_bars": SMALL_CASE_TEST,
        "lambda": SMALL_CASE_LAMBDA,
        "beta": SMALL_CASE_BETA,
        "strategies": reports,
        "limitations": limitations(),
        "note": (
            "Two expanding folds on a 96-bar SYNTHETIC path have no power to "
            "claim an edge. Grid and flat are both hard-book out-of-sample scores."
        ),
        **_label(DATA_SOURCE_SYNTHETIC),
    }
