"""Strategy specifications (faithful parameterizations + citations) and the
per-strategy evaluation driver.

Each spec names the signal builder, the formation parameters from the original
paper, the evaluation horizon(s), and the sign of the effect the paper reports.
``evaluate_strategy`` produces the receipt block; ``verdict`` is an explicit
honest verdict — "replicated" requires both the paper's sign and a |t| >= 2
Newey-West statistic on the primary IC horizon (resp. the in-window mean test
for the calendar strategy). Null and negative results are recorded, never
hidden.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import polars as pl

from quant_fund.research.replication.evaluation import (
    daily_ic,
    decile_spread,
    directional_scores,
    forward_returns,
    ic_summary,
    market_timing_eval,
)
from quant_fund.research.replication.signals import (
    bab_signal,
    low_volatility_signal,
    overnight_signal,
    reversal_signal,
    tsmom_signal,
    tug_of_war_signal,
    turn_of_month_signal,
)

CITATIONS: dict[str, str] = {
    "mop2012": (
        "Moskowitz, T. J., Ooi, Y. H., & Pedersen, L. H. (2012). "
        "Time series momentum. Journal of Financial Economics, 104(2), 228-250. "
        "doi:10.1016/j.jfineco.2011.11.003"
    ),
    "jegadeesh1990": (
        "Jegadeesh, N. (1990). Evidence of predictable behavior of security "
        "returns. Journal of Finance, 45(3), 881-898. doi:10.1111/j.1540-6261.1990.tb05110.x"
    ),
    "lehmann1990": (
        "Lehmann, B. N. (1990). Fads, martingales, and market efficiency. "
        "Quarterly Journal of Economics, 105(1), 1-28. doi:10.2307/2937816"
    ),
    "bbw2011": (
        "Baker, M., Bradley, B., & Wurgler, J. (2011). Benchmarks as limits to "
        "arbitrage: Understanding the low-volatility anomaly. Financial Analysts "
        "Journal, 67(1), 40-54. doi:10.2469/faj.v67.n1.4"
    ),
    "fp2014": (
        "Frazzini, A., & Pedersen, L. H. (2014). Betting against beta. "
        "Journal of Financial Economics, 111(1), 1-25. doi:10.1016/j.jfineco.2013.10.005"
    ),
    "lps2019": (
        "Lou, D., Polk, C., & Skouras, S. (2019). A tug of war: Overnight "
        "versus intraday expected returns. Journal of Financial Economics, "
        "134(1), 192-213. doi:10.1016/j.jfineco.2019.03.011"
    ),
    "ls1988": (
        "Lakonishok, J., & Smidt, S. (1988). Are seasonal anomalies real? A "
        "ninety-year perspective. Review of Financial Studies, 1(4), 403-425. "
        "doi:10.1093/rfs/1.4.403"
    ),
    "mx2008": (
        "McConnell, J. J., & Xu, W. (2008). Equity returns at the turn of the "
        "month. Financial Analysts Journal, 64(2), 49-64. doi:10.2469/faj.v64.n2.8"
    ),
}


@dataclass(frozen=True)
class StrategySpec:
    """One replicated strategy: signal builder + paper-faithful evaluation."""

    key: str
    paper_key: str
    signal_fn: str
    params: dict[str, Any]
    outcome_col: str = "ret_cc"  # which return the IC/deciles score against
    primary_horizon: int = 21  # IC horizon the verdict uses
    extra_horizons: tuple[int, ...] = (1,)
    expected_ic_sign: float = 1.0
    market_timing: bool = False
    needs_market: bool = False
    hypothesis: str = ""
    signal_definition: str = ""
    known_limitations: tuple[str, ...] = field(default=())


STRATEGY_SPECS: tuple[StrategySpec, ...] = (
    StrategySpec(
        key="tsmom_mop2012",
        paper_key="mop2012",
        signal_fn="tsmom_signal",
        params={"lookback": 252, "vol_com": 60},
        outcome_col="ret_cc",
        primary_horizon=21,
        extra_horizons=(1,),
        expected_ic_sign=1.0,
        hypothesis=(
            "H-TSMOM: sign(past 12M return)/ex-ante-vol positively predicts "
            "next-month cross-sectional returns (MOP2012 Table 1-style)."
        ),
        signal_definition="sign(sum log ret_cc, 252d) / ewm_std(ret_cc, com=60)",
    ),
    StrategySpec(
        key="str_jegadeesh1990",
        paper_key="jegadeesh1990",
        signal_fn="reversal_signal",
        params={"lookback": 21},
        outcome_col="ret_cc",
        primary_horizon=21,
        extra_horizons=(1,),
        expected_ic_sign=1.0,
        hypothesis=(
            "H-STR-1M: negative past-1M return positively predicts next-month "
            "returns (Jegadeesh 1990 monthly contrarian)."
        ),
        signal_definition="-(sum log ret_cc, 21d)",
    ),
    StrategySpec(
        key="str_lehmann1990",
        paper_key="lehmann1990",
        signal_fn="reversal_signal",
        params={"lookback": 5},
        outcome_col="ret_cc",
        primary_horizon=5,
        extra_horizons=(1,),
        expected_ic_sign=1.0,
        hypothesis=(
            "H-STR-1W: negative past-1W return positively predicts next-week "
            "returns (Lehmann 1990 weekly reversal)."
        ),
        signal_definition="-(sum log ret_cc, 5d)",
    ),
    StrategySpec(
        key="lowvol_bbw2011",
        paper_key="bbw2011",
        signal_fn="low_volatility_signal",
        params={"lookback": 252},
        outcome_col="ret_cc",
        primary_horizon=21,
        extra_horizons=(1,),
        expected_ic_sign=1.0,
        hypothesis=(
            "H-LOWVOL: lower trailing-1Y daily-return volatility predicts "
            "higher next-month returns (BBW 2011 Table-style quintiles)."
        ),
        signal_definition="-(rolling_std ret_cc, 252d)",
    ),
    StrategySpec(
        key="bab_fp2014",
        paper_key="fp2014",
        signal_fn="bab_signal",
        params={
            "lookback": 252,
            "shrink_weight": 0.6,
            "shrink_target": 1.0,
        },
        outcome_col="ret_cc",
        primary_horizon=21,
        extra_horizons=(1,),
        expected_ic_sign=1.0,
        needs_market=True,
        hypothesis=(
            "H-BAB: lower shrunk beta (0.6*beta_hat + 0.4) predicts higher "
            "next-month returns (Frazzini-Pedersen 2014)."
        ),
        signal_definition="-(0.6 * corr_i,M * sigma_i/sigma_M + 0.4), 252d daily",
        known_limitations=(
            "FP compute correlation on 1Y daily and vol on 5Y daily; the panel "
            "is short so both use the 252d window.",
            "Market is the equal-weight universe mean (no market caps).",
        ),
    ),
    StrategySpec(
        key="overnight_lps2019",
        paper_key="lps2019",
        signal_fn="overnight_signal",
        params={"lookback": 20},
        outcome_col="ret_on",
        primary_horizon=1,
        extra_horizons=(),
        expected_ic_sign=1.0,
        hypothesis=(
            "H-ON-PERSIST: cumulative past-20d overnight return predicts the "
            "next-day OVERNIGHT return (LPS 2019 overnight persistence)."
        ),
        signal_definition="sum(log(1+ret_on), 20d)",
    ),
    StrategySpec(
        key="tug_of_war_lps2019",
        paper_key="lps2019",
        signal_fn="tug_of_war_signal",
        params={"lookback": 20},
        outcome_col="ret_cc",
        primary_horizon=1,
        extra_horizons=(21,),
        expected_ic_sign=1.0,
        hypothesis=(
            "H-TUG: past (overnight - intraday) return predicts next-day total "
            "return (LPS 2019 tug-of-war; paper finds net total effect is weak)."
        ),
        signal_definition="sum(log(1+ret_on), 20d) - sum(log(1+ret_id), 20d)",
    ),
    StrategySpec(
        key="tom_mx2008",
        paper_key="mx2008",
        signal_fn="turn_of_month_signal",
        params={},
        market_timing=True,
        hypothesis=(
            "H-TOM: equal-weight market daily returns inside the -1..+3 "
            "turn-of-month window exceed outside-window returns "
            "(McConnell & Xu 2008; Lakonishok & Smidt 1988)."
        ),
        signal_definition="1 on last trading day of month + first 3 of next, else 0",
        known_limitations=(
            "Window is calendar-defined on the panel's own trading days; "
            "~24 months of daily data makes this a small-sample test.",
        ),
    ),
)

SIGNAL_BUILDERS = {
    "tsmom_signal": tsmom_signal,
    "reversal_signal": reversal_signal,
    "low_volatility_signal": low_volatility_signal,
    "bab_signal": bab_signal,
    "overnight_signal": overnight_signal,
    "tug_of_war_signal": tug_of_war_signal,
    "turn_of_month_signal": turn_of_month_signal,
}


def _verdict(sign_ok: bool, significant: bool) -> str:
    if significant and sign_ok:
        return "replicated"
    if significant and not sign_ok:
        return "contradicted"
    return "inconclusive"


def _market_strategy_block(market: pl.DataFrame, spec: StrategySpec) -> dict[str, Any]:
    flags = turn_of_month_signal(market.select(pl.lit("MKT").alias("security_id"), "event_time"))
    result = market_timing_eval(market, flags)
    diff = result.get("descriptive_in_minus_out", float("nan"))
    p_value = result.get("descriptive_welch_p", float("nan"))
    sign_ok = bool(math.isfinite(diff) and diff > 0.0)
    significant = bool(math.isfinite(p_value) and p_value < 0.05)
    descriptive = {k: v for k, v in result.items() if k.startswith("descriptive_")}
    scores = {k: v for k, v in result.items() if not k.startswith("descriptive_")}
    return {
        "scores": scores,
        "descriptive": descriptive,
        "verdict": {
            "verdict": _verdict(sign_ok, significant),
            "direction_consistent_with_paper": sign_ok,
            "significant_at_5pct": significant,
            "primary_statistic": "in_window_minus_out_window_mean",
            "primary_statistic_value": diff,
            "primary_p_value": p_value,
        },
    }


def _untestable_block(reason: str) -> dict[str, Any]:
    """Honest null block when the dataset cannot express the anomaly."""
    return {
        "scores": {"n_scored": 0.0},
        "descriptive": {},
        "verdict": {
            "verdict": "untestable",
            "untestable_reason": reason,
            "direction_consistent_with_paper": None,
            "significant_at_5pct": None,
            "primary_statistic": None,
            "primary_statistic_value": None,
            "primary_p_value": None,
        },
    }


def _degenerate_reason(panel: pl.DataFrame, signal: pl.DataFrame, spec: StrategySpec) -> str | None:
    """Detect inputs that cannot express the anomaly (fail honest, not NaN)."""
    sig = signal["signal"].drop_nulls()
    if sig.len() == 0 or (sig.std() or 0.0) == 0.0:
        return (
            "degenerate_signal: signal has zero variance on this panel "
            f"(built from {spec.signal_fn})"
        )
    fwd = forward_returns(panel, spec.primary_horizon, spec.outcome_col)["fwd_ret"].drop_nulls()
    if fwd.len() == 0 or (fwd.std() or 0.0) == 0.0:
        return (
            f"degenerate_outcome: {spec.outcome_col} has zero variance on this panel "
            "(the dataset cannot express this return leg)"
        )
    return None


def _data_notes(panel: pl.DataFrame, spec: StrategySpec) -> list[str]:
    """Dataset-specific caveats worth printing into the receipt."""
    notes: list[str] = []
    if spec.paper_key == "lps2019":
        on = panel["ret_on"].drop_nulls()
        if on.len() == 0 or (on.std() or 0.0) == 0.0:
            notes.append(
                "This panel has open == previous close for every bar, so the "
                "overnight leg is identically zero; the tug-of-war signal "
                "degenerates to -(cumulative intraday) ~ a 20d reversal signal."
            )
    return notes


def _cross_sectional_block(
    panel: pl.DataFrame,
    market: pl.DataFrame,
    spec: StrategySpec,
    *,
    ic_min_assets: int,
    n_dir_buckets: int,
    min_bucket_obs: int,
) -> dict[str, Any]:
    builder = SIGNAL_BUILDERS[spec.signal_fn]
    if spec.needs_market:
        signal = builder(panel, market, **spec.params)
    else:
        signal = builder(panel, **spec.params)
    reason = _degenerate_reason(panel, signal, spec)
    if reason is not None:
        block = _untestable_block(reason)
        block["data_notes"] = _data_notes(panel, spec)
        return block
    horizons = tuple({spec.primary_horizon, *spec.extra_horizons})
    scores: dict[str, Any] = {}
    descriptive: dict[str, Any] = {}
    for horizon in sorted(horizons):
        fwd = forward_returns(panel, horizon, spec.outcome_col)
        ic_frame = daily_ic(signal, fwd, min_assets=ic_min_assets)
        block = ic_summary(ic_frame, horizon)
        prefix = f"h{horizon}_"
        scores.update({prefix + k: v for k, v in block.items()})
        descriptive.update(
            {
                f"descriptive_h{horizon}_{k.removeprefix('descriptive_')}": v
                for k, v in decile_spread(signal, fwd).items()
            }
        )
    # Proper-score leg: next-day direction of the outcome column.
    fwd1 = (
        forward_returns(panel, 1, spec.outcome_col)
        .with_columns((pl.col("fwd_ret") > 0.0).cast(pl.Float64).alias("up"))
        .select("security_id", "event_time", "up")
    )
    scores.update(
        {
            f"direction_{k}": v
            for k, v in directional_scores(
                signal, fwd1, n_buckets=n_dir_buckets, min_bucket_obs=min_bucket_obs
            ).items()
        }
    )
    primary = spec.primary_horizon
    mean_ic = scores.get(f"h{primary}_mean_ic", float("nan"))
    t_stat = scores.get(f"h{primary}_ic_tstat_nw", float("nan"))
    sign_ok = bool(math.isfinite(mean_ic) and math.copysign(1.0, mean_ic) == spec.expected_ic_sign)
    significant = bool(math.isfinite(t_stat) and abs(t_stat) >= 2.0)
    data_notes = _data_notes(panel, spec)
    return {
        **({"data_notes": data_notes} if data_notes else {}),
        "scores": scores,
        "descriptive": descriptive,
        "verdict": {
            "verdict": _verdict(sign_ok, significant),
            "direction_consistent_with_paper": sign_ok,
            "significant_at_5pct": significant,
            "primary_statistic": f"h{primary}_ic_tstat_nw",
            "primary_statistic_value": t_stat,
            "primary_p_value": scores.get(f"h{primary}_ic_p_value_nw"),
        },
    }


def evaluate_strategy(
    panel: pl.DataFrame,
    market: pl.DataFrame,
    spec: StrategySpec,
    *,
    ic_min_assets: int = 8,
    n_dir_buckets: int = 5,
    min_bucket_obs: int = 40,
) -> dict[str, Any]:
    """Produce the full receipt block for one strategy spec."""
    if spec.market_timing:
        block = _market_strategy_block(market, spec)
    else:
        block = _cross_sectional_block(
            panel,
            market,
            spec,
            ic_min_assets=ic_min_assets,
            n_dir_buckets=n_dir_buckets,
            min_bucket_obs=min_bucket_obs,
        )
    return {
        "paper_key": spec.paper_key,
        "citation": CITATIONS[spec.paper_key],
        "hypothesis": spec.hypothesis,
        "signal_definition": spec.signal_definition,
        "params": dict(spec.params),
        "outcome_column": spec.outcome_col,
        "primary_horizon": spec.primary_horizon,
        "expected_ic_sign": spec.expected_ic_sign,
        "known_limitations": list(spec.known_limitations),
        **block,
    }
