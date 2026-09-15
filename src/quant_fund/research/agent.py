"""Artificial Hedge scientific lab runner.

Families: ranking, alpha, volatility, distribution, regime, tail,
drawdown, liquidity, contextual bandits, conformal, e-values,
Jackknife+, CRC, weighted CQR, interval caps, quantile Thompson.
Proper scores only — no Sharpe.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from quant_fund import __firm__, __version__
from quant_fund.config.models import AppConfig
from quant_fund.metrics.inference import (
    benjamini_hochberg,
    diebold_mariano,
    mean_difference_t,
    mean_tstat,
    onesided_from_twosided,
    two_proportion_test,
)
from quant_fund.pipeline.dataset import build_gold, panel
from quant_fund.reporting.report import latest_report_dir, write_report
from quant_fund.research.benches import (
    bench_alpha,
    bench_conformal,
    bench_crc,
    bench_distribution,
    bench_drawdown,
    bench_evalues,
    bench_interval_risk,
    bench_jackknife_plus,
    bench_liquidity,
    bench_quantile_bandit,
    bench_ranking,
    bench_regime,
    bench_rl,
    bench_tail,
    bench_volatility,
    bench_weighted_conformal,
)
from quant_fund.utils.seeds import set_global_seed


@dataclass
class HypothesisResult:
    id: str
    statement: str
    test: str
    statistic: float
    p_value: float
    reject_raw: bool
    reject_fdr: bool
    decision: str
    family: str = "discovery"
    meets_floor: bool | None = None


@dataclass
class ResearchNotebook:
    firm: str
    product: str
    version: str
    generated_at: str
    data_source: str
    synthetic: bool
    disclaimer: str
    ranking_target: str
    families: dict[str, Any]
    rankers: list[dict[str, Any]]
    hypotheses: list[HypothesisResult]
    artifacts: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if obj is None:
        return None
    return str(obj)


def _hyp(
    hid: str,
    statement: str,
    test: str,
    stat: float,
    p: float,
    yes: str,
    no: str,
    alpha: float = 0.05,
    *,
    family: str = "discovery",
    meets_floor: bool | None = None,
) -> HypothesisResult:
    rej = bool(np.isfinite(p) and p < alpha)
    return HypothesisResult(
        id=hid,
        statement=statement,
        test=test,
        statistic=float(stat) if np.isfinite(stat) else float("nan"),
        p_value=float(p) if np.isfinite(p) else float("nan"),
        reject_raw=rej,
        reject_fdr=False,
        decision=yes if rej else no,
        family=family,
        meets_floor=meets_floor,
    )


def _series_array(blob: dict[str, Any], *keys: str) -> np.ndarray | None:
    for key in keys:
        raw = blob.get(key)
        if raw is None or isinstance(raw, (str, bytes, dict)):
            continue
        if isinstance(raw, (int, float, np.floating, np.integer, bool)):
            continue
        try:
            arr = np.asarray(raw, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            continue
        if int(np.isfinite(arr).sum()) >= 3:
            return arr
    return None


def _first_int(blob: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        v = blob.get(key)
        if v is None or isinstance(v, (list, tuple, dict, np.ndarray)):
            continue
        try:
            n = int(v)
        except (TypeError, ValueError):
            continue
        if n > 0:
            return n
    return None


def _contrast_inference(
    blob: dict[str, Any],
    mean_gap: float,
    *,
    series_keys: tuple[str, ...] = (),
    baseline_keys: tuple[str, ...] = (),
    gap_keys: tuple[str, ...] = (),
    n_keys: tuple[str, ...] = ("n_dates", "n"),
    loss_dm: bool = False,
) -> tuple[float, float, str]:
    """Real p-value for a mean gap. Never the dummy map p=0 if better else 1."""
    gap_s = _series_array(blob, *gap_keys) if gap_keys else None
    a = _series_array(blob, *series_keys) if series_keys else None
    b = _series_array(blob, *baseline_keys) if baseline_keys else None
    if gap_s is not None:
        _mu, t, p = mean_tstat(gap_s)
        return float(t), float(p), "HAC t-stat of date-level gap"
    if a is not None and b is not None:
        n_pair = min(int(a.size), int(b.size))
        a, b = a[:n_pair], b[:n_pair]
        if loss_dm:
            dm = diebold_mariano(a, b, name_a="model", name_b="baseline")
            return float(dm.statistic), float(dm.p_value), "Diebold–Mariano on date-level loss"
        _mu, t, p = mean_tstat(a - b)
        return float(t), float(p), "HAC t-stat of paired date-level difference"
    n_rep = _first_int(blob, *n_keys)
    t, p = mean_difference_t(mean_gap, n_rep or 0)
    return t, p, "mean-difference t using reported n_dates, not dummy 0/1"


def _apply_family_fdr(
    hyps: list[HypothesisResult], family: str, alpha: float = 0.05
) -> None:
    idx = [i for i, h in enumerate(hyps) if h.family == family]
    if not idx:
        return
    pvals = np.array([hyps[i].p_value for i in idx], dtype=float)
    finite = np.isfinite(pvals)
    reject = np.zeros(len(idx), dtype=bool)
    if finite.any():
        sub, _ = benjamini_hochberg(pvals[finite], alpha=alpha)
        reject[np.where(finite)[0]] = sub
    for j, r in zip(idx, reject, strict=True):
        hyps[j].reject_fdr = bool(r)


def _build_hypotheses(
    families: dict[str, Any],
    rankers: list[dict[str, Any]],
) -> list[HypothesisResult]:
    hyps: list[HypothesisResult] = []
    by = {r["name"]: r for r in rankers}
    oracle = by.get("oracle_raw")
    if oracle is not None:
        hyps.append(
            _hyp(
                "H1_ranking_oracle",
                "Labeled SYNTHETIC oracle has positive date-level ranking IC.",
                "HAC t-stat of date IC",
                float(oracle.get("t_ic", float("nan"))),
                float(oracle.get("p_ic", float("nan"))),
                "Oracle ranking signal recovered.",
                "Oracle IC not significant.",
                family="discovery",
            )
        )
        hyps.append(
            _hyp(
                "H2_decile_mono",
                "Oracle decile long-short mean is positive (ranking science).",
                "HAC t-stat of decile LS",
                float(oracle.get("ls_t", float("nan"))),
                float(oracle.get("ls_p", float("nan"))),
                "Deciles are informative.",
                "Decile spread not distinguishable from 0.",
                family="discovery",
            )
        )
    vol = families.get("volatility") or {}
    if vol.get("dm_p") is not None:
        hyps.append(
            _hyp(
                "H3_vol_dm",
                "EWMA and rolling 20d vol forecasts have unequal MSE (Diebold–Mariano).",
                "Diebold–Mariano",
                float(vol.get("dm_stat", float("nan"))),
                float(vol.get("dm_p", float("nan"))),
                f"DM prefers {vol.get('dm_preferred')}.",
                "Equal vol-forecast accuracy not rejected.",
                family="discovery",
            )
        )
    tail = families.get("tail") or {}
    if tail.get("kupiec_p") is not None:
        hyps.append(
            _hyp(
                "H4_var_kupiec",
                "Vol-scaled historical 95% VaR hit rate matches the 5% nominal (Kupiec POF).",
                "Kupiec POF",
                float(tail.get("kupiec_lr", float("nan"))),
                float(tail.get("kupiec_p", float("nan"))),
                "Hit rate rejects the nominal 5% (miscalibrated or small sample).",
                "VaR hits consistent with 5% nominal.",
                family="calibration",
            )
        )
    dd = families.get("drawdown") or {}
    if dd.get("brier") is not None and dd.get("brier_base_rate") is not None:
        gap = float(dd["brier_base_rate"] - dd["brier"])
        t, p_two, test = _contrast_inference(
            dd,
            gap,
            series_keys=("brier_by_date", "brier_clf_by_date"),
            baseline_keys=("brier_base_by_date", "brier_base_rate_by_date"),
            gap_keys=("brier_gap_by_date",),
            n_keys=("n_dates", "n_test", "n"),
            loss_dm=True,
        )
        # DM: mean(clf − base); negative t means lower Brier. Gap HAC: positive t means win.
        greater = "Diebold" not in test
        p = onesided_from_twosided(t, p_two, greater=greater)
        hyps.append(
            _hyp(
                "H5_drawdown_brier",
                "Drawdown classifier Brier beats the unconditional base rate.",
                test + " (one-sided, lower Brier)",
                gap,
                p,
                "Classifier improves on the base rate.",
                "Classifier does not beat the base rate.",
                family="discovery",
            )
        )
    rl = families.get("reinforcement") or {}
    if rl.get("mean_advantage_vs_ridge") is not None:
        adv = float(rl["mean_advantage_vs_ridge"])
        t, p_two, test = _contrast_inference(
            rl,
            adv,
            series_keys=("policy_reward",),
            baseline_keys=("ridge_reward",),
            gap_keys=("reward_gap_series",),
        )
        p = onesided_from_twosided(t, p_two, greater=True)
        hyps.append(
            _hyp(
                "H6_linucb_vs_uniform",
                "LinUCB top-k reward exceeds a static public ridge top-k policy "
                "(same public features, same dates).",
                test + " (one-sided vs ridge)",
                adv,
                p,
                "LinUCB beats static public ridge on the scientific reward.",
                "LinUCB does not beat static public ridge.",
                family="discovery",
            )
        )
    conf = families.get("conformal") or {}
    aci = conf.get("aci") if isinstance(conf, dict) else None
    if isinstance(aci, dict) and aci.get("kupiec_p") is not None:
        hyps.append(
            _hyp(
                "H7_aci_coverage",
                "ACI 90% prediction-set miss rate wrapping the operational scaled wrappee matches nominal α=0.10 (Kupiec on conformal misses).",
                "Kupiec POF on ACI misses",
                float(aci.get("kupiec_lr", float("nan"))),
                float(aci.get("kupiec_p", float("nan"))),
                "ACI miss rate differs from 10% nominal.",
                "ACI coverage consistent with 90% nominal.",
                family="calibration",
            )
        )
    mond = conf.get("mondrian_aci") if isinstance(conf, dict) else None
    if isinstance(mond, dict) and mond.get("high_x_kupiec_p") is not None:
        hyps.append(
            _hyp(
                "H8_mondrian_high_vol",
                "Mondrian ACI high-vol (X) slice miss rate matches nominal α=0.10 (not a |Y|-slice guarantee).",
                "Kupiec POF on Mondrian high-X misses",
                float(mond.get("high_x_kupiec_lr", float("nan"))),
                float(mond.get("high_x_kupiec_p", float("nan"))),
                "High-vol Mondrian miss rate differs from 10% nominal.",
                "High-vol Mondrian coverage consistent with 90% nominal.",
                family="calibration",
            )
        )
    ev = families.get("evalues") or {}
    if ev.get("e_sup") is not None:
        e_sup = float(ev["e_sup"])
        p_ville = float(min(1.0, 1.0 / max(e_sup, 1.0)))
        hyps.append(
            _hyp(
                "H9_eprocess_aci",
                "ACI miss e-process is consistent with nominal α=0.10 (Ville, level 0.05).",
                "Anytime-valid Bernoulli e-process (Ville)",
                e_sup,
                p_ville,
                "E-process crossed 20: too many ACI misses for α=0.10.",
                "E-process stayed below 20; ACI misses consistent with α=0.10.",
                family="calibration",
            )
        )
    jp = families.get("jackknife_plus") or {}
    if jp.get("coverage") is not None:
        jp_cov = float(jp["coverage"])
        floor = float(jp.get("coverage_floor", 0.8))
        meets = bool(np.isfinite(jp_cov) and jp_cov + 1e-12 >= floor)
        hyps.append(
            HypothesisResult(
                id="H10_jackknife_coverage",
                statement="Jackknife+ coverage meets the 1-2α finite-sample floor.",
                test="coverage − (1−2α) floor check; not a Kupiec null, not FDR",
                statistic=jp_cov - floor if np.isfinite(jp_cov) else float("nan"),
                p_value=float("nan"),
                reject_raw=False,
                reject_fdr=False,
                decision="Jackknife+ coverage is above the 1-2α floor."
                if meets
                else "Jackknife+ coverage is below the 1-2α floor.",
                family="bound",
                meets_floor=meets,
            )
        )
    crc = families.get("crc") or {}
    if crc.get("kupiec_p") is not None:
        hyps.append(
            _hyp(
                "H11_crc_var",
                "CRC 95% VaR-style hit rate on the scaled wrappee matches the 5% nominal (Kupiec POF).",
                "Kupiec POF on CRC hits",
                float(crc.get("kupiec_lr", float("nan"))),
                float(crc.get("kupiec_p", float("nan"))),
                "CRC hit rate rejects the nominal 5%.",
                "CRC hits consistent with 5% nominal.",
                family="calibration",
            )
        )
    wcqr = families.get("weighted_conformal") or {}
    if wcqr.get("kupiec_p") is not None:
        hyps.append(
            _hyp(
                "H12_weighted_cqr",
                "Weighted split CQR miss rate matches nominal α=0.10 (Kupiec POF).",
                "Kupiec POF on weighted CQR misses",
                float(wcqr.get("kupiec_lr", float("nan"))),
                float(wcqr.get("kupiec_p", float("nan"))),
                "Weighted CQR miss rate differs from 10% nominal.",
                "Weighted CQR coverage consistent with 90% nominal.",
                family="calibration",
            )
        )
    caps = families.get("interval_risk") or {}
    if caps.get("bind_wide") is not None and caps.get("bind_tight") is not None:
        wide_b = float(caps["bind_wide"])
        tight_b = float(caps["bind_tight"])
        contrast = (
            wide_b - tight_b if np.isfinite(wide_b) and np.isfinite(tight_b) else float("nan")
        )
        gap_s = _series_array(caps, "bind_gap_by_date")
        n_wide = caps.get("n_wide")
        n_tight = caps.get("n_tight")
        if gap_s is not None:
            _mu, t, p_two = mean_tstat(gap_s)
            p = onesided_from_twosided(t, p_two, greater=True)
            test = "one-sided HAC t on date-level bind_wide − bind_tight"
        elif n_wide is not None and n_tight is not None:
            _z, p = two_proportion_test(
                int(caps.get("n_bind_wide") or 0),
                int(n_wide),
                int(caps.get("n_bind_tight") or 0),
                int(n_tight),
                alternative="greater",
            )
            test = "one-sided two-proportion test of bind rates"
        else:
            _t, p = mean_difference_t(contrast, _first_int(caps, "n_dates", "n") or 0)
            test = "mean-difference t using reported n_dates, not dummy 0/1"
        weight_rule = str(caps.get("weight_rule") or "")
        if weight_rule == "equal_weight_per_date":
            statement = (
                "Equal-weight names (1/n on the date) in the wide-interval half "
                "bind more than the tight half."
            )
        else:
            statement = "Wide-interval names bind more than tight-interval names after interval caps."
        hyps.append(
            _hyp(
                "H13_interval_caps",
                statement,
                test,
                contrast,
                p,
                "Wide sets bind more than tight sets.",
                "Interval caps do not bind the wide half more.",
                family="discovery",
            )
        )
    qb = families.get("quantile_bandit") or {}
    if qb.get("mean_advantage_vs_ridge") is not None:
        adv = float(qb["mean_advantage_vs_ridge"])
        t, p_two, test = _contrast_inference(
            qb,
            adv,
            series_keys=("policy_reward",),
            baseline_keys=("ridge_reward",),
            gap_keys=("reward_gap_series",),
        )
        p = onesided_from_twosided(t, p_two, greater=True)
        hyps.append(
            _hyp(
                "H14_quantile_thompson",
                "Quantile Thompson top-k reward exceeds a static public ridge top-k policy "
                "(same public features, same dates).",
                test + " (one-sided vs ridge)",
                adv,
                p,
                "Quantile Thompson beats static public ridge on the scientific reward.",
                "Quantile Thompson does not beat static public ridge.",
                family="discovery",
            )
        )
    _apply_family_fdr(hyps, "calibration")
    _apply_family_fdr(hyps, "discovery")
    return hyps


def run_research(config: AppConfig) -> ResearchNotebook:
    set_global_seed(config.train.random_seed)
    build_gold(config, refresh=config.data.source == "synthetic")
    df = panel(config)
    if "security_id" in df.columns:
        df = df.filter(pl.col("security_id") != config.data.benchmark_id)
    label = config.train.ranking_target
    if label not in df.columns:
        cands = [c for c in df.columns if c.startswith("future_idio_return")]
        if not cands:
            cands = [c for c in df.columns if c.startswith("future_excess_return")]
        label = cands[0] if cands else "future_return_1"

    rankers = bench_ranking(df, config, label)
    families = {
        "ranking": {"target": label, "models": rankers},
        "alpha": bench_alpha(df, config, label),
        "volatility": bench_volatility(df, config),
        "distribution": bench_distribution(df, config),
        "regime": bench_regime(df, config),
        "tail": bench_tail(df, config),
        "drawdown": bench_drawdown(df, config),
        "liquidity": bench_liquidity(df),
        "reinforcement": bench_rl(df, label),
        "conformal": bench_conformal(df, config),
        "evalues": bench_evalues(df, config),
        "jackknife_plus": bench_jackknife_plus(df, config),
        "crc": bench_crc(df, config),
        "weighted_conformal": bench_weighted_conformal(df, config),
        "interval_risk": bench_interval_risk(df, config),
        "quantile_bandit": bench_quantile_bandit(df, label),
    }

    hyps = _build_hypotheses(families, rankers)

    synthetic = config.data.source == "synthetic"
    disclaimer = (
        "SYNTHETIC scientific lab. Scores are proper rules (IC, QLIKE, pinball, "
        "CRPS, Brier, Kupiec, CQR/ACI/Mondrian coverage, e-process, CRC, "
        "Jackknife+, weighted CQR, interval caps, bandit regret). Not a live-P&L claim."
        if synthetic
        else "Scientific lab scores with HAC/DM/Kupiec uncertainty. No live P&L claim."
    )

    notebook = ResearchNotebook(
        firm=__firm__,
        product="Dipcatcher",
        version=__version__,
        generated_at=datetime.now(tz=UTC).isoformat(),
        data_source="SYNTHETIC" if synthetic else str(config.data.source),
        synthetic=synthetic,
        disclaimer=disclaimer,
        ranking_target=label,
        families=families,
        rankers=rankers,
        hypotheses=hyps,
    )

    dest_dir = Path(config.data.root) / "metadata" / "research"
    dest_dir.mkdir(parents=True, exist_ok=True)
    sections: dict[str, Any] = {
        "disclaimer": disclaimer,
        "lab": {
            "firm": __firm__,
            "product": "Dipcatcher",
            "role": "scientific hedge research lab",
            "target": label,
            "source": notebook.data_source,
        },
        "hypotheses": {
            h.id: (
                f"[{h.family}] {h.decision} | {h.test} "
                f"stat={h.statistic:.4g} "
                f"p={'n/a' if not np.isfinite(h.p_value) else f'{h.p_value:.4g}'}"
                + (
                    f" meets_floor={h.meets_floor}"
                    if h.family == "bound"
                    else f" reject_fdr={h.reject_fdr}"
                )
            )
            for h in hyps
        },
        "ranking": {
            r["name"]: (
                f"set={r['feature_set']} IC={r['mean_ic']:.4f} RankIC={r.get('mean_rank_ic') or 0:.4f} "
                f"t={r['t_ic']:.2f} mono={r.get('decile_monotonicity') or 0:.3f}"
            )
            for r in rankers
        },
        "alpha": families["alpha"],
        "volatility": families["volatility"],
        "distribution": families["distribution"],
        "regime": families["regime"],
        "tail": families["tail"],
        "drawdown": families["drawdown"],
        "liquidity": families["liquidity"],
        "reinforcement": families["reinforcement"],
        "conformal": families["conformal"],
        "evalues": families["evalues"],
        "jackknife_plus": families["jackknife_plus"],
        "crc": families["crc"],
        "weighted_conformal": families["weighted_conformal"],
        "interval_risk": families["interval_risk"],
        "quantile_bandit": families["quantile_bandit"],
    }
    md_path = dest_dir / "latest.md"
    write_report(
        md_path,
        f"{__firm__} · Dipcatcher scientific lab notebook",
        sections,
        synthetic=synthetic,
    )
    write_report(
        latest_report_dir(Path(config.data.root)) / "latest.md",
        f"{__firm__} · Dipcatcher scientific lab notebook",
        sections,
        synthetic=synthetic,
    )
    notebook.artifacts = {"json": str(dest_dir / "latest.json"), "markdown": str(md_path)}
    (dest_dir / "latest.json").write_text(json.dumps(notebook.to_dict(), indent=2))
    return notebook
