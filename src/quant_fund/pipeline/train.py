"""Train forecast families on gold panels. Holdout is not used for Optuna."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from quant_fund.config.models import AppConfig
from quant_fund.metrics.cross_section import date_ic_series
from quant_fund.metrics.scoring import icir, mean_pinball, qlike, quantile_crossing_rate
from quant_fund.models.alpha import HistoricalMeanAlpha
from quant_fund.models.distribution import (
    EmpiricalDistribution,
    GaussianDistribution,
    LinearQuantileDistribution,
    TreeQuantileDistribution,
)
from quant_fund.models.ranking import (
    CompositeRanker,
    ElasticNetRanker,
    LGBMLambdaRanker,
    LGBMRegRanker,
    RidgeRanker,
    XGBRegRanker,
    group_sizes,
)
from quant_fund.models.regime import GaussianHMMRegime, SingleStateRegime, VolThresholdRegime
from quant_fund.models.tail import DrawdownClassifier, GaussianTail, HistoricalTail
from quant_fund.models.volatility import EWMAVol, GARCHVol, HARVol, RollingVol, TreeVol
from quant_fund.pipeline.dataset import design_matrix, panel
from quant_fund.registry.mlflow_store import (
    configure_tracking,
    log_run,
)
from quant_fund.utils.seeds import set_global_seed
from quant_fund.validation.walk_forward import walk_forward


def _split_fold(dates, x, y, fold):
    tr = np.isin(dates, np.array(fold.train_times, dtype=object))
    va = np.isin(dates, np.array(fold.val_times, dtype=object))
    te = np.isin(dates, np.array(fold.test_times, dtype=object))
    return tr, va, te


def train_ranking(config: AppConfig, model_name: str = "ridge") -> dict[str, Any]:
    set_global_seed(config.train.random_seed)
    df = panel(config)
    label = config.train.ranking_target
    if label not in df.columns:
        # pick first available excess
        cands = [c for c in df.columns if c.startswith("future_excess_return")]
        label = cands[0] if cands else "future_return_5"
    x, y, dates, feats, ids = design_matrix(df, label)
    times = sorted(set(dates.tolist()))
    folds = walk_forward(
        times,
        config.validation,
        horizon_bars=5,
        embargo_bars=config.embargo_bars(),
    )
    ics: list[float] = []
    rics: list[float] = []
    last_model: Any = None
    if not folds:
        # fit on all but last 40 days
        uniq = times
        cut = max(len(uniq) - 40, len(uniq) // 2)
        tr_times = uniq[:cut]
        te_times = uniq[cut:]
        tr = np.isin(dates, tr_times)
        te = np.isin(dates, te_times)
        model = _make_ranker(model_name, config)
        _fit_ranker(model, model_name, x[tr], y[tr], dates[tr])
        pred = model.predict(x[te])
        dated = date_ic_series(pred, y[te], dates[te], min_names=4)
        ics.extend(dated.pearson.tolist())
        rics.extend(dated.spearman.tolist())
        last_model = model
        date_metrics = dated
    else:
        fold = folds[-1]
        tr, va, te = _split_fold(dates, x, y, fold)
        model = _make_ranker(model_name, config)
        _fit_ranker(model, model_name, x[tr], y[tr], dates[tr])
        use_va = bool(va.any())
        pred = model.predict(x[va] if use_va else x[te])
        yy = y[va] if use_va else y[te]
        dd = dates[va] if use_va else dates[te]
        dated = date_ic_series(pred, yy, dd, min_names=4)
        ics.extend(dated.pearson.tolist())
        rics.extend(dated.spearman.tolist())
        last_model = model
        date_metrics = dated
    metrics = {
        "mean_ic": float(date_metrics.mean_pearson) if ics else float(np.nanmean(ics)),
        "mean_rank_ic": float(date_metrics.mean_spearman) if rics else float(np.nanmean(rics)),
        "icir": float(date_metrics.icir_pearson) if ics else icir(np.array(ics)),
        "ic_tstat": float(date_metrics.t_pearson),
        "ic_pvalue": float(date_metrics.p_pearson),
        "n_ic_dates": int(date_metrics.n_dates),
    }
    configure_tracking()
    run_id = log_run(
        family="ranking",
        name=model_name,
        params={"features": feats, "label": label, "model": model_name},
        metrics=metrics,
        tags={"git": "local", "data": config.data.source},
    )
    out = Path(config.data.root) / "metadata" / f"ranker_{model_name}.joblib"
    last_model.save(out)
    return {
        "metrics": metrics,
        "run_id": run_id,
        "path": str(out),
        "features": feats,
        "n": int(x.shape[0]),
    }


def _make_ranker(name: str, config: AppConfig) -> Any:
    t = config.train
    return {
        "composite": CompositeRanker(),
        "ridge": RidgeRanker(t.ridge_alpha),
        "elasticnet": ElasticNetRanker(t.ridge_alpha, t.elasticnet_l1),
        "xgboost": XGBRegRanker(t.xgb_n_estimators, t.xgb_max_depth, t.random_seed),
        "lightgbm": LGBMRegRanker(t.lgbm_n_estimators, t.lgbm_num_leaves, t.random_seed),
        "lambdarank": LGBMLambdaRanker(t.lgbm_n_estimators, t.random_seed, "lambdarank"),
        "xendcg": LGBMLambdaRanker(t.lgbm_n_estimators, t.random_seed, "rank_xendcg"),
    }[name]


def _fit_ranker(model: Any, name: str, x, y, dates) -> None:
    if name in {"lambdarank", "xendcg"}:
        order = np.argsort(dates, kind="mergesort")
        model.fit(x[order], y[order], group=group_sizes(dates[order]))
    else:
        model.fit(x, y)


def train_distribution(config: AppConfig, model_name: str = "gaussian") -> dict[str, Any]:
    set_global_seed(config.train.random_seed)
    df = panel(config)
    label = config.train.distribution_target
    if label not in df.columns:
        label = [c for c in df.columns if c.startswith("future_log_return")][0]
    x, y, dates, feats, _ = design_matrix(df, label)
    taus = config.quantiles.levels
    model: Any = {
        "empirical": EmpiricalDistribution(taus),
        "gaussian": GaussianDistribution(taus),
        "linear_qr": LinearQuantileDistribution(taus),
        "xgboost": TreeQuantileDistribution(taus, "xgboost", config.train.random_seed),
        "lightgbm": TreeQuantileDistribution(taus, "lightgbm", config.train.random_seed),
    }[model_name]
    n = x.shape[0]
    cut = int(n * 0.7)
    model.fit(x[:cut], y[:cut])
    q = model.predict(x[cut:])
    yy = y[cut:]
    mid = taus.index(min(taus, key=lambda t: abs(t - 0.5))) if taus else 0
    pin = mean_pinball(yy, q[:, mid], taus[mid])
    cross = quantile_crossing_rate(q, np.array(taus))
    metrics = {"mean_pinball": pin, "crossing_rate": cross}
    configure_tracking()
    run_id = log_run(
        family="distribution",
        name=model_name,
        params={"model": model_name},
        metrics=metrics,
        tags={"data": config.data.source},
    )
    path = Path(config.data.root) / "metadata" / f"dist_{model_name}.joblib"
    model.save(path)
    return {"metrics": metrics, "run_id": run_id, "path": str(path)}


def train_volatility(config: AppConfig, model_name: str = "ewma") -> dict[str, Any]:
    set_global_seed(config.train.random_seed)
    df = panel(config)
    label = config.train.volatility_target
    if label not in df.columns:
        cands = [c for c in df.columns if c.startswith("future_realized_var")]
        label = cands[0] if cands else "vol_20"
    x, y, dates, feats, _ = design_matrix(
        df, label, ["vol_20", "vol_ewma", "vol_parkinson", "ret_1", "vol_of_vol"]
    )
    model: Any = {
        "rolling": RollingVol(),
        "ewma": EWMAVol(config.features.ewma_lambda),
        "garch": GARCHVol(),
        "har": HARVol(config.train.har_log),
        "xgboost": TreeVol("xgboost", config.train.random_seed),
        "lightgbm": TreeVol("lightgbm", config.train.random_seed),
    }[model_name]
    cut = int(x.shape[0] * 0.7)
    model.fit(x[:cut], y[:cut])
    pred = np.clip(model.predict(x[cut:]), config.train.qlike_floor, None)
    metrics = {"qlike": qlike(y[cut:], pred, config.train.qlike_floor)}
    configure_tracking()
    run_id = log_run(family="volatility", name=model_name, params={}, metrics=metrics, tags={})
    path = Path(config.data.root) / "metadata" / f"vol_{model_name}.joblib"
    model.save(path)
    return {"metrics": metrics, "run_id": run_id, "path": str(path)}


def train_alpha(config: AppConfig, model_name: str = "ridge") -> dict[str, Any]:
    if model_name == "mean":
        # reuse ranking path with historical mean
        df = panel(config)
        label = config.train.ranking_target
        x, y, _, _, _ = design_matrix(df, label)
        m = HistoricalMeanAlpha().fit(x, y)
        path = Path(config.data.root) / "metadata" / "alpha_mean.joblib"
        m.save(path)
        return {"metrics": {"mean": float(m.mean_)}, "path": str(path)}
    return train_ranking(config, "ridge" if model_name == "ridge" else model_name)


def train_regime(config: AppConfig, model_name: str = "hmm") -> dict[str, Any]:
    df = panel(config)
    cols = [c for c in ["mkt_ret_1", "mkt_vol_20", "cs_dispersion", "breadth"] if c in df.columns]
    sub = df.select(["event_time", *cols]).unique("event_time").drop_nulls().sort("event_time")
    x = sub.select(cols).to_numpy().astype(float)
    m: Any
    if model_name == "hmm":
        m = GaussianHMMRegime(config.train.n_hmm_states, config.train.random_seed).fit(x)
        extra = m.aic_bic(x)
    elif model_name == "threshold":
        m = VolThresholdRegime().fit(x)
        extra = {}
    else:
        m = SingleStateRegime().fit(x)
        extra = {}
    path = Path(config.data.root) / "metadata" / f"regime_{model_name}.joblib"
    m.save(path)
    return {"metrics": extra, "path": str(path), "labels": getattr(m, "labels", {})}


def train_tail(config: AppConfig, model_name: str = "historical") -> dict[str, Any]:
    df = panel(config)
    label = [c for c in df.columns if c.startswith("future_return")][0]
    x, y, _, _, _ = design_matrix(df, label)
    m: Any
    if model_name == "gaussian":
        m = GaussianTail().fit(x, y)
    elif model_name == "drawdown":
        lab = [c for c in df.columns if c.startswith("future_tail_event")]
        x, y, _, _, _ = design_matrix(df, lab[0] if lab else label)
        m = DrawdownClassifier().fit(x, y)
    else:
        m = HistoricalTail().fit(x, y)
    path = Path(config.data.root) / "metadata" / f"tail_{model_name}.joblib"
    m.save(path)
    metrics: dict[str, float] = {}
    if hasattr(m, "predict_var_es"):
        var, es = m.predict_var_es()
        metrics = {"var": var, "es": es}
    return {"metrics": metrics, "path": str(path)}


def train_family(config: AppConfig, family: str, model_name: str | None = None) -> dict[str, Any]:
    dispatch = {
        "ranking": lambda: train_ranking(config, model_name or "ridge"),
        "alpha": lambda: train_alpha(config, model_name or "ridge"),
        "distribution": lambda: train_distribution(config, model_name or "gaussian"),
        "volatility": lambda: train_volatility(config, model_name or "ewma"),
        "regime": lambda: train_regime(config, model_name or "hmm"),
        "tail": lambda: train_tail(config, model_name or "historical"),
        "covariance": lambda: {"metrics": {}, "note": "covariance is estimated at forecast time"},
        "liquidity": lambda: {"metrics": {}, "note": "liquidity uses parameterized cost model"},
    }
    if family not in dispatch:
        raise ValueError(f"unknown family {family}")
    result = dispatch[family]()
    if config.data.source == "synthetic":
        result["data_source"] = "SYNTHETIC"
    return result
