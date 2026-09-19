"""Causal cross-sectional forecast entry points."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
import joblib
import numpy as np
import polars as pl
from .constants import *
from .predictor import RobinhoodPlusPredictor, RobinhoodPlusForecast

def kline_columns_present(columns) -> bool:
    return all(c in columns for c in PRICE_COLS) and VOLUME_COL in columns

def extract_kline(frame: pl.DataFrame) -> np.ndarray:
    if not kline_columns_present(frame.columns): return np.empty((0,6), dtype=float)
    cols=list(PRICE_COLS)+[VOLUME_COL]
    try: arr=frame.select(cols).to_numpy().astype(float, copy=False)
    except (TypeError, ValueError): return np.empty((0,6), dtype=float)
    amount = arr[:,4] * arr[:,3]
    return np.column_stack([arr, amount])

@dataclass
class RobinhoodPlusNameForecast:
    forecast: RobinhoodPlusForecast
    security_id: str
    symbol: str

class RobinhoodPlusEngine:
    """Serializable wrapper around the deterministic predictor."""
    def __init__(self, *, lookback=64, pred_len=5, sample_count=8, s1_bits=5, s2_bits=5,
                 clip=5.0, temperature=1.0, top_p=0.9, max_context=512, seed=42, decoder="markov"):
        self.params = dict(lookback=lookback,pred_len=pred_len,sample_count=sample_count,s1_bits=s1_bits,
                           s2_bits=s2_bits,clip=clip,temperature=temperature,top_p=top_p,max_context=max_context,
                           seed=seed,decoder=decoder)
        self.predictor = RobinhoodPlusPredictor(**self.params)
    def fit(self, x: np.ndarray, y: np.ndarray, **kwargs: Any) -> "RobinhoodPlusEngine":
        return self
    def predict_kline(self, x: np.ndarray) -> RobinhoodPlusForecast:
        return self.predictor.predict_kline(x)
    def save(self, path: Path) -> None:
        path=Path(path); path.parent.mkdir(parents=True, exist_ok=True); joblib.dump(self,path)
    @classmethod
    def load(cls, path: Path) -> "RobinhoodPlusEngine": return joblib.load(path)

def _asset(hit: RobinhoodPlusForecast, sid: str, asof: datetime) -> RobinhoodPlusNameForecast:
    return RobinhoodPlusNameForecast(hit, sid, sid)

def forecast_robinhood_plus_cross_section(frame: pl.DataFrame, asof: datetime, *, lookback=64,
    pred_len=5, sample_count=8, s1_bits=5, s2_bits=5, clip=5.0, temperature=1.0,
    top_p=0.9, max_context=512, seed=42, decoder="markov", security_ids=None,
    quantile_levels=(0.05,0.5,0.95), horizons=(1,5,20)) -> dict[str, RobinhoodPlusNameForecast]:
    if not kline_columns_present(frame.columns) or "event_time" not in frame.columns or "security_id" not in frame.columns: return {}
    f=frame
    if "available_time" in f.columns: f=f.filter(pl.col("available_time") <= pl.lit(asof))
    f=f.filter(pl.col("event_time") <= pl.lit(asof)).sort(["security_id","event_time"])
    requested=[str(x) for x in security_ids] if security_ids is not None else [str(x) for x in f["security_id"].unique().to_list()]
    out={}
    for sid in requested:
        sub=f.filter(pl.col("security_id")==sid)
        x=extract_kline(sub)
        if len(x)==0: continue
        pred=RobinhoodPlusPredictor(lookback=lookback,pred_len=pred_len,sample_count=sample_count,s1_bits=s1_bits,s2_bits=s2_bits,clip=clip,temperature=temperature,top_p=top_p,max_context=max_context,seed=seed,decoder=decoder,horizons=horizons)
        hit=pred.predict_kline(x)
        out[sid]=_asset(hit,sid,asof)
    return out
