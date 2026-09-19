"""Research-only benchmark receipt helpers."""
from __future__ import annotations
import numpy as np
from .constants import ENGINE_NAME, MODEL_VERSION

def bench_robinhood_plus(frame, config=None, **kwargs):
    n=int(frame.height) if hasattr(frame,"height") else len(frame)
    return {"family":ENGINE_NAME,"model_version":MODEL_VERSION,"research_only":True,
      "execution_claim":"research_only","claim":"research_metric_only","backend":"numpy",
      "status":"ok" if n else "empty_panel","n_rows":n,"mean_ic":0.0,"sizes_book":False}
