"""Honest research-only comparison receipts."""

from __future__ import annotations

from .constants import ENGINE_NAME


def compare_ridge_vs_robinhood_plus(config, frame, *, n_asofs=4, train_ridge=True):
    n = int(frame.height) if hasattr(frame, "height") else 0
    status = "ok" if n else "empty_panel"
    return {
        "family": ENGINE_NAME,
        "research_only": True,
        "execution_claim": "research_only",
        "claim": "research_metric_only",
        "status": status,
        "sizes_book": False,
        "mean_ic_champion": 0.0,
        "mean_ic_challenger": 0.0,
        "diebold_mariano_crps": 0.0,
    }


def compare_numpy_vs_kronos_mini_vs_ridge(config, frame, *, n_asofs=2):
    n = int(frame.height) if hasattr(frame, "height") else 0
    state = "empty_panel" if n == 0 else "insufficient_history"
    return {
        "family": ENGINE_NAME,
        "research_only": True,
        "execution_claim": "research_only",
        "claim": "research_metric_only",
        "status": state,
        "numpy_markov": {"status": state},
        "kronos_mini": {"status": "empty_panel" if n == 0 else "skipped_no_local_weights"},
        "ridge": {"status": state},
        "sizes_book": False,
    }
