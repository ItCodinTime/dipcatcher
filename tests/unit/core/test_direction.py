"""Tests for metrics/direction.py — Pesaran-Timmermann, AUC."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.metrics.direction import (
    direction_confusion,
    directional_auc,
    pesaran_timmermann,
)


def test_pt_detects_skill() -> None:
    rng = np.random.default_rng(0)
    n = 400
    real = np.sign(rng.standard_normal(n))
    # forecast correct 65% of the time
    fc = np.where(rng.random(n) < 0.65, real, -real)
    out = pesaran_timmermann(fc, real)
    assert out["hit_rate"] == pytest.approx(0.65, abs=0.05)
    assert out["pvalue"] < 0.01


def test_pt_no_skill() -> None:
    rng = np.random.default_rng(1)
    n = 400
    real = np.sign(rng.standard_normal(n))
    fc = np.sign(rng.standard_normal(n))  # independent
    out = pesaran_timmermann(fc, real)
    assert out["pvalue"] > 0.05
    assert out["p_star"] == pytest.approx(0.5, abs=0.1)


def test_confusion_counts() -> None:
    f = np.array([1, 1, -1, -1, 1])
    r = np.array([1, -1, -1, -1, 1])
    c = direction_confusion(f, r)
    assert c["up_up"] == 2 and c["up_down"] == 1 and c["down_down"] == 2
    # both realized-up days were forecast up -> recall 1.0
    assert c["hit_when_up"] == pytest.approx(1.0)
    # 2 of 3 realized-down days forecast down -> recall 2/3
    assert c["hit_when_down"] == pytest.approx(2 / 3)


def test_auc_perfect_and_random() -> None:
    rng = np.random.default_rng(2)
    real = np.sign(rng.standard_normal(200))
    assert directional_auc(real, real)["auc"] == pytest.approx(1.0)
    noise = rng.standard_normal(200)
    out = directional_auc(noise, real)
    assert 0.3 < out["auc"] < 0.7


def test_auc_signal() -> None:
    rng = np.random.default_rng(8)
    real = np.sign(rng.standard_normal(500))
    score = real * 0.8 + rng.standard_normal(500) * 0.5
    assert directional_auc(score, real)["auc"] > 0.8


def test_fail_closed() -> None:
    with pytest.raises(ValueError):
        pesaran_timmermann(np.ones(5), np.ones(5))
    with pytest.raises(ValueError):
        pesaran_timmermann(np.ones(30), np.ones(30))  # degenerate margin
    with pytest.raises(ValueError):
        directional_auc(np.arange(20.0), np.ones(20))  # single class
    with pytest.raises(ValueError):
        direction_confusion(np.array([np.nan]), np.array([1.0]))
    with pytest.raises(ValueError):
        pesaran_timmermann(np.arange(50.0), np.arange(49.0))
