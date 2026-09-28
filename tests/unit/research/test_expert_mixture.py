"""Online expert-mixture over the SYNTHETIC fleet — causality + convexity pins."""

from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest
from typer.testing import CliRunner

from quant_fund.cli.main import app
from quant_fund.metrics.scoring import pinball_loss
from quant_fund.research.catalog import family_blob_forbidden_metrics_absent
from quant_fund.research.expert_mixture import (
    _loss_tensor,
    ewa_weights,
    fixed_share_weights,
    run_expert_mixture_eval,
    uniform_weights,
    write_expert_mixture_receipt,
)
from quant_fund.research.fleet_eval import fleet_head_factories

HEADS = ["empirical", "gaussian", "skew_t", "gmm"]
TAUS = [0.05, 0.25, 0.5, 0.75, 0.95]


def _run(seed: int = 0, shards=("iid_gaussian", "regime_switch"), n_eval: int = 96):
    return run_expert_mixture_eval(
        seed=seed,
        n_train=192,
        n_eval=n_eval,
        alpha=0.05,
        head_names=HEADS,
        shard_names=list(shards),
        taus=TAUS,
    )


def test_emits_expert_and_mixer_rows() -> None:
    frame, receipt = _run()
    assert receipt["kind"] == "expert_mixture_eval"
    assert receipt["data_label"] == "SYNTHETIC"
    assert receipt["schema"] == "receipt.v2"
    assert receipt["verdict"] == "pass"
    assert family_blob_forbidden_metrics_absent(receipt["payload"])
    ok = frame.filter(pl.col("status") == "ok")
    for shard in ("iid_gaussian", "regime_switch"):
        members = ok.filter(pl.col("shard") == shard)
        roles = members["role"].to_list()
        assert roles.count("expert") == len(HEADS)
        assert roles.count("mixer") == 3


def test_weights_are_simplex_and_first_row_uniform() -> None:
    rng = np.random.default_rng(0)
    losses = rng.gamma(2.0, 0.001, size=(6, 64))
    for w in (uniform_weights(losses), ewa_weights(losses), fixed_share_weights(losses)):
        assert w.shape == losses.shape
        assert np.all(w >= 0.0)
        np.testing.assert_allclose(w.sum(axis=0), 1.0, atol=1e-12)
        np.testing.assert_allclose(w[:, 0], 1.0 / 6.0)


def test_ewa_concentrates_on_the_better_expert() -> None:
    rng = np.random.default_rng(1)
    losses = rng.normal(0.01, 0.002, size=(4, 200))
    losses[2] -= 0.004  # expert 2 strictly better
    w = ewa_weights(losses)
    assert w[2, -1] > 0.9
    fs = fixed_share_weights(losses)
    assert fs[2, -1] > 0.8


def test_fixed_share_resurrects_a_late_expert() -> None:
    """Expert 0 is terrible early and great late; fixed-share must retake it
    while plain EWA stays locked out — the tracking property."""
    k, t = 3, 400
    losses = np.full((k, t), 0.010)
    losses[0, : t // 2] = 0.040  # bad first half
    losses[0, t // 2 :] = 0.002  # brilliant second half
    losses[1] = 0.010
    losses[2] = 0.011
    w_ewa = ewa_weights(losses)
    w_fs = fixed_share_weights(losses, alpha=0.1)
    assert w_ewa[0, -1] < 0.05  # EWA never forgives the early catastrophe
    assert w_fs[0, -1] > 0.5  # fixed-share tracks the switch


def test_causality_future_perturbation() -> None:
    """Perturbing losses in the tail leaves every earlier weight identical."""
    rng = np.random.default_rng(2)
    losses = np.abs(rng.normal(0.01, 0.003, size=(4, 120)))
    perturbed = losses.copy()
    perturbed[:, 60:] = rng.normal(5.0, 1.0, size=(4, 60))
    for fn in (ewa_weights, fixed_share_weights):
        np.testing.assert_allclose(fn(losses)[:, :60], fn(perturbed)[:, :60])


def test_mixture_convexity_bound_per_row() -> None:
    """pinball(y, Σw·q, τ) ≤ Σ w·pinball(y, q_h, τ) — the row-level theorem."""
    rng = np.random.default_rng(3)
    k, t, j = 5, 80, len(TAUS)
    grids = np.sort(rng.normal(0.0, 0.02, size=(k, t, j)), axis=2)
    y = rng.normal(0.0, 0.02, size=t)
    w = ewa_weights(_loss_tensor(grids, y, np.asarray(TAUS)))
    q_mix = np.einsum("kt,ktj->tj", w, grids)
    losses_h = _loss_tensor(grids, y, np.asarray(TAUS))
    for i in range(t):
        mix_row_loss = float(
            np.mean([pinball_loss(y[i], q_mix[i, jj], float(TAUS[jj])) for jj in range(j)])
        )
        weighted_bound = float(np.sum(w[:, i] * losses_h[:, i]))
        assert mix_row_loss <= weighted_bound + 1e-12


def test_uniform_mixture_never_worse_than_worst_expert() -> None:
    """CRPS(uniform) ≤ (1/K)Σ CRPS_h ≤ max_h CRPS_h — convexity aggregated."""
    frame, _ = _run()
    ok = frame.filter(pl.col("status") == "ok")
    for shard in ok["shard"].unique().to_list():
        sub = ok.filter(pl.col("shard") == shard)
        crps = {r["member"]: r["crps"] for r in sub.iter_rows(named=True)}
        worst = max(v for m, v in crps.items() if m in HEADS)
        assert crps["uniform"] <= worst + 1e-12


def test_failed_head_excluded_not_silent() -> None:
    factories = fleet_head_factories(TAUS, 0, names=HEADS)

    class _DeadHead:
        def fit(self, x, y, **kw):
            raise RuntimeError("deliberate failure")

    factories = dict(factories)
    factories["dead"] = _DeadHead
    from quant_fund.research.expert_mixture import run_expert_mixture

    frame, payload = run_expert_mixture(
        factories, ["iid_gaussian"], n_train=192, n_eval=64, seed=0, taus=TAUS
    )
    dead = frame.filter(pl.col("member") == "dead")
    assert dead["status"].to_list() == ["error"]
    mixers = frame.filter((pl.col("role") == "mixer") & (pl.col("status") == "ok"))
    assert mixers.height == 3


def test_determinism() -> None:
    f1, _ = _run(seed=5)
    f2, _ = _run(seed=5)
    assert f1.equals(f2)


def test_receipt_seal_and_write(tmp_path) -> None:
    _, receipt = _run(n_eval=64)
    path = write_expert_mixture_receipt(receipt, tmp_path)
    sealed = json.loads(path.read_text())
    assert sealed["receipt_sha256"].startswith(path.stem.removeprefix("expert_mixture_"))


def test_receipt_rejects_non_synthetic(tmp_path) -> None:
    _, receipt = _run(n_eval=64)
    bad = dict(receipt)
    bad["data_label"] = "LIVE"
    with pytest.raises(ValueError, match="honesty contract"):
        write_expert_mixture_receipt(bad, tmp_path)


def test_validation() -> None:
    with pytest.raises(ValueError, match="alpha"):
        run_expert_mixture_eval(
            seed=0, n_train=64, n_eval=32, alpha=1.0, head_names=HEADS, shard_names=["iid_gaussian"]
        )
    with pytest.raises(ValueError, match="n_eval"):
        run_expert_mixture_eval(
            seed=0, n_train=64, n_eval=1, alpha=0.05, head_names=HEADS, shard_names=["iid_gaussian"]
        )


def test_cli_requires_dev() -> None:
    result = CliRunner().invoke(app, ["expert-mixture"])
    assert result.exit_code != 0
