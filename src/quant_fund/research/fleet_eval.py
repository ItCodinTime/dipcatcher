"""Fleet evaluation of distribution challengers on seeded SYNTHETIC shards.

Each shard generator draws a labeled synthetic return series with a planted
distributional property (multimodality, heavy tails, skew, regime-switching or
GARCH vol clustering). ``run_distribution_fleet`` fits every head on the
leading slice of every shard, predicts the trailing slice, and scores with
proper scores only (pinball, CRPS, PIT-KS, interval coverage) — never headline
P&L metrics. ``write_fleet_receipt`` persists the sealed JSON evidence under
``receipts/``. All output is SYNTHETIC correctness evidence, not market data.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

from quant_fund.metrics.probability import pit_ks
from quant_fund.metrics.scoring import (
    coverage,
    crps_from_quantiles,
    mean_pinball,
    pit_values,
)
from quant_fund.models.distribution import (
    EmpiricalDistribution,
    GaussianDistribution,
    GMMDistribution,
    IsotonicPitDistribution,
    SkewTDistribution,
    StackedDistribution,
)
from quant_fund.models.skew_t import skew_t_ppf
from quant_fund.research.catalog import family_blob_forbidden_metrics_absent
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

Array = NDArray[np.float64]

FLEET_EVAL_SCHEMA = "fleet_eval.v1"
DEFAULT_TAUS: tuple[float, ...] = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)
COVERAGE_LEVELS: tuple[float, ...] = (0.8, 0.9)


@dataclass(frozen=True)
class SyntheticShard:
    """One seeded SYNTHETIC return shard: inert design matrix + target series."""

    name: str
    x: Array
    y: Array
    config: dict[str, Any]


ShardGenerator = Callable[[int, int], SyntheticShard]
HeadFactory = Callable[[], Any]


def _require_n(n: int) -> int:
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("shard size n must be a positive integer")
    return int(n)


def _dummy_x(n: int) -> Array:
    """Inert single-column design matrix — unconditional heads ignore x."""
    return np.ones((_require_n(n), 1), dtype=np.float64)


def iid_gaussian(n: int, seed: int) -> SyntheticShard:
    """Homoskedastic N(mu, sigma) baseline shard (SYNTHETIC)."""
    rng = np.random.default_rng(seed)
    config: dict[str, Any] = {
        "data_label": "SYNTHETIC",
        "seed": int(seed),
        "mu": 0.0,
        "sigma": 0.02,
    }
    y = rng.normal(config["mu"], config["sigma"], _require_n(n))
    return SyntheticShard("iid_gaussian", _dummy_x(n), np.asarray(y, dtype=float), config)


def bimodal_mixture(n: int, seed: int) -> SyntheticShard:
    """Two-component Gaussian mixture — planted multimodality (SYNTHETIC)."""
    rng = np.random.default_rng(seed)
    config: dict[str, Any] = {
        "data_label": "SYNTHETIC",
        "seed": int(seed),
        "weight_lo": 0.65,
        "mu": (-0.025, 0.03),
        "sigma": (0.012, 0.016),
    }
    n = _require_n(n)
    hi = rng.random(n) >= float(config["weight_lo"])
    mus = np.asarray(config["mu"], dtype=float)
    sigs = np.asarray(config["sigma"], dtype=float)
    y = rng.normal(np.where(hi, mus[1], mus[0]), np.where(hi, sigs[1], sigs[0]))
    return SyntheticShard("bimodal_mixture", _dummy_x(n), np.asarray(y, dtype=float), config)


def heavy_tail(n: int, seed: int) -> SyntheticShard:
    """Student-t df=3 shard — planted heavy tails, infinite kurtosis (SYNTHETIC)."""
    rng = np.random.default_rng(seed)
    config: dict[str, Any] = {
        "data_label": "SYNTHETIC",
        "seed": int(seed),
        "df": 3.0,
        "scale": 0.01,
    }
    y = rng.standard_t(float(config["df"]), _require_n(n)) * float(config["scale"])
    return SyntheticShard("heavy_tail", _dummy_x(n), np.asarray(y, dtype=float), config)


def left_skew(n: int, seed: int) -> SyntheticShard:
    """Hansen (1994) skew-t shard with lam < 0 — planted left skew (SYNTHETIC)."""
    rng = np.random.default_rng(seed)
    config: dict[str, Any] = {
        "data_label": "SYNTHETIC",
        "seed": int(seed),
        "nu": 6.0,
        "lam": -0.6,
        "mu": 0.0,
        "sigma": 0.02,
    }
    u = np.clip(rng.uniform(0.0, 1.0, _require_n(n)), 1e-9, 1.0 - 1e-9)
    nu = float(config["nu"])
    lam = float(config["lam"])
    mu = float(config["mu"])
    sigma = float(config["sigma"])
    y = np.asarray([skew_t_ppf(float(uu), nu, lam, mu, sigma) for uu in u], dtype=np.float64)
    return SyntheticShard("left_skew", _dummy_x(n), y, config)


def regime_switch(n: int, seed: int) -> SyntheticShard:
    """Two-state Markov-switching vol AR(1) mixture (Hamilton 1989; SYNTHETIC)."""
    rng = np.random.default_rng(seed)
    config: dict[str, Any] = {
        "data_label": "SYNTHETIC",
        "seed": int(seed),
        "rho": 0.1,
        "sigma": (0.008, 0.035),
        "p_leave": (0.04, 0.08),
    }
    n = _require_n(n)
    sig_lo, sig_hi = config["sigma"]
    p01, p10 = config["p_leave"]
    state = np.empty(n, dtype=np.int64)
    state[0] = 0
    u = rng.random(n)
    for t in range(1, n):
        stay = (1.0 - p01) if state[t - 1] == 0 else (1.0 - p10)
        state[t] = state[t - 1] if u[t] < stay else 1 - state[t - 1]
    eps = rng.normal(0.0, np.where(state == 0, sig_lo, sig_hi))
    rho = float(config["rho"])
    y = np.empty(n, dtype=float)
    y[0] = eps[0]
    for t in range(1, n):
        y[t] = rho * y[t - 1] + eps[t]
    config["n_state1"] = int(state.sum())
    return SyntheticShard("regime_switch", _dummy_x(n), y, config)


def garch_cluster(n: int, seed: int) -> SyntheticShard:
    """GJR-GARCH(1,1) path (Glosten-Jagannathan-Runkle 1993; SYNTHETIC)."""
    rng = np.random.default_rng(seed)
    config: dict[str, Any] = {
        "data_label": "SYNTHETIC",
        "seed": int(seed),
        "omega": 4e-6,
        "alpha": 0.05,
        "gamma": 0.08,
        "beta": 0.9,
        "burn": 128,
    }
    n = _require_n(n)
    omega = float(config["omega"])
    alpha = float(config["alpha"])
    gamma = float(config["gamma"])
    beta = float(config["beta"])
    persistence = alpha + gamma / 2.0 + beta
    if not 0.0 < persistence < 1.0:
        raise ValueError("GJR-GARCH persistence must lie in (0, 1)")
    burn = int(config["burn"])
    total = n + burn
    z = rng.normal(0.0, 1.0, total)
    eps = np.empty(total, dtype=float)
    var = np.empty(total, dtype=float)
    var[0] = omega / (1.0 - persistence)
    eps[0] = np.sqrt(var[0]) * z[0]
    for t in range(1, total):
        var[t] = omega + (alpha + gamma * (eps[t - 1] < 0.0)) * eps[t - 1] ** 2 + beta * var[t - 1]
        eps[t] = np.sqrt(var[t]) * z[t]
    config["persistence"] = float(persistence)
    return SyntheticShard("garch_cluster", _dummy_x(n), np.asarray(eps[burn:], dtype=float), config)


SHARD_GENERATORS: dict[str, ShardGenerator] = {
    "iid_gaussian": iid_gaussian,
    "bimodal_mixture": bimodal_mixture,
    "heavy_tail": heavy_tail,
    "left_skew": left_skew,
    "regime_switch": regime_switch,
    "garch_cluster": garch_cluster,
}

# Head-name registry for the `fleet` CLI --models flag. Constructors take the
# scoring tau grid and a seed; only unconditional heads from
# models/distribution.py plus the torch-optional neural heads (imported lazily
# inside the factory so this module never requires the ``nn`` extra) are
# registered (no cross-PR head dependencies).
FLEET_HEAD_REGISTRY: dict[str, Callable[[Sequence[float], int], Any]] = {
    "empirical": lambda taus, seed: EmpiricalDistribution(list(taus)),
    "gaussian": lambda taus, seed: GaussianDistribution(list(taus)),
    "skew_t": lambda taus, seed: SkewTDistribution(list(taus)),
    "gmm": lambda taus, seed: GMMDistribution(list(taus), seed=int(seed)),
    "isotonic": lambda taus, seed: IsotonicPitDistribution(list(taus)),
    "stack": lambda taus, seed: StackedDistribution(list(taus), seed=int(seed)),
    "nbeats": lambda taus, seed: _nbeats(taus, seed),
    "nhits": lambda taus, seed: _nhits(taus, seed),
}


def _nbeats(taus: Sequence[float], seed: int) -> Any:
    from quant_fund.models.nbeats import NBeatsDistribution

    return NBeatsDistribution(list(taus), seed=int(seed))


def _nhits(taus: Sequence[float], seed: int) -> Any:
    from quant_fund.models.nbeats import NHiTsDistribution

    return NHiTsDistribution(list(taus), seed=int(seed))


def resolve_shard_generators(names: Iterable[str] | None = None) -> dict[str, ShardGenerator]:
    """Resolve shard names against ``SHARD_GENERATORS`` (default: all)."""
    if names is None:
        return dict(SHARD_GENERATORS)
    resolved: dict[str, ShardGenerator] = {}
    for raw in names:
        name = str(raw).strip()
        if not name:
            continue
        if name not in SHARD_GENERATORS:
            raise ValueError(f"unknown fleet shard {name!r}")
        resolved[name] = SHARD_GENERATORS[name]
    if not resolved:
        raise ValueError("fleet requires at least one shard")
    return resolved


def _head_factory(name: str, taus: Sequence[float], seed: int) -> HeadFactory:
    def _factory() -> Any:
        return FLEET_HEAD_REGISTRY[name](taus, seed)

    return _factory


def fleet_head_factories(
    taus: Sequence[float],
    seed: int,
    names: Iterable[str] | None = None,
) -> dict[str, HeadFactory]:
    """Build ``{name: factory}`` for the fleet from ``FLEET_HEAD_REGISTRY``."""
    chosen = list(FLEET_HEAD_REGISTRY) if names is None else [str(s).strip() for s in names]
    chosen = [name for name in chosen if name]
    if not chosen:
        raise ValueError("fleet requires at least one head")
    unknown = sorted(set(chosen) - set(FLEET_HEAD_REGISTRY))
    if unknown:
        raise ValueError(f"unknown fleet head(s): {', '.join(unknown)}")
    return {name: _head_factory(name, taus, seed) for name in chosen}


def _pinball_key(tau: float) -> str:
    return f"pinball_{tau:g}"


def _coverage_key(level: float) -> str:
    return f"coverage_{int(round(level * 100))}"


def _central_interval_index(taus: Array, level: float) -> tuple[int, int] | None:
    """Indices of the central ``level`` interval on the tau grid, else None."""
    lo, hi = (1.0 - level) / 2.0, (1.0 + level) / 2.0
    i = np.flatnonzero(np.isclose(taus, lo, atol=1e-9))
    j = np.flatnonzero(np.isclose(taus, hi, atol=1e-9))
    if i.size == 0 or j.size == 0:
        return None
    return int(i[0]), int(j[0])


def _score_row(
    shard: SyntheticShard,
    shard_seed: int,
    name: str,
    factory: HeadFactory,
    n_train: int,
    n_eval: int,
    taus: Array,
    coverage_index: dict[float, tuple[int, int] | None],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "shard": shard.name,
        "model": name,
        "family": "distribution",
        "status": "ok",
        "error": None,
        "n_train": n_train,
        "n_eval": n_eval,
        "seed": shard_seed,
        "crps": None,
        "pit_ks": None,
        "pit_ks_p": None,
        **{_coverage_key(level): None for level in COVERAGE_LEVELS},
        **{_pinball_key(float(t)): None for t in taus},
    }
    try:
        model = factory()
        model.fit(shard.x[:n_train], shard.y[:n_train])
        meta = model.metadata()
        row["family"] = str(meta.family)
        q = np.asarray(model.predict(shard.x[n_train : n_train + n_eval]), dtype=float)
        if q.ndim != 2 or q.shape[0] != n_eval or q.shape[1] != taus.size:
            raise ValueError(f"predict returned shape {q.shape}; expected ({n_eval}, {taus.size})")
        if not np.isfinite(q).all():
            raise ValueError("predict returned non-finite quantiles")
        if np.any(np.diff(q, axis=1) < 0.0):
            raise ValueError("predict returned crossing quantiles")
        y_eval = shard.y[n_train : n_train + n_eval]
        row["crps"] = crps_from_quantiles(y_eval, q, taus)
        ks, ks_p = pit_ks(pit_values(y_eval, q, taus))
        row["pit_ks"] = ks
        # The usual KS p-value assumes independent PIT draws. These two
        # generators deliberately include serial dependence.
        row["pit_ks_p"] = None if shard.name in {"regime_switch", "garch_cluster"} else ks_p
        for level, pair in coverage_index.items():
            if pair is not None:
                row[_coverage_key(level)] = coverage(y_eval, q[:, pair[0]], q[:, pair[1]])
        for j, tau in enumerate(taus):
            row[_pinball_key(float(tau))] = mean_pinball(y_eval, q[:, j], float(tau))
    except Exception as exc:
        row["status"] = "error"
        row["error"] = str(exc)
    return row


def run_distribution_fleet(
    factories: Mapping[str, HeadFactory],
    shards: Iterable[str] | Mapping[str, ShardGenerator] | None = None,
    n_train: int = 512,
    n_eval: int = 256,
    *,
    seed: int = 0,
    taus: Sequence[float] = DEFAULT_TAUS,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Fit every head on each shard's leading slice; score the trailing slice.

    Scores are proper rules only: per-tau pinball, quantile CRPS, PIT
    Kolmogorov-Smirnov, and central 80/90% coverage. A head that fails to fit
    or predict is recorded as an ``error`` row (visible, never silent); the
    harness itself fails closed on degenerate arguments or degenerate shard
    output. Returns the results frame plus the unsealed receipt payload.
    """
    if not isinstance(factories, Mapping) or not factories:
        raise ValueError("fleet requires a nonempty mapping of head factories")
    if (
        isinstance(n_train, bool)
        or not isinstance(n_train, (int, np.integer))
        or n_train < 1
        or isinstance(n_eval, bool)
        or not isinstance(n_eval, (int, np.integer))
        or n_eval < 1
    ):
        raise ValueError("n_train and n_eval must be positive")
    n_train = int(n_train)
    n_eval = int(n_eval)
    tau_arr = np.asarray(list(taus), dtype=float)
    if (
        tau_arr.size == 0
        or not np.isfinite(tau_arr).all()
        or np.any((tau_arr <= 0.0) | (tau_arr >= 1.0))
        or np.any(np.diff(tau_arr) <= 0.0)
    ):
        raise ValueError("taus must be a nonempty strictly increasing grid inside (0, 1)")
    resolved: Mapping[str, ShardGenerator]
    if shards is None:
        resolved = SHARD_GENERATORS
    elif isinstance(shards, Mapping):
        resolved = shards
    else:
        resolved = resolve_shard_generators(shards)
    if not resolved:
        raise ValueError("fleet requires at least one shard")
    for name in resolved:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("shard names must be nonempty strings")

    coverage_index = {level: _central_interval_index(tau_arr, level) for level in COVERAGE_LEVELS}
    n_shard = n_train + n_eval
    rows: list[dict[str, Any]] = []
    shard_meta: dict[str, Any] = {}
    for shard_index, (shard_name, generator) in enumerate(resolved.items()):
        shard_seed = int(seed) + shard_index
        shard = generator(n_shard, shard_seed)
        if not isinstance(shard, SyntheticShard):
            raise ValueError(f"shard {shard_name!r} did not return a SyntheticShard")
        y = np.asarray(shard.y, dtype=float).reshape(-1)
        x = np.asarray(shard.x, dtype=float)
        if shard.name != shard_name or shard.config.get("data_label") != "SYNTHETIC":
            raise ValueError(f"shard {shard_name!r} must match its name and SYNTHETIC label")
        if y.size != n_shard or x.ndim != 2 or x.shape[0] != n_shard:
            raise ValueError(
                f"shard {shard_name!r} produced {y.size} rows; needs exactly {n_shard}"
            )
        if not np.isfinite(y).all() or not np.isfinite(x).all():
            raise ValueError(f"shard {shard_name!r} produced non-finite data")
        shard = SyntheticShard(shard.name, x, y, dict(shard.config))
        shard_meta[shard_name] = {
            "n": int(y.size),
            "seed": shard_seed,
            "x_sha256": hash_bytes(x.tobytes()),
            "y_sha256": hash_bytes(y.tobytes()),
            "config": shard.config,
        }
        for model_name, factory in factories.items():
            rows.append(
                _score_row(
                    shard, shard_seed, model_name, factory, n_train, n_eval, tau_arr, coverage_index
                )
            )

    columns = [
        "shard",
        "model",
        "family",
        "status",
        "error",
        "n_train",
        "n_eval",
        "seed",
        "crps",
        "pit_ks",
        "pit_ks_p",
        *[_coverage_key(level) for level in COVERAGE_LEVELS],
        *[_pinball_key(float(t)) for t in tau_arr],
    ]
    schema = {
        **{key: pl.String for key in ("shard", "model", "family", "status", "error")},
        **{key: pl.Int64 for key in ("n_train", "n_eval", "seed")},
        **{
            key: pl.Float64
            for key in columns
            if key
            not in {"shard", "model", "family", "status", "error", "n_train", "n_eval", "seed"}
        },
    }
    frame = pl.DataFrame(rows, schema=schema, orient="row").select(columns)

    inputs_sha256 = hash_bytes(
        canonical_json_bytes(
            {
                "shards": {
                    name: {"x_sha256": meta["x_sha256"], "y_sha256": meta["y_sha256"]}
                    for name, meta in shard_meta.items()
                },
                "models": sorted(str(k) for k in factories),
                "taus": [float(t) for t in tau_arr],
                "n_train": n_train,
                "n_eval": n_eval,
                "seed": int(seed),
            }
        )
    )
    receipt: dict[str, Any] = {
        "schema": FLEET_EVAL_SCHEMA,
        "kind": "distribution_fleet_eval",
        "data_label": "SYNTHETIC",
        "live_pnl_claim": False,
        "generated_at": datetime.now(UTC).isoformat(),
        "git_revision": git_revision(),
        "seed": int(seed),
        "n_train": n_train,
        "n_eval": n_eval,
        "taus": [float(t) for t in tau_arr],
        "models": sorted(str(k) for k in factories),
        "shards": shard_meta,
        "inputs_sha256": inputs_sha256,
        "n_rows": len(rows),
        "n_error_rows": sum(1 for row in rows if row["status"] != "ok"),
        "results": rows,
    }
    return frame, receipt


def _atomic_write_text(path: Path, content: str) -> None:
    """Publish a complete immutable text artifact without replacing an existing one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise FileExistsError(f"receipt path is a symlink: {path}")
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"receipt already exists with different content: {path}")
        return
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            mode="w",
            encoding="utf-8",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError:
            if path.is_symlink() or path.read_text(encoding="utf-8") != content:
                raise FileExistsError(
                    f"receipt already exists with different content: {path}"
                ) from None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def write_fleet_receipt(
    receipt: Mapping[str, Any],
    receipts_dir: Path | str = Path("receipts"),
) -> Path:
    """Seal a fleet receipt and write ``receipts/fleet_eval_<hash>.json``.

    The filename hash is the sha256 of the canonical receipt payload; the same
    digest is embedded as ``receipt_sha256`` (mirroring the real_benchmark
    seal convention). The write is atomic.
    """
    research_blob = {key: value for key, value in receipt.items() if key != "live_pnl_claim"}
    if (
        receipt.get("schema") != FLEET_EVAL_SCHEMA
        or receipt.get("data_label") != "SYNTHETIC"
        or receipt.get("live_pnl_claim") is not False
        or not isinstance(receipt.get("results"), list)
        or not receipt["results"]
        or not family_blob_forbidden_metrics_absent(research_blob)
    ):
        raise ValueError("fleet receipt violates its synthetic research contract")
    canonical = json.loads(canonical_json_bytes(dict(receipt)))
    digest = hash_bytes(canonical_json_bytes(canonical))
    payload = {**canonical, "receipt_sha256": digest}
    path = Path(receipts_dir) / f"fleet_eval_{digest[:16]}.json"
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path
