"""P5.2 / P5.6 — causal vol-target overlay + ADV participation-capacity bench.

``vol_target_scales`` (P5.2) computes delay-1 leverage factors: the scale
applied at date ``t`` is a function of realized returns strictly before ``t``
(trailing or EWMA estimate of the book vol), so a spiked past compresses
leverage and a quiet past expands it up to ``max_leverage``. No lookahead,
no alpha manufacture — it only caps ex-ante risk.

``run_capacity_bench`` (P5.6) feeds seeded SYNTHETIC target-weight books and
dollar-ADV surfaces through a participation cap: for each book × AUM level it
reports the share of dates on which the required rebalance fits inside
``cap × ADV``, the median worst-name days-to-trade, mean participation, and a
square-root-impact cost estimate in basis points. Output is a sealed
``capacity_overlay_eval`` receipt labeled SYNTHETIC and marked ``dev_only``
— a size/feasibility bound, never a market claim.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
from numpy.typing import NDArray

from quant_fund.research.catalog import family_blob_forbidden_metrics_absent
from quant_fund.research.fleet_eval import _atomic_write_text
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

Array = NDArray[np.float64]

CAPACITY_SCHEMA = "capacity_overlay.v1"

_EPS = 1e-12


def vol_target_scales(
    returns: Sequence[float],
    *,
    vol_target: float = 0.10,
    lookback: int = 21,
    ewma_lambda: float | None = None,
    max_leverage: float = 1.5,
    ann_factor: float = 252.0,
) -> Array:
    """Delay-1 leverage factors from past returns only.

    ``scales[t] = clip(vol_target / sigma_hat_t, 0, max_leverage)`` where
    ``sigma_hat_t`` is the annualized trailing std (or EWMA std when
    ``ewma_lambda`` is set) of ``returns[max(0, t-lookback):t]``. Dates before
    ``lookback`` observations accumulate get a neutral scale of 1.0. A
    non-finite estimate flattens the book (0.0); a near-zero estimate caps at
    ``max_leverage``. Fail-closed on non-finite or non-positive inputs to the
    constructor parameters.
    """
    if not np.isfinite(vol_target) or vol_target <= 0:
        raise ValueError("vol_target must be finite and positive")
    if not np.isfinite(max_leverage) or max_leverage <= 0:
        raise ValueError("max_leverage must be finite and positive")
    if lookback < 2:
        raise ValueError("lookback must be >= 2")
    if ewma_lambda is not None and not 0.0 < ewma_lambda < 1.0:
        raise ValueError("ewma_lambda must be in (0, 1)")
    if not np.isfinite(ann_factor) or ann_factor <= 0:
        raise ValueError("ann_factor must be finite and positive")
    r = np.asarray(returns, dtype=float)
    if r.ndim != 1 or r.size < 2:
        raise ValueError("returns must be a 1-D series with >= 2 observations")
    scales = np.empty(r.size, dtype=float)
    lam_weights: Array | None = None
    if ewma_lambda is not None:
        idx = np.arange(lookback, dtype=float)
        w = ewma_lambda ** (lookback - 1 - idx)
        lam_weights = w / w.sum()
    target_daily = vol_target / np.sqrt(ann_factor)
    for t in range(r.size):
        window = r[max(0, t - lookback) : t]
        if window.size < lookback:
            scales[t] = 1.0  # warmup: neutral leverage until the estimate exists
            continue
        if not np.all(np.isfinite(window)):
            scales[t] = 0.0  # unmeasurable vol -> flatten
            continue
        if lam_weights is not None:
            mu = float(np.dot(lam_weights, window))
            sigma = float(np.sqrt(np.dot(lam_weights, (window - mu) ** 2)))
        else:
            sigma = float(np.std(window, ddof=1))
        if not np.isfinite(sigma):
            scales[t] = 0.0
        elif sigma <= _EPS:
            scales[t] = max_leverage
        else:
            scales[t] = float(np.clip(target_daily / sigma, 0.0, max_leverage))
    return scales


@dataclass(frozen=True)
class SyntheticBook:
    """One seeded SYNTHETIC book: target weights (T×N) + dollar ADV (T×N)."""

    name: str
    weights: Array
    adv_dollar: Array

    def __post_init__(self) -> None:
        if self.weights.ndim != 2 or self.adv_dollar.shape != self.weights.shape:
            raise ValueError("weights and adv_dollar must share a 2-D shape")
        if self.weights.shape[1] < 2 or self.weights.shape[0] < 2:
            raise ValueError("book needs >= 2 dates and >= 2 names")


def _prices_and_adv(n_dates: int, n_names: int, seed: int) -> tuple[Array, Array]:
    rng = np.random.default_rng(seed)
    adv_shares = np.exp(rng.normal(np.log(1e6), 0.8, size=(n_dates, n_names)))
    prices = np.exp(rng.normal(np.log(50.0), 0.5, size=n_names))
    return prices, adv_shares * prices[None, :]


def uniform_book(n_dates: int, n_names: int, seed: int) -> SyntheticBook:
    """Equal-weight book on uniform dollar ADV."""

    _, adv = _prices_and_adv(n_dates, n_names, seed)
    w = np.full((n_dates, n_names), 1.0 / n_names)
    return SyntheticBook("uniform", w, adv)


def concentrated_book(n_dates: int, n_names: int, seed: int) -> SyntheticBook:
    """90% of the book in one name — binds capacity at modest AUM."""

    _, adv = _prices_and_adv(n_dates, n_names, seed)
    w = np.full((n_dates, n_names), 0.10 / (n_names - 1))
    w[:, 0] = 0.90
    return SyntheticBook("concentrated", w, adv)


def thin_adv_book(n_dates: int, n_names: int, seed: int) -> SyntheticBook:
    """One name with 1% of normal dollar ADV — the capacity bottleneck."""

    _, adv = _prices_and_adv(n_dates, n_names, seed)
    adv[:, -1] *= 0.01
    w = np.full((n_dates, n_names), 1.0 / n_names)
    return SyntheticBook("thin_adv", w, adv)


def rotating_book(n_dates: int, n_names: int, seed: int) -> SyntheticBook:
    """Weights flip between two halves every 5 days — high turnover book."""

    _, adv = _prices_and_adv(n_dates, n_names, seed)
    w = np.zeros((n_dates, n_names))
    half = n_names // 2
    for t in range(n_dates):
        if (t // 5) % 2 == 0:
            w[t, :half] = 1.0 / half
        else:
            w[t, half:] = 1.0 / (n_names - half)
    return SyntheticBook("rotating", w, adv)


BOOK_GENERATORS: dict[str, Callable[[int, int, int], SyntheticBook]] = {
    "uniform": uniform_book,
    "concentrated": concentrated_book,
    "thin_adv": thin_adv_book,
    "rotating": rotating_book,
}


def resolve_books(
    names: Iterable[str] | None = None,
) -> dict[str, Callable[[int, int, int], SyntheticBook]]:
    if names is None:
        return dict(BOOK_GENERATORS)
    out: dict[str, Callable[[int, int, int], SyntheticBook]] = {}
    for name in names:
        key = str(name)
        if key not in BOOK_GENERATORS:
            raise ValueError(f"unknown book generator: {key}")
        out[key] = BOOK_GENERATORS[key]
    return out


def capacity_metrics(
    book: SyntheticBook,
    *,
    aum: float,
    participation_cap: float,
    impact_coeff: float = 0.1,
) -> dict[str, float | int]:
    """Feasibility + cost metrics for one book at one AUM level.

    Required rebalance notional is ``|w_t - w_{t-1}| * aum`` per name (the
    first date rebalances from cash). A date is feasible when every name's
    required notional fits in ``participation_cap * adv_dollar[t, n]``.
    days_to_trade is the max over names of required/(cap*ADV); impact_bps is
    the mean ``1e4 * impact_coeff * sqrt(participation)`` across executed
    notional, the standard square-root market-impact form.
    """
    if not np.isfinite(aum) or aum <= 0:
        raise ValueError("aum must be finite and positive")
    if not np.isfinite(participation_cap) or not 0.0 < participation_cap <= 1.0:
        raise ValueError("participation_cap must be in (0, 1]")
    if not np.isfinite(impact_coeff) or impact_coeff < 0:
        raise ValueError("impact_coeff must be finite and non-negative")
    w = book.weights
    adv = book.adv_dollar
    if np.any(~np.isfinite(adv)) or np.any(adv <= 0):
        raise ValueError("adv_dollar must be finite and positive everywhere")
    if np.any(~np.isfinite(w)):
        raise ValueError("weights must be finite")
    prev = np.vstack([np.zeros((1, w.shape[1])), w[:-1]])
    required = np.abs(w - prev) * aum
    ceiling = participation_cap * adv
    participation = np.divide(required, adv, out=np.zeros_like(required), where=adv > 0)
    feasible = bool(np.all(required <= ceiling + _EPS))
    days = np.divide(required, ceiling, out=np.full_like(required, np.inf), where=ceiling > 0)
    days_to_trade = float(np.max(days)) if np.isfinite(days).all() else float("inf")
    traded = participation > 0
    impact_bps = (
        float(1e4 * impact_coeff * np.sqrt(participation[traded]).mean()) if traded.any() else 0.0
    )
    return {
        "feasible": int(feasible),
        "max_participation": float(participation.max()),
        "mean_participation": float(participation[traded].mean()) if traded.any() else 0.0,
        "days_to_trade": days_to_trade,
        "impact_bps": impact_bps,
    }


def run_capacity_bench(
    books: Iterable[SyntheticBook] | None = None,
    *,
    n_dates: int = 126,
    n_names: int = 32,
    seed: int = 11,
    aum_grid: Sequence[float] = (1e6, 1e7, 5e7, 1e8, 5e8, 1e9),
    participation_cap: float = 0.10,
    impact_coeff: float = 0.1,
) -> tuple[pl.DataFrame, dict[str, object]]:
    """Score every book × AUM grid cell and assemble the sealed receipt payload."""
    if n_dates < 2 or n_names < 2:
        raise ValueError("n_dates and n_names must be >= 2")
    books_l = (
        list(books)
        if books is not None
        else [gen(n_dates, n_names, seed + i) for i, gen in enumerate(BOOK_GENERATORS.values())]
    )
    if not books_l:
        raise ValueError("at least one book is required")
    grid = [float(a) for a in aum_grid]
    if not grid or any(a <= 0 or not np.isfinite(a) for a in grid):
        raise ValueError("aum_grid entries must be finite and positive")
    rows: list[dict[str, object]] = []
    book_meta: list[dict[str, object]] = []
    for book in books_l:
        book_meta.append(
            {
                "name": book.name,
                "n_dates": int(book.weights.shape[0]),
                "n_names": int(book.weights.shape[1]),
                "weights_sha256": hash_bytes(np.ascontiguousarray(book.weights).tobytes()),
                "adv_sha256": hash_bytes(np.ascontiguousarray(book.adv_dollar).tobytes()),
            }
        )
        for aum in grid:
            m = capacity_metrics(
                book, aum=aum, participation_cap=participation_cap, impact_coeff=impact_coeff
            )
            rows.append(
                {
                    "book": book.name,
                    "aum": aum,
                    "participation_cap": float(participation_cap),
                    "status": "ok",
                    **m,
                }
            )
    frame = pl.DataFrame(rows)
    inputs_sha256 = hash_bytes(
        canonical_json_bytes(
            {
                "books": book_meta,
                "aum_grid": grid,
                "participation_cap": float(participation_cap),
                "impact_coeff": float(impact_coeff),
                "seed": int(seed),
            }
        )
    )
    receipt: dict[str, object] = {
        "schema": CAPACITY_SCHEMA,
        "kind": "capacity_overlay_eval",
        "data_label": "SYNTHETIC",
        "live_pnl_claim": False,
        "dev_only": True,
        "generated_at": datetime.now(UTC).isoformat(),
        "git_revision": git_revision(),
        "seed": int(seed),
        "books": book_meta,
        "inputs_sha256": inputs_sha256,
        "n_rows": len(rows),
        "results": rows,
    }
    return frame, receipt


def capacity_contract_errors(receipt: Mapping[str, object]) -> list[str]:
    """Fail-closed contract for a ``capacity_overlay.v1`` payload (writer + verifier)."""
    research_blob = {k: v for k, v in receipt.items() if k != "live_pnl_claim"}
    errors: list[str] = []
    if receipt.get("schema") != CAPACITY_SCHEMA:
        errors.append("schema_not_capacity_v1")
    if receipt.get("kind") != "capacity_overlay_eval":
        errors.append("kind_not_capacity_overlay_eval")
    if receipt.get("data_label") != "SYNTHETIC":
        errors.append("data_label_not_synthetic")
    if receipt.get("live_pnl_claim") is not False:
        errors.append("live_pnl_claim_not_false")
    if receipt.get("dev_only") is not True:
        errors.append("dev_only_not_true")
    if not family_blob_forbidden_metrics_absent(research_blob):
        errors.append("forbidden_metric_keys")
    results = receipt.get("results")
    if isinstance(results, list):
        n_rows = receipt.get("n_rows")
        if n_rows is not None and n_rows != len(results):
            errors.append("n_rows_mismatch")
    return errors


def capacity_dataset_identity(receipt: Mapping[str, object]) -> dict[str, object]:
    """Book content digests bound by a v2 ``dataset_hash``."""
    books = receipt.get("books")
    if not isinstance(books, list):
        raise ValueError("capacity receipt has no books block")
    dataset: dict[str, object] = {}
    for meta in books:
        if not isinstance(meta, Mapping):
            raise ValueError("capacity book metadata is not an object")
        dataset[str(meta.get("name"))] = {
            "weights_sha256": meta.get("weights_sha256"),
            "adv_sha256": meta.get("adv_sha256"),
        }
    return dataset


def capacity_params(receipt: Mapping[str, object]) -> dict[str, object]:
    """Run parameters bound by a v2 ``params_hash``."""
    books = receipt.get("books")
    return {
        "seed": receipt.get("seed"),
        "inputs_sha256": receipt.get("inputs_sha256"),
        "books": sorted(str(meta.get("name")) for meta in books if isinstance(meta, Mapping))
        if isinstance(books, list)
        else None,
    }


def capacity_verdict(receipt: Mapping[str, object]) -> str:
    """pass iff every scored row reported ``status == "ok"``."""
    results = receipt.get("results")
    if not isinstance(results, list) or not results:
        return "fail"
    return (
        "pass"
        if all(isinstance(row, Mapping) and row.get("status") == "ok" for row in results)
        else "fail"
    )


def capacity_receipt_v2(receipt: Mapping[str, object]) -> dict[str, object]:
    """Wrap a ``capacity_overlay.v1`` payload in the unified ``receipt.v2`` envelope.

    The v1 payload is embedded verbatim under ``payload``; the envelope binds
    the book content digests, run params, this module's source hash, and the
    loaded numeric stack. Validates the v1 contract first — a malformed v1
    receipt is never wrapped.
    """
    from quant_fund.research.receipt_v2 import build_receipt_v2

    if capacity_contract_errors(receipt):
        raise ValueError("capacity receipt violates its synthetic research contract")
    return build_receipt_v2(
        kind=str(receipt["kind"]),
        data_label=str(receipt["data_label"]),
        dataset=capacity_dataset_identity(receipt),
        params=capacity_params(receipt),
        code_files=(Path(__file__),),
        verdict=capacity_verdict(receipt),
        payload=dict(receipt),
        generated_at=str(receipt["generated_at"]),
        revision=str(receipt["git_revision"]),
    )


def capacity_v2_consistency_errors(envelope: Mapping[str, object]) -> list[str]:
    """Re-derive a capacity receipt.v2 envelope's bound digests from its payload."""
    errors: list[str] = []
    payload = envelope.get("payload")
    if not isinstance(payload, Mapping):
        return ["payload_not_object"]
    contract_errors = capacity_contract_errors(payload)
    errors.extend(f"payload_{name}" for name in contract_errors)
    if contract_errors:
        return errors
    try:
        dataset = capacity_dataset_identity(payload)
    except ValueError as exc:
        return [*errors, f"payload_{exc}"]
    if hash_bytes(canonical_json_bytes(dataset)) != envelope.get("dataset_hash"):
        errors.append("dataset_hash_mismatch")
    if hash_bytes(canonical_json_bytes(capacity_params(payload))) != envelope.get("params_hash"):
        errors.append("params_hash_mismatch")
    if capacity_verdict(payload) != envelope.get("verdict"):
        errors.append("verdict_mismatch")
    return errors


def write_capacity_receipt(
    receipt: Mapping[str, object],
    receipts_dir: Path | str = Path("receipts"),
    *,
    receipt_version: int = 1,
) -> Path:
    """Seal a capacity receipt to ``receipts/capacity_eval_<hash>.json``.

    The filename hash is the sha256 of the canonical receipt payload; the
    same digest is embedded as ``receipt_sha256``. The write is atomic.
    ``receipt_version=2`` wraps the v1 payload in the unified ``receipt.v2``
    envelope before sealing.
    """
    from quant_fund.research.receipt_v2 import seal_receipt

    if receipt_version == 1:
        if capacity_contract_errors(receipt):
            raise ValueError("capacity receipt violates the honesty contract")
        body: Mapping[str, object] = receipt
    elif receipt_version == 2:
        body = capacity_receipt_v2(receipt)
    else:
        raise ValueError(f"receipt_version must be 1 or 2, got {receipt_version!r}")
    payload = seal_receipt(body)
    path = Path(receipts_dir) / f"capacity_eval_{payload['receipt_sha256'][:16]}.json"
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def format_capacity_table(frame: pl.DataFrame) -> str:
    """Render the capacity curve: feasible fraction and days-to-trade per AUM."""
    cols = ["book", "aum", "feasible", "max_participation", "days_to_trade", "impact_bps"]
    header = " | ".join(f"{c:>18}" for c in cols)
    lines = [header, "-+-".join("-" * 18 for _ in cols)]
    for row in frame.sort(["book", "aum"]).iter_rows(named=True):
        cells = [
            f"{row['book']:>18}",
            f"{row['aum']:>18.0f}",
            f"{row['feasible']:>18d}",
            f"{row['max_participation']:>18.4f}",
            f"{row['days_to_trade']:>18.3f}",
            f"{row['impact_bps']:>18.2f}",
        ]
        lines.append(" | ".join(cells))
    return "\n".join(lines)
