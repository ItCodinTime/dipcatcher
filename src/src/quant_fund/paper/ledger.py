"""Persist paper / shadow ledgers under data/metadata/paper/."""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from quant_fund.execution.simulated_broker import BrokerSnapshot, OrderRecord, SimulatedBroker
from quant_fund.metrics.analytics import analytics_export_digest, validate_analytics_export
from quant_fund.utils.hashing import hash_bytes


def _safe_run_id(run_id: str) -> str:
    value = str(run_id)
    if (
        not value.strip()
        or value in {".", ".."}
        or Path(value).name != value
        or any(char in value for char in ("/", "\\"))
    ):
        raise ValueError("paper run_id must be a non-empty path-safe identifier")
    return value


def _promotion_receipt_digest(receipt: dict[str, Any]) -> str:
    """Canonical self-excluding digest for a paper promotion receipt."""
    normalized = dict(receipt)
    normalized.pop("receipt_sha256", None)
    return hash_bytes(
        json.dumps(normalized, sort_keys=True, separators=(",", ":"), allow_nan=True).encode()
    )


def paper_root(data_root: Path | str, subdir: str = "paper") -> Path:
    relative = Path(subdir)
    if (
        not subdir.strip()
        or subdir.startswith(("/", chr(92)))
        or relative.is_absolute()
        or bool(relative.drive)
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ValueError("paper ledger subdir must be a safe relative path")
    path = Path(data_root) / "metadata" / relative
    path.mkdir(parents=True, exist_ok=True)
    return path


def _nan() -> float:
    return float("nan")


def _order_row(rec: OrderRecord, asof: datetime | None) -> dict[str, Any]:
    o = rec.order
    fill = rec.fill
    return {
        "asof": asof,
        "slot": str(rec.slot),
        "order_id": str(o.order_id),
        "security_id": str(o.security_id),
        "side": str(o.side.value),
        "quantity": float(o.quantity),
        "status": str(o.status.value),
        "reject_reason": "" if rec.reject_reason is None else str(rec.reject_reason),
        "signal_time": o.signal_time,
        "order_time": o.order_time,
        "fill_price": _nan() if fill is None else float(fill.price),
        "fill_qty": _nan() if fill is None else float(fill.quantity),
        "fee": _nan() if fill is None else float(fill.fee),
        "spread_cost": _nan() if fill is None else float(fill.spread_cost),
        "impact_cost": _nan() if fill is None else float(fill.impact_cost),
        "fill_id": "" if fill is None else str(fill.fill_id),
        "is_partial": False if fill is None else bool(fill.is_partial),
    }


def _snap_row(snap: BrokerSnapshot) -> dict[str, Any]:
    return {
        "asof": snap.asof,
        "slot": str(snap.slot),
        "cash": float(snap.cash),
        "nav": float(snap.nav),
        "gross": float(snap.gross),
        "net": float(snap.net),
        "n_names": float(len([s for s, q in snap.shares.items() if abs(q) > 1e-12])),
    }


def _position_rows(snap: BrokerSnapshot) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sid in sorted(snap.shares):
        qty = snap.shares[sid]
        if abs(qty) <= 1e-12:
            continue
        rows.append(
            {
                "asof": snap.asof,
                "slot": snap.slot,
                "security_id": sid,
                "shares": float(qty),
            }
        )
    return rows


class PaperLedger:
    """Append-only paper ledger (orders, equity, positions, broker state, meta)."""

    SCHEMA_VERSION = 2

    def __init__(
        self,
        data_root: Path | str,
        run_id: str,
        subdir: str = "paper",
        *,
        load_existing: bool = False,
    ) -> None:
        self.run_id = _safe_run_id(run_id)
        self.root = paper_root(data_root, subdir) / self.run_id
        self.root.mkdir(parents=True, exist_ok=True)
        self._orders: list[dict[str, Any]] = []
        self._equity: list[dict[str, Any]] = []
        self._shadow_equity: list[dict[str, Any]] = []
        self._positions: list[dict[str, Any]] = []
        self._cash_events: list[dict[str, Any]] = []
        if load_existing:
            for name, target in (
                ("orders", self._orders),
                ("equity", self._equity),
                ("shadow_equity", self._shadow_equity),
                ("positions", self._positions),
                ("cash_ledger", self._cash_events),
            ):
                path = self.root / f"{name}.parquet"
                if path.is_file():
                    target.extend(pl.read_parquet(path).to_dicts())
        self._meta: dict[str, Any] = {
            "run_id": run_id,
            "schema_version": self.SCHEMA_VERSION,
            "label": "PAPER_SIMULATED",
            "disclaimer": (
                "Paper/shadow ledger uses simulated fills. "
                "Not live broker connectivity. SYNTHETIC bars ⇒ research-only diagnostics."
            ),
        }

    def set_meta(self, **kwargs: Any) -> None:
        self._meta.update(kwargs)

    def record_orders(self, records: list[OrderRecord], asof: datetime | None) -> None:
        for rec in records:
            self._orders.append(_order_row(rec, asof))
            fill = rec.fill
            if fill is not None:
                notional = float(fill.quantity) * float(fill.price)
                fee_tot = float(fill.fee) + float(fill.spread_cost) + float(fill.impact_cost)
                side = str(rec.order.side.value)
                cash_delta = -notional - fee_tot if side == "buy" else notional - fee_tot
                self._cash_events.append(
                    {
                        "asof": asof,
                        "slot": rec.slot,
                        "order_id": str(rec.order.order_id),
                        "security_id": str(fill.security_id),
                        "cash_delta": float(cash_delta),
                        "notional": float(notional),
                        "fees": float(fee_tot),
                        "side": side,
                    }
                )

    def record_snapshot(self, snap: BrokerSnapshot) -> None:
        self._equity.append(_snap_row(snap))
        self._positions.extend(_position_rows(snap))

    def record_broker(self, broker: SimulatedBroker, asof: datetime | None) -> None:
        self.record_snapshot(broker.snapshot(asof=asof))

    def record_shadow_equity(self, row: dict[str, Any]) -> None:
        """Record a capital-free shadow diagnostic row for append-safe resume."""
        self._shadow_equity.append(dict(row))

    def save_broker_state(
        self,
        champion: SimulatedBroker,
        shadow: SimulatedBroker | None,
        *,
        last_decision: datetime | None,
        last_exec: datetime | None,
        step: int,
        resume_fingerprint: str | None = None,
        mark_ages: dict[str, int] | None = None,
    ) -> Path:
        blob = {
            "run_id": self.run_id,
            "schema_version": self.SCHEMA_VERSION,
            "step": int(step),
            "last_decision": last_decision.isoformat() if last_decision else None,
            "last_exec": last_exec.isoformat() if last_exec else None,
            "resume_fingerprint": resume_fingerprint,
            "mark_ages": {
                str(security_id): int(age) for security_id, age in (mark_ages or {}).items()
            },
            "champion": champion.to_dict(),
            "shadow": None if shadow is None else shadow.to_dict(),
        }
        path = self.root / "broker_state.json"
        path.write_text(json.dumps(blob, indent=2, default=str))
        return path

    def flush(self) -> dict[str, Path]:
        paths: dict[str, Path] = {}
        if self._orders:
            op = self.root / "orders.parquet"
            pl.DataFrame(self._orders, infer_schema_length=None).write_parquet(op)
            paths["orders"] = op
        if self._equity:
            ep = self.root / "equity.parquet"
            pl.DataFrame(self._equity, infer_schema_length=None).write_parquet(ep)
            paths["equity"] = ep
        if self._shadow_equity:
            sp = self.root / "shadow_equity.parquet"
            pl.DataFrame(self._shadow_equity, infer_schema_length=None).write_parquet(sp)
            paths["shadow_equity"] = sp
        if self._positions:
            pp = self.root / "positions.parquet"
            pl.DataFrame(self._positions, infer_schema_length=None).write_parquet(pp)
            paths["positions"] = pp
        if self._cash_events:
            cp = self.root / "cash_ledger.parquet"
            pl.DataFrame(self._cash_events, infer_schema_length=None).write_parquet(cp)
            paths["cash_ledger"] = cp
        mp = self.root / "meta.json"
        mp.write_text(json.dumps(self._meta, indent=2, default=str))
        paths["meta"] = mp
        data_meta_paper = self.root.parent  # .../metadata/paper
        (data_meta_paper / "latest_run.json").write_text(
            json.dumps({"run_id": self.run_id, "path": str(self.root)}, indent=2)
        )
        paths["latest"] = data_meta_paper / "latest_run.json"
        return paths

    def load_equity(self) -> pl.DataFrame:
        path = self.root / "equity.parquet"
        if not path.is_file():
            return pl.DataFrame()
        return pl.read_parquet(path)

    def load_shadow_equity(self) -> pl.DataFrame:
        path = self.root / "shadow_equity.parquet"
        if not path.is_file():
            return pl.DataFrame()
        return pl.read_parquet(path)

    def write_promotion_dry_run(self, receipt: dict[str, Any]) -> Path:
        path = self.root / "promotion_dry_run.json"
        path.write_text(json.dumps(receipt, indent=2, default=str))
        return path


def load_broker_state(
    data_root: Path | str, run_id: str, subdir: str = "paper"
) -> dict[str, Any] | None:
    path = paper_root(data_root, subdir) / _safe_run_id(run_id) / "broker_state.json"
    if not path.is_file():
        return None
    try:
        state = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None
    return state if isinstance(state, dict) else None


def latest_run_id(data_root: Path | str, subdir: str = "paper") -> str | None:
    path = paper_root(data_root, subdir) / "latest_run.json"
    if not path.is_file():
        return None
    try:
        blob = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None
    if not isinstance(blob, dict) or not isinstance(blob.get("run_id"), str):
        return None
    return _safe_run_id(blob["run_id"]) if blob["run_id"] else None


def promotion_dry_run(
    *,
    run_id: str | None = None,
    mean_l1: float,
    max_l1: float,
    n_steps: int,
    champion_nav: float | None,
    shadow_gross: float | None,
    max_mean_l1: float,
    min_steps: int,
    data_source: str,
    rolling_mean_l1: float | None = None,
    rolling_window: int | None = None,
    challenger_metrics: dict[str, Any] | None = None,
    primary_challenger: str | None = None,
    risk_accounting: dict[str, int] | None = None,
    kill_tripped_mid_run: bool = False,
    allow_missing_divergence: bool = False,
) -> dict[str, Any]:
    """Shadow→champion promotion dry-run. Never touches live capital or aliases.

    Richer Wave-5 report: optional rolling L1, multi-challenger snapshot,
    risk-gate reject accounting. ``would_promote_live`` remains hard-closed.
    """
    required_numbers = {
        "mean_l1": mean_l1,
        "max_l1": max_l1,
        "max_mean_l1": max_mean_l1,
    }
    invalid_required = any(
        (
            not math.isfinite(float(value))
            and not (
                allow_missing_divergence
                and key in {"mean_l1", "max_l1"}
                and math.isnan(float(value))
            )
        )
        or float(value) < 0.0
        for key, value in required_numbers.items()
    )
    if invalid_required:
        raise ValueError("promotion divergence values must be finite and non-negative")
    if int(n_steps) < 0 or int(min_steps) < 0:
        raise ValueError("promotion step counts must be non-negative")
    for name, value in (
        ("champion_nav", champion_nav),
        ("shadow_gross", shadow_gross),
        ("rolling_mean_l1", rolling_mean_l1),
    ):
        if value is not None and (not math.isfinite(float(value)) or float(value) < 0.0):
            raise ValueError(f"{name} must be finite and non-negative when provided")
    if rolling_window is not None and int(rolling_window) < 1:
        raise ValueError("rolling_window must be positive when provided")
    synthetic = str(data_source).upper() == "SYNTHETIC"
    reasons: list[str] = []
    ok = True
    if n_steps < int(min_steps):
        ok = False
        reasons.append(f"insufficient_steps:{n_steps}<{min_steps}")
    if mean_l1 != mean_l1:  # NaN
        ok = False
        reasons.append("missing_divergence")
    elif mean_l1 > float(max_mean_l1):
        ok = False
        reasons.append(f"mean_l1_above_threshold:{mean_l1:.4f}>{max_mean_l1}")
    if kill_tripped_mid_run:
        ok = False
        reasons.append("kill_switch_tripped_mid_run")
    if synthetic:
        # SYNTHETIC evidence is never promotable to live — dry-run may still
        # report would_promote_paper for infrastructure checks.
        reasons.append("synthetic_evidence_not_live_promotable")
    receipt: dict[str, Any] = {
        "would_promote_paper": bool(ok),
        "would_promote_live": False,  # hard closed — no live promotion from paper dry-run
        "mean_l1": float(mean_l1),
        "max_l1": float(max_l1),
        "n_steps": int(n_steps),
        "champion_nav": champion_nav,
        "shadow_gross": shadow_gross,
        "max_mean_l1": float(max_mean_l1),
        "min_steps": int(min_steps),
        "reasons": reasons,
        "data_source": "SYNTHETIC" if synthetic else data_source,
        "live_pnl_claim": False,
        "research_only": True,
        "note": "dry_run_only_no_live_capital",
        "schema_version": 2,
    }
    if run_id is not None:
        receipt["run_id"] = _safe_run_id(run_id)
    if rolling_mean_l1 is not None:
        receipt["rolling_mean_l1"] = float(rolling_mean_l1)
    if rolling_window is not None:
        receipt["rolling_window"] = int(rolling_window)
    if primary_challenger is not None:
        receipt["primary_challenger"] = str(primary_challenger)
    if challenger_metrics is not None:
        receipt["challenger_metrics"] = challenger_metrics
    if risk_accounting is not None:
        receipt["risk_accounting"] = risk_accounting
    receipt["kill_tripped_mid_run"] = bool(kill_tripped_mid_run)
    receipt["receipt_sha256"] = _promotion_receipt_digest(receipt)
    # Fail-closed honesty invariants (Wave 9): never emit a live-promote receipt.
    if receipt.get("would_promote_live") is not False:
        raise ValueError("promotion_dry_run must set would_promote_live=False")
    if receipt.get("live_pnl_claim") is not False:
        raise ValueError("promotion_dry_run must set live_pnl_claim=False")
    if receipt.get("research_only") is not True:
        raise ValueError("promotion_dry_run must set research_only=True")
    return receipt


# ---------------------------------------------------------------------------
# Ledger schema validation (Wave 6)
# ---------------------------------------------------------------------------

LEDGER_META_REQUIRED = ("run_id", "schema_version", "label", "disclaimer")
LEDGER_EQUITY_REQUIRED = ("asof", "slot", "cash", "nav", "gross", "net")  # snapshot schema
LEDGER_ORDER_REQUIRED = (
    "asof",
    "slot",
    "order_id",
    "security_id",
    "side",
    "quantity",
    "status",
)
LEDGER_BROKER_STATE_REQUIRED = (
    "run_id",
    "schema_version",
    "step",
    "champion",
)
LEDGER_CHAMPION_STATE_REQUIRED = ("slot", "cash", "shares", "allow_capital")


def _valid_iso_cursor(value: Any) -> bool:
    """Return whether a persisted resume cursor is a non-empty ISO datetime."""
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return True


def _finite_number(value: Any) -> bool:
    """Accept JSON numbers, but reject booleans, strings, NaN, and infinity."""
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


LEDGER_CASH_REQUIRED = (
    "asof",
    "slot",
    "order_id",
    "security_id",
    "cash_delta",
    "notional",
    "fees",
    "side",
)


PROMOTION_DRY_RUN_REQUIRED = (
    "would_promote_paper",
    "would_promote_live",
    "live_pnl_claim",
    "research_only",
)


def validate_promotion_dry_run_receipt(promo: dict[str, Any]) -> list[str]:
    """Fail-closed checks for a promotion_dry_run JSON / receipt dict.

    Missing honesty flags or ``would_promote_live=true`` / ``live_pnl_claim=true``
    / ``research_only=false`` are errors (never warnings).
    """
    errors: list[str] = []
    if not isinstance(promo, dict):
        return ["promotion_receipt_not_object"]
    for key in PROMOTION_DRY_RUN_REQUIRED:
        if key not in promo:
            errors.append(f"promotion_missing:{key}")
    if "would_promote_paper" in promo and not isinstance(promo["would_promote_paper"], bool):
        errors.append("promotion_would_promote_paper_must_be_bool")
    # Present-but-wrong honesty flags
    if "would_promote_live" in promo and promo.get("would_promote_live") is not False:
        errors.append("promotion_would_promote_live_must_be_false")
    if "live_pnl_claim" in promo and promo.get("live_pnl_claim") is not False:
        errors.append("promotion_live_pnl_claim_must_be_false")
    if "research_only" in promo and promo.get("research_only") is not True:
        errors.append("promotion_research_only_must_be_true")
    for key in ("mean_l1", "max_l1", "max_mean_l1", "rolling_mean_l1"):
        if key in promo:
            value = promo[key]
            missing_divergence_nan = (
                isinstance(value, float)
                and math.isnan(value)
                and key in {"mean_l1", "max_l1"}
                and promo.get("would_promote_paper") is False
                and "missing_divergence" in (promo.get("reasons") or [])
            )
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or (
                    not missing_divergence_nan
                    and (not math.isfinite(float(value)) or float(value) < 0.0)
                )
            ):
                errors.append(f"promotion_{key}_must_be_finite_nonnegative")
    for key in ("n_steps", "min_steps", "rolling_window"):
        if key in promo:
            value = promo[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                errors.append(f"promotion_{key}_must_be_nonnegative_int")
    if (
        "rolling_window" in promo
        and isinstance(promo["rolling_window"], int)
        and promo["rolling_window"] < 1
    ):
        errors.append("promotion_rolling_window_must_be_positive")
    if "run_id" in promo:
        if not isinstance(promo["run_id"], str):
            errors.append("promotion_run_id_invalid")
        else:
            try:
                _safe_run_id(promo["run_id"])
            except (TypeError, ValueError):
                errors.append("promotion_run_id_invalid")
    # schema_version >= 2 seals are mandatory (unsigned v1 legacy still allowed).
    schema_ver = promo.get("schema_version")
    require_seal = (
        isinstance(schema_ver, int) and not isinstance(schema_ver, bool) and schema_ver >= 2
    )
    if require_seal and "receipt_sha256" not in promo:
        errors.append("promotion_receipt_sha256_missing")
    if "receipt_sha256" in promo:
        digest = promo["receipt_sha256"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            errors.append("promotion_receipt_sha256_invalid")
        elif digest != _promotion_receipt_digest(promo):
            errors.append("promotion_receipt_sha256_mismatch")
    if (
        isinstance(promo.get("mean_l1"), (int, float))
        and isinstance(promo.get("max_l1"), (int, float))
        and float(promo["max_l1"]) < float(promo["mean_l1"])
    ):
        errors.append("promotion_max_l1_below_mean_l1")
    if (
        isinstance(promo.get("rolling_mean_l1"), (int, float))
        and isinstance(promo.get("max_l1"), (int, float))
        and float(promo["rolling_mean_l1"]) > float(promo["max_l1"])
    ):
        errors.append("promotion_rolling_mean_l1_above_max_l1")
    if promo.get("would_promote_paper") is True:
        if (
            isinstance(promo.get("n_steps"), int)
            and isinstance(promo.get("min_steps"), int)
            and promo["n_steps"] < promo["min_steps"]
        ):
            errors.append("promotion_claimed_without_min_steps")
        if (
            isinstance(promo.get("mean_l1"), (int, float))
            and isinstance(promo.get("max_mean_l1"), (int, float))
            and float(promo["mean_l1"]) > float(promo["max_mean_l1"])
        ):
            errors.append("promotion_claimed_above_mean_l1_threshold")
        if promo.get("kill_tripped_mid_run") is True:
            errors.append("promotion_claimed_after_kill_switch")
    return errors


def validate_ledger_schema(
    run_dir: Path | str,
    *,
    expect_version: int | None = None,
    require_complete: bool = False,
) -> dict[str, Any]:
    """Validate on-disk paper ledger against SCHEMA_VERSION expectations.

    Returns a report: ``ok``, ``errors``, ``warnings``, ``present``, ``schema_version``.
    Does not raise — callers decide fail-closed policy. Research/infra helper only.
    """
    root = Path(run_dir)
    errors: list[str] = []
    warnings: list[str] = []
    present: dict[str, bool] = {}
    schema_version: int | None = None

    meta_path = root / "meta.json"
    present["meta.json"] = meta_path.is_file()
    meta: dict[str, Any] = {}
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"meta.json_invalid_json:{exc}")
        else:
            if not isinstance(meta, dict):
                errors.append("meta.json_not_object")
                meta = {}
            for key in LEDGER_META_REQUIRED:
                if key not in meta:
                    errors.append(f"meta_missing:{key}")
            if meta.get("run_id") != root.name:
                errors.append("meta_run_id_mismatch")
            raw_ver = meta.get("schema_version")
            if isinstance(raw_ver, bool) or not isinstance(raw_ver, int) or raw_ver < 0:
                errors.append("meta_schema_version_not_int")
                schema_version = None
            else:
                schema_version = raw_ver
            expected = (
                int(expect_version) if expect_version is not None else PaperLedger.SCHEMA_VERSION
            )
            if schema_version is not None and schema_version != expected:
                warnings.append(f"schema_version_mismatch:got={schema_version}:expected={expected}")
            if meta.get("live_pnl_claim") is True:
                errors.append("meta_live_pnl_claim_must_be_false_or_absent")
    else:
        errors.append("meta.json_missing")

    state = {}
    state_path = root / "broker_state.json"
    present["broker_state.json"] = state_path.is_file()
    if state_path.is_file():
        try:
            state = json.loads(state_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"broker_state_invalid_json:{exc}")
        else:
            if not isinstance(state, dict):
                errors.append("broker_state_not_object")
                state = {}
            for key in LEDGER_BROKER_STATE_REQUIRED:
                if key not in state:
                    errors.append(f"broker_state_missing:{key}")
            if state.get("run_id") != meta.get("run_id"):
                errors.append("broker_state_run_id_mismatch")
            fingerprint = state.get("resume_fingerprint")
            if fingerprint is not None and (
                not isinstance(fingerprint, str)
                or len(fingerprint) != 64
                or any(character not in "0123456789abcdef" for character in fingerprint)
            ):
                errors.append("broker_state_resume_fingerprint_invalid")
            state_step = state.get("step")
            if isinstance(state_step, bool) or not isinstance(state_step, int) or state_step < 0:
                errors.append("broker_state_step_invalid")
            for cursor_key in ("last_decision", "last_exec"):
                cursor = state.get(cursor_key)
                if cursor is not None and not _valid_iso_cursor(cursor):
                    errors.append(f"broker_state_{cursor_key}_invalid")
            mark_ages = state.get("mark_ages")
            if mark_ages is not None:
                if not isinstance(mark_ages, dict):
                    errors.append("broker_state_mark_ages_not_object")
                else:
                    for security_id, age in mark_ages.items():
                        if (
                            not isinstance(security_id, str)
                            or isinstance(age, bool)
                            or not isinstance(age, int)
                            or age < 0
                        ):
                            errors.append("broker_state_mark_ages_invalid")
                            break
            champ = state.get("champion")
            if not isinstance(champ, dict):
                errors.append("broker_state_champion_not_object")
            else:
                for key in LEDGER_CHAMPION_STATE_REQUIRED:
                    if key not in champ:
                        errors.append(f"broker_state_champion_missing:{key}")
                slot = champ.get("slot")
                if not isinstance(slot, str) or not slot.strip():
                    errors.append("broker_state_champion_slot_invalid")
                cash = champ.get("cash")
                if not _finite_number(cash):
                    errors.append("broker_state_champion_cash_invalid")
                shares = champ.get("shares")
                if not isinstance(shares, dict):
                    errors.append("broker_state_champion_shares_not_object")
                else:
                    for security_id, quantity in shares.items():
                        if not isinstance(security_id, str) or not _finite_number(quantity):
                            errors.append("broker_state_champion_shares_invalid")
                            break
                if not isinstance(champ.get("allow_capital"), bool):
                    errors.append("broker_state_champion_allow_capital_invalid")
                for count_key in ("n_orders", "n_fills"):
                    count = champ.get(count_key)
                    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                        errors.append(f"broker_state_champion_{count_key}_invalid")
                history = champ.get("history")
                if history is not None:
                    if not isinstance(history, list):
                        errors.append("broker_state_champion_history_not_list")
                    else:
                        for index, record in enumerate(history):
                            if not isinstance(record, dict):
                                errors.append(
                                    f"broker_state_champion_history_record_invalid:{index}"
                                )
                                continue
                            for key in ("order", "reject_reason", "fill", "slot"):
                                if key not in record:
                                    errors.append(
                                        f"broker_state_champion_history_missing:{index}:{key}"
                                    )
                        if (
                            isinstance(champ.get("n_orders"), int)
                            and len(history) != champ["n_orders"]
                        ):
                            errors.append("broker_state_champion_history_order_count_mismatch")
                        fill_count = sum(
                            1
                            for record in history
                            if isinstance(record, dict) and record.get("fill") is not None
                        )
                        if isinstance(champ.get("n_fills"), int) and fill_count != champ["n_fills"]:
                            errors.append("broker_state_champion_history_fill_count_mismatch")
    else:
        message = "broker_state.json_missing"
        (errors if require_complete else warnings).append(message)

    equity_path = root / "equity.parquet"
    present["equity.parquet"] = equity_path.is_file()
    equity_rows = None
    if equity_path.is_file():
        try:
            eq = pl.read_parquet(equity_path)
            equity_rows = eq.height
            cols = set(eq.columns)
            # Accept either ledger snapshot schema (asof/nav/cash) or loop equity
            # schema (event_time/nav/cash) — both are valid Wave-2+ artifacts.
            if "nav" not in cols:
                errors.append("equity_missing:nav")
            if "cash" not in cols and "asof" in cols:
                # snapshot schema always has cash
                errors.append("equity_missing:cash")
            if "asof" not in cols and "event_time" not in cols:
                errors.append("equity_missing:asof_or_event_time")
        except Exception as exc:  # noqa: BLE001 — report, don't crash validator
            errors.append(f"equity_unreadable:{type(exc).__name__}:{exc}")
    elif require_complete:
        errors.append("equity.parquet_missing")

    # Positive broker cursors must have matching durable champion equity rows.
    # Otherwise resume could skip observations that have no auditable NAV history.
    state_step = state.get("step") if isinstance(state, dict) else None
    if isinstance(state_step, bool) or not isinstance(state_step, int) or state_step < 0:
        state_step = None
    if state_step is not None and state_step > 0:
        if not present["equity.parquet"]:
            errors.append("broker_state_step_requires_equity")
        elif equity_rows == 0:
            errors.append("broker_state_step_equity_empty")
        elif equity_rows is not None and equity_rows != state_step:
            errors.append(
                f"broker_state_step_equity_count_mismatch:step={state_step}:rows={equity_rows}"
            )

    orders_path = root / "orders.parquet"
    present["orders.parquet"] = orders_path.is_file()
    if orders_path.is_file():
        try:
            od = pl.read_parquet(orders_path)
            cols = set(od.columns)
            for key in ("order_id", "security_id", "side", "status"):
                if key not in cols:
                    errors.append(f"orders_missing:{key}")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"orders_unreadable:{type(exc).__name__}:{exc}")
    else:
        message = "orders.parquet_missing"
        (errors if require_complete else warnings).append(message)

    cash_path = root / "cash_ledger.parquet"
    present["cash_ledger.parquet"] = cash_path.is_file()
    if cash_path.is_file():
        try:
            cash = pl.read_parquet(cash_path)
            cols = set(cash.columns)
            for key in LEDGER_CASH_REQUIRED:
                if key not in cols:
                    errors.append(f"cash_ledger_missing:{key}")
            # Fail-closed: cash deltas must be finite when present.
            if (
                cash.height > 0
                and "cash_delta" in cols
                and not cash["cash_delta"].is_finite().all()
            ):
                errors.append("cash_ledger_cash_delta_nonfinite")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"cash_ledger_unreadable:{type(exc).__name__}:{exc}")
    elif require_complete:
        errors.append("cash_ledger.parquet_missing")

    promo_path = root / "promotion_dry_run.json"
    present["promotion_dry_run.json"] = promo_path.is_file()
    if promo_path.is_file():
        try:
            promo = json.loads(promo_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"promotion_dry_run_invalid_json:{exc}")
        else:
            promo_errors = validate_promotion_dry_run_receipt(promo)
            if "run_id" in promo and promo.get("run_id") != meta.get("run_id"):
                errors.append("promotion_run_id_mismatch")
            errors.extend(promo_errors)

    analytics_path = root / "analytics_export.json"
    present["analytics_export.json"] = analytics_path.is_file()
    if analytics_path.is_file():
        try:
            analytics = json.loads(analytics_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"analytics_export_invalid_json:{exc}")
        else:
            analytics_report = validate_analytics_export(analytics)
            if isinstance(analytics, dict):
                if "run_id" in analytics and analytics.get("run_id") != meta.get("run_id"):
                    errors.append("analytics_export_run_id_mismatch")
                digest = analytics.get("analytics_export_sha256")
                if digest is not None:
                    if (
                        not isinstance(digest, str)
                        or len(digest) != 64
                        or any(character not in "0123456789abcdef" for character in digest)
                    ):
                        errors.append("analytics_export_sha256_invalid")
                    elif digest != analytics_export_digest(analytics):
                        errors.append("analytics_export_sha256_mismatch")
            errors.extend(
                f"analytics_export_invalid:{error}" for error in analytics_report.get("errors", [])
            )
    else:
        message = "analytics_export.json_missing"
        (errors if require_complete else warnings).append(message)

    return {
        "ok": len(errors) == 0,
        "require_complete": require_complete,
        "errors": errors,
        "warnings": warnings,
        "present": present,
        "schema_version": schema_version,
        "run_dir": str(root),
        "role": "ledger_schema_validation",
        "live_pnl_claim": False,
        "research_only": True,
    }
