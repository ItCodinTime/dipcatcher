"""arena_table: honest pooled ranking over merged SOTA loss shards (synthetic).

Fabricated npz shards match the real schema written by
``scripts/sota_eval_kronos.py`` (``crps_matrix`` [R,M], ``pinball_cube``
[R,M,T], ``asset_ids``, ``model_names``, ``meta_json``).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_PATH = Path(__file__).resolve().parents[2] / "scripts" / "arena_table.py"
_SPEC = importlib.util.spec_from_file_location("arena_table", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
arena = importlib.util.module_from_spec(_SPEC)
sys.modules["arena_table"] = arena
_SPEC.loader.exec_module(arena)

MODELS = ["kronos_small", "timesfm", "dip_garch_t", "dip_fast"]


def _write_losses(
    path: Path,
    *,
    model_names: list[str] | None = None,
    crps: np.ndarray,
    pinball: np.ndarray | None = None,
    asset_names: tuple[str, ...] = ("BTCUSDT",),
    targets: tuple[str, ...] = ("kronos_small", "timesfm"),
    extra_meta: dict | None = None,
) -> Path:
    crps = np.asarray(crps, dtype=float)
    n_rows, n_models = crps.shape
    names = model_names or MODELS[:n_models]
    if pinball is None:
        pinball = np.stack([crps * (k + 1) / 10.0 for k in range(3)], axis=2)
    meta = {
        "asset_names": list(asset_names),
        "targets": list(targets),
        "config": {
            "lookback": 400,
            "window": 250,
            "garch_window": 750,
            "origins_per_asset": n_rows // max(1, len(asset_names)),
            "samples_per_origin": 16,
            "seed": 7,
            "taus": [0.05, 0.5, 0.95],
        },
        "bars_sha256": {"btc.parquet": "a" * 64},
        "artifact_sha256": {t: "b" * 64 for t in targets},
        "bar_interval_ns": 86400000000000,
        "scoring_contract": "native_shapes_timesfm_point_first.v2",
    }
    if extra_meta:
        meta.update(extra_meta)
    np.savez_compressed(
        path,
        crps_matrix=crps,
        pinball_cube=np.asarray(pinball, dtype=float),
        asset_ids=np.arange(n_rows) % max(1, len(asset_names)),
        model_names=np.asarray(names),
        meta_json=np.array(json.dumps(meta)),
    )
    return path


def _crps_fixture() -> np.ndarray:
    # column order = MODELS: kronos_small 0.30, timesfm 0.10, dip_garch_t 0.20,
    # dip_fast 0.25 (with one NaN row -> 2 finite).
    return np.array(
        [
            [0.30, 0.10, 0.20, 0.25],
            [0.30, 0.10, 0.20, 0.25],
            [0.30, 0.10, 0.20, np.nan],
        ]
    )


def test_watermark_is_first_line(tmp_path: Path) -> None:
    path = _write_losses(tmp_path / "m.losses.npz", crps=_crps_fixture())
    report = arena.build_report(path)
    assert report.splitlines()[0] == "research-only; no live-PnL claim"


def test_leaderboard_ranks_by_finite_mean(tmp_path: Path) -> None:
    path = _write_losses(tmp_path / "m.losses.npz", crps=_crps_fixture())
    losses = arena.load_losses(path)
    rows = arena.rows_from_losses(losses)
    arena.rank_rows(rows, "dip_garch_t")
    order = [r.model for r in arena.ordered_rows(rows)]
    assert order == ["timesfm", "dip_garch_t", "dip_fast", "kronos_small"]
    assert [r.rank for r in arena.ordered_rows(rows)] == [1, 2, 3, 4]


def test_nan_rows_counted_honestly(tmp_path: Path) -> None:
    path = _write_losses(tmp_path / "m.losses.npz", crps=_crps_fixture())
    losses = arena.load_losses(path)
    rows = arena.rows_from_losses(losses)
    fast = next(r for r in rows if r.model == "dip_fast")
    assert fast.n_finite == 2  # one NaN row dropped, not silently filled
    assert fast.mean_crps == pytest.approx(0.25)
    full = next(r for r in rows if r.model == "dip_garch_t")
    assert full.n_finite == 3
    report = arena.build_report(path)
    assert "NaN losses present" in report
    fast_line = next(ln for ln in report.splitlines() if "`dip_fast`" in ln)
    assert "| 2 |" in fast_line


def test_baseline_delta_math(tmp_path: Path) -> None:
    path = _write_losses(tmp_path / "m.losses.npz", crps=_crps_fixture())
    report = arena.build_report(path, baseline="dip_garch_t")
    fast_line = next(ln for ln in report.splitlines() if "`dip_fast`" in ln)
    assert "+0.050000" in fast_line  # 0.25 - 0.20
    assert "+25.0%" in fast_line
    tfm_line = next(ln for ln in report.splitlines() if "| `timesfm`" in ln)
    assert "-0.100000" in tfm_line
    assert "-50.0%" in tfm_line
    base_line = next(ln for ln in report.splitlines() if "| `dip_garch_t`" in ln)
    assert "+0.000000" in base_line


def test_type_column_flags_published_vs_dip(tmp_path: Path) -> None:
    path = _write_losses(tmp_path / "m.losses.npz", crps=_crps_fixture())
    losses = arena.load_losses(path)
    rows = arena.rows_from_losses(losses)
    types = {r.model: r.type for r in rows}
    assert types["kronos_small"] == "published"
    assert types["timesfm"] == "published"
    assert types["dip_garch_t"] == "dip"
    assert types["dip_fast"] == "dip"


def test_mcs_column_from_companion_receipt(tmp_path: Path) -> None:
    path = _write_losses(tmp_path / "m.losses.npz", crps=_crps_fixture())
    receipt = {
        "mcs_status": "computed",
        "mcs_included": {
            "kronos_small": False,
            "timesfm": False,
            "dip_garch_t": True,
            "dip_fast": True,
        },
        "mcs_p_values": {
            "kronos_small": 0.01,
            "timesfm": 0.02,
            "dip_garch_t": 1.0,
            "dip_fast": 0.5,
        },
    }
    (tmp_path / "m.json").write_text(json.dumps(receipt), encoding="utf-8")
    report = arena.build_report(path)
    assert "mcs@0.10" in report
    fast_line = next(ln for ln in report.splitlines() if "`dip_fast`" in ln)
    assert "in (p=0.500)" in fast_line
    kronos_line = next(ln for ln in report.splitlines() if "`kronos_small`" in ln)
    assert "out (p=0.010)" in kronos_line


def test_no_receipt_omits_mcs_column(tmp_path: Path) -> None:
    path = _write_losses(tmp_path / "m.losses.npz", crps=_crps_fixture())
    report = arena.build_report(path)
    assert "mcs@0.10" not in report
    assert "no companion receipt" in report


def test_pinball_means_rendered_per_tau(tmp_path: Path) -> None:
    path = _write_losses(tmp_path / "m.losses.npz", crps=_crps_fixture())
    report = arena.build_report(path)
    assert "pin@0.05" in report and "pin@0.5" in report and "pin@0.95" in report
    # pinball cube = crps * (k+1)/10 -> tau .05 mean for dip_garch_t = 0.02
    base_line = next(ln for ln in report.splitlines() if "| `dip_garch_t`" in ln)
    assert "0.020000" in base_line


def test_receipt_only_json_input(tmp_path: Path) -> None:
    receipt = {
        "n_rows": 3,
        "mean_crps_pooled": {"timesfm": 0.1, "dip_garch_t": 0.2, "dip_solo": 0.3},
        "mean_pinball_pooled": {
            "timesfm": {"0.05": 0.01, "0.5": 0.02, "0.95": 0.01},
            "dip_garch_t": {"0.05": 0.02, "0.5": 0.04, "0.95": 0.02},
            "dip_solo": {"0.05": 0.03, "0.5": 0.06, "0.95": 0.03},
        },
        "coverage": {
            "timesfm": {"finite_crps": 3},
            "dip_garch_t": {"finite_crps": 3},
            "dip_solo": {"finite_crps": 2},
        },
        "config": {"targets": ["timesfm"], "taus": [0.05, 0.5, 0.95]},
        "challengers": ["dip_garch_t", "dip_solo"],
        "mcs_included": {"timesfm": False, "dip_garch_t": True, "dip_solo": True},
        "mcs_p_values": {"timesfm": 0.01, "dip_garch_t": 1.0, "dip_solo": 0.7},
    }
    path = tmp_path / "r.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    report = arena.build_report(path)
    assert "receipt-only" in report
    solo_line = next(ln for ln in report.splitlines() if "`dip_solo`" in ln)
    assert "| 2 |" in solo_line  # n_finite from coverage block
    order = [ln for ln in report.splitlines() if ln.startswith("| ")]
    assert order[1].startswith("| 1 | `timesfm`")


def test_compare_mode_paired_delta(tmp_path: Path) -> None:
    a = _write_losses(
        tmp_path / "a.losses.npz",
        model_names=["dip_x", "dip_y"],
        targets=(),
        crps=np.array([[0.2, 0.3], [0.4, np.nan], [np.nan, 0.3]]),
    )
    b = _write_losses(
        tmp_path / "b.losses.npz",
        model_names=["dip_x", "dip_z"],
        targets=(),
        crps=np.array([[0.1, 9.9], [0.2, 9.9], [0.3, 9.9]]),
    )
    report = arena.compare_report(arena.load_losses(a), arena.load_losses(b))
    x_line = next(ln for ln in report.splitlines() if "`dip_x`" in ln)
    # rows finite in both: rows 0,1 -> mean_A 0.3, mean_B 0.15, delta -0.15
    assert "| 2 |" in x_line  # n_pair
    assert "0.300000" in x_line and "0.150000" in x_line
    assert "-0.150000" in x_line and "-50.0%" in x_line
    y_line = next(ln for ln in report.splitlines() if "`dip_y`" in ln)
    assert "a only" in y_line and "0.300000" in y_line
    z_line = next(ln for ln in report.splitlines() if "`dip_z`" in ln)
    assert "b only" in z_line and "9.900000" in z_line


def test_compare_unpaired_row_grids_disclosed(tmp_path: Path) -> None:
    a = _write_losses(
        tmp_path / "a.losses.npz",
        model_names=["dip_x"],
        targets=(),
        crps=np.array([[0.2], [0.4]]),
    )
    b = _write_losses(
        tmp_path / "b.losses.npz",
        model_names=["dip_x"],
        targets=(),
        crps=np.array([[0.1], [0.2], [0.3]]),
    )
    report = arena.compare_report(arena.load_losses(a), arena.load_losses(b))
    assert "both*" in report
    assert "row counts differ" in report
    x_line = next(ln for ln in report.splitlines() if "`dip_x`" in ln)
    assert "0.300000" in x_line and "0.200000" in x_line  # own-file pooled means
    assert "-0.100000" in x_line


def test_missing_base_marked_not_present(tmp_path: Path) -> None:
    path = _write_losses(tmp_path / "m.losses.npz", crps=_crps_fixture())
    report = arena.build_report(path, baseline="dip_nonexistent")
    assert "NOT PRESENT" in report
    fast_line = next(ln for ln in report.splitlines() if "`dip_fast`" in ln)
    assert "| — | — |" in fast_line  # delta cells unavailable


def test_legacy_loss_matrix_shard(tmp_path: Path) -> None:
    # v1 shards: loss_matrix + model_names only — no pinball, no meta.
    path = tmp_path / "legacy.losses.npz"
    np.savez_compressed(
        path,
        loss_matrix=np.array([[0.4, 0.2], [0.3, 0.1]]),
        model_names=np.asarray(["kronos", "dip_empirical"]),
    )
    losses = arena.load_losses(path)
    rows = arena.rows_from_losses(losses)
    types = {r.model: r.type for r in rows}
    assert types["kronos"] == "published"  # name hint without meta
    assert types["dip_empirical"] == "dip"
    report = arena.build_report(path)
    assert "pin@" not in report  # no pinball columns


def test_cli_writes_out_file(tmp_path: Path) -> None:
    path = _write_losses(tmp_path / "m.losses.npz", crps=_crps_fixture())
    out = tmp_path / "report.md"
    rc = arena.main(["--losses", str(path), "--out", str(out)])
    assert rc == 0
    text = out.read_text(encoding="utf-8")
    assert text.splitlines()[0] == "research-only; no live-PnL claim"
    assert "| rank |" in text
