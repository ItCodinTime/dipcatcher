"""Loss-column correction must preserve paired evidence and artifact identity."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

_PATH = Path(__file__).resolve().parents[2] / "scripts" / "splice_kronos_fix.py"
_SPEC = importlib.util.spec_from_file_location("sota_splice", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
splice = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(splice)


def _write_pair(tmp_path: Path, *, stem: str = "h4_btcusdt_4h") -> tuple[Path, Path]:
    original = tmp_path / f"{stem}.losses.npz"
    prefix = "h4fix_" if stem.startswith("h4_") else "d1fix_"
    corrected = tmp_path / f"{prefix}{stem.split('_', 1)[1]}.losses.npz"
    names = ["kronos_small", "other_target", "dip_student_t", "dip_empirical"]
    crps = np.array([[0.5, 0.3, np.nan, 0.2], [0.4, 0.3, 0.2, 0.1], [0.3, 0.3, 0.1, 0.2]])
    pin = np.repeat(crps[:, :, None] / 2, 3, axis=2)
    meta = {
        "asset_names": ["BTCUSDT"],
        "targets": ["kronos_small", "other_target"],
        "config": {
            "origins_per_asset": 3,
            "samples_per_origin": 16,
            "lookback": 400,
            "window": 250,
            "garch_window": 750,
            "seed": 7,
            "taus": [0.05, 0.5, 0.95],
        },
        "bars_sha256": {"btcusdt_4h.parquet": "a" * 64},
        "artifact_sha256": {
            "kronos_small": "b" * 64,
            "kronos_small_tokenizer": "c" * 64,
            "other_target": "d" * 64,
        },
    }
    data = {"crps": crps, "pin": pin, "aids": np.zeros(3, dtype=int), "names": names, "meta": meta}
    splice.save(original, data)
    data["names"] = [names[i] for i in (0, 2, 3)]
    data["crps"] = crps[:, [0, 2, 3]].copy()
    data["pin"] = pin[:, [0, 2, 3]].copy()
    data["crps"][:, 0] /= 2
    data["pin"][:, 0] /= 2
    meta["targets"] = ["kronos_small"]
    meta["artifact_sha256"] = {"kronos_small": "b" * 64, "kronos_small_tokenizer": "e" * 64}
    splice.save(corrected, data)
    return original, corrected


def test_correction_updates_artifacts_and_preserves_other_models(tmp_path: Path) -> None:
    original, corrected = _write_pair(tmp_path)
    result = splice.splice_pair(original, corrected)
    baseline, fix = splice.load(original), splice.load(corrected)
    np.testing.assert_array_equal(result["crps"][:, 0], fix["crps"][:, 0])
    np.testing.assert_array_equal(result["pin"][:, 0], fix["pin"][:, 0])
    np.testing.assert_array_equal(result["crps"][:, 1:], baseline["crps"][:, 1:])
    np.testing.assert_array_equal(result["pin"][:, 1:], baseline["pin"][:, 1:])
    assert result["meta"]["artifact_sha256"]["kronos_small_tokenizer"] == "e" * 64
    assert result["meta"]["artifact_sha256"]["other_target"] == "d" * 64
    transform = result["meta"]["transformations"][-1]
    assert transform["previous_artifact_sha256"]["kronos_small_tokenizer"] == "c" * 64
    assert (
        transform["inputs"]["corrected"]["sha256"]
        == hashlib.sha256(corrected.read_bytes()).hexdigest()
    )
    assert transform["alignment"]["origin_timestamps_verified"] is False
    assert splice.load(original)["meta"]["artifact_sha256"]["kronos_small_tokenizer"] == "c" * 64


@pytest.mark.parametrize("field", ["crps", "pin"])
@pytest.mark.parametrize("change", ["tiny", "nan_to_zero"])
def test_exact_alignment_rejects_rounding_and_missing_value_changes(
    tmp_path: Path,
    field: str,
    change: str,
) -> None:
    original, corrected = _write_pair(tmp_path)
    data = splice.load(corrected)
    index = (1, 2) if field == "crps" else (1, 2, 0)
    if change == "tiny":
        data[field][index] += 1e-12
    else:
        index = (0, 1) if field == "crps" else (0, 1, 0)
        data[field][index] = 0.0
    splice.save(corrected, data)
    with pytest.raises(ValueError, match="exact alignment required"):
        splice.splice_pair(original, corrected)


@pytest.mark.parametrize(
    "change, message",
    [
        ("bars", "bars provenance"),
        ("seed", "protocol mismatch"),
        ("missing_protocol", "missing protocol"),
        ("model", "weights differ"),
        ("tokenizer", "missing artifact identity"),
        ("assets", "asset identity"),
        ("challengers", "challenger set"),
    ],
)
def test_pair_identity_is_required(tmp_path: Path, change: str, message: str) -> None:
    original, corrected = _write_pair(tmp_path)
    data = splice.load(corrected)
    if change == "bars":
        data["meta"]["bars_sha256"]["btcusdt_4h.parquet"] = "f" * 64
    elif change == "seed":
        data["meta"]["config"]["seed"] = 11
    elif change == "missing_protocol":
        del data["meta"]["config"]["lookback"]
    elif change == "model":
        data["meta"]["artifact_sha256"]["kronos_small"] = "f" * 64
    elif change == "tokenizer":
        del data["meta"]["artifact_sha256"]["kronos_small_tokenizer"]
    elif change == "assets":
        data["meta"]["asset_names"] = ["ETHUSDT"]
    elif change == "challengers":
        data["names"][-1] = "different_challenger"
    splice.save(corrected, data)
    with pytest.raises(ValueError, match=message):
        splice.splice_pair(original, corrected)


def test_complete_case_preserves_correction_and_exposes_coverage(tmp_path: Path) -> None:
    original, corrected = _write_pair(tmp_path)
    fixed = splice.splice_pair(original, corrected)
    fixed["pin"][2, 1, 0] = np.nan
    subset = splice.complete_case(fixed)
    assert "dip_student_t" not in subset["names"]
    assert subset["meta"]["artifact_sha256"]["kronos_small_tokenizer"] == "e" * 64
    assert subset["meta"]["kronos_small_corrected"] == fixed["meta"]["kronos_small_corrected"]
    assert subset["meta"]["transformations"][0] == fixed["meta"]["transformations"][0]
    transform = subset["meta"]["transformations"][-1]
    assert transform["n_rows_input"] == 3 and transform["n_rows_output"] == 2
    assert transform["n_rows_dropped"] == 1
    assert transform["model_scored_counts_before_exclusion"]["dip_student_t"] == 2
    assert transform["retained_input_row_indices"] == [0, 1]
    assert len(fixed["meta"]["transformations"]) == 1


def test_main_preflights_all_pairs_before_any_write(tmp_path: Path) -> None:
    _write_pair(tmp_path, stem="d1_aaa")
    _, bad = _write_pair(tmp_path, stem="d1_zzz")
    data = splice.load(bad)
    data["pin"][1, 2, 0] += 1e-12
    splice.save(bad, data)
    output = tmp_path / "outputs"
    with pytest.raises(ValueError, match="exact alignment required"):
        splice.main([str(tmp_path), "--output-dir", str(output)])
    assert not output.exists()


def test_main_check_only_and_separate_output_preserve_inputs(tmp_path: Path) -> None:
    original, corrected = _write_pair(tmp_path)
    before = {p: p.read_bytes() for p in (original, corrected)}
    output = tmp_path / "outputs"
    assert splice.main([str(tmp_path), "--output-dir", str(output), "--check-only"]) == 0
    assert not output.exists()
    assert splice.main([str(tmp_path), "--output-dir", str(output)]) == 0
    assert len(list(output.glob("*.npz"))) == 2
    for path, contents in before.items():
        assert path.read_bytes() == contents
    with np.load(output / "h4_btcusdt_4h.cc.npz", allow_pickle=False) as z:
        meta = json.loads(str(z["meta_json"]))
    assert len(meta["transformations"]) == 2
    assert meta["artifact_sha256"]["kronos_small_tokenizer"] == "e" * 64
