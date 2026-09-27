"""Report assembly + rendering on synthetic heads (SYNTHETIC labels)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest
from quant_fund.research.catalog import family_blob_forbidden_metrics_absent

from quant_fund.research.explainability import build_report, write_report
from quant_fund.research.explainability.report import EXPLAINABILITY_REPORT_SCHEMA

from .conftest import FEATURES


def test_build_report_end_to_end(planted_model) -> None:
    model, x, y = planted_model
    report = build_report(
        model,
        x,
        y,
        times=np.arange(y.shape[0]),
        scoring="pinball:0.5",
        top_k=4,
        n_blocks=3,
        n_repeats=3,
        seed=42,
        synthetic=True,
        generated_at="2026-01-01T00:00:00+00:00",
    )
    assert report.model_name == "linear_head"
    assert report.model_family == "test"
    assert report.feature_names == tuple(FEATURES)
    assert report.attribution.attributions[0].feature == "signal_core"
    assert len(report.pd_curves) == 4
    assert report.drift is not None and report.drift.n_blocks == 3
    assert report.baseline_score > 0.0


def test_report_dict_is_json_safe_and_honest(planted_model) -> None:
    model, x, y = planted_model
    report = build_report(model, x, y, n_blocks=2, n_repeats=2, seed=1, synthetic=True)
    payload = report.to_dict()
    blob = json.dumps(payload)  # raises if not serializable
    assert payload["schema"] == EXPLAINABILITY_REPORT_SCHEMA
    assert payload["synthetic"] is True
    assert payload["claim"] == "research_only"
    # Honesty contract: no forbidden headline metric keys anywhere in the tree.
    assert family_blob_forbidden_metrics_absent(payload)
    assert '"signal_core"' in blob


def test_report_is_deterministic_given_seed_and_stamp(planted_model) -> None:
    model, x, y = planted_model
    kwargs = dict(n_blocks=3, n_repeats=3, seed=19, generated_at="2026-01-01T00:00:00Z")
    first = build_report(model, x, y, **kwargs).to_dict()
    second = build_report(model, x, y, **kwargs).to_dict()
    assert first == second


def test_markdown_labels_synthetic_and_lists_importances(planted_model) -> None:
    model, x, y = planted_model
    report = build_report(model, x, y, n_blocks=2, n_repeats=2, seed=2, synthetic=True)
    md = report.to_markdown()
    assert "SYNTHETIC" in md
    assert "signal_core" in md
    assert "Permutation importances" in md
    assert "research_only" in md
    assert "| rank | feature |" in md


def test_html_is_self_contained(planted_model) -> None:
    model, x, y = planted_model
    report = build_report(model, x, y, times=np.arange(y.shape[0]), n_blocks=3, n_repeats=2, seed=3)
    page = report.to_html()
    assert 'src="data:image/png;base64,' in page
    assert "http://" not in page
    assert "https://" not in page
    assert "<script" not in page
    assert "signal_core" in page
    # Base64 payload decodes to a real PNG signature.
    match = re.search(r"data:image/png;base64,([A-Za-z0-9+/=]+)", page)
    assert match is not None
    import base64

    assert base64.b64decode(match.group(1))[:4] == b"\x89PNG"


def test_write_report_persists_all_artifacts(planted_model, tmp_path: Path) -> None:
    model, x, y = planted_model
    report = build_report(model, x, y, n_blocks=2, n_repeats=2, seed=4, synthetic=True)
    paths = write_report(report, tmp_path / "explainability")
    for key in ("markdown", "html", "json"):
        assert paths[key].is_file()
        sidecar = paths[key].with_name(f"{paths[key].name}.sha256")
        assert sidecar.is_file()
        import hashlib

        assert sidecar.read_text().strip() == hashlib.sha256(paths[key].read_bytes()).hexdigest()
    parsed = json.loads(paths["json"].read_text())
    assert parsed["schema"] == EXPLAINABILITY_REPORT_SCHEMA
    assert parsed["importances"][0]["feature"] == "signal_core"


def test_build_report_requires_model_or_predict() -> None:
    with pytest.raises(ValueError, match="model or a predict"):
        build_report(None, np.ones((10, 2)), np.ones(10))


def test_build_report_predict_callable_path(planted_model) -> None:
    model, x, y = planted_model
    report = build_report(
        None,
        x,
        y,
        predict=model.predict,
        feature_names=FEATURES,
        n_blocks=2,
        n_repeats=2,
        seed=5,
        model_name="callable_head",
    )
    assert report.model_name == "callable_head"
    assert report.attribution.attributions[0].feature == "signal_core"


def test_drift_disabled_for_too_few_rows() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(3, 2))
    y = x[:, 0]
    report = build_report(
        None,
        x,
        y,
        predict=lambda b: np.asarray(b)[:, 0],
        n_blocks=4,
        seed=1,
    )
    assert report.drift is None
