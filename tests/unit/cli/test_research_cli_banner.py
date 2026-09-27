"""Research CLI always prints DATA_LABEL + BH-FDR families."""

from __future__ import annotations

from types import SimpleNamespace

from quant_fund.cli.main import format_data_label, format_fdr_families


def test_format_data_label_synthetic_and_vendor() -> None:
    assert format_data_label(synthetic=True, data_source="whatever") == "DATA_LABEL=SYNTHETIC"
    assert format_data_label(synthetic=False, data_source="vendor_x") == "DATA_LABEL=vendor_x"


def test_format_fdr_families_always_prints_three_buckets() -> None:
    hyps = [
        SimpleNamespace(family="calibration", reject_fdr=True),
        SimpleNamespace(family="calibration", reject_fdr=False),
        SimpleNamespace(family="discovery", reject_fdr=True),
        SimpleNamespace(family="bound", reject_fdr=False),
    ]
    line = format_fdr_families(hyps)
    assert line.startswith("BH-FDR families (split, never pooled):")
    assert "calibration n=2 rejects=1" in line
    assert "discovery n=1 rejects=1" in line
    assert "bound n=1 (not FDR-adjusted)" in line


def test_format_fdr_families_empty_still_prints() -> None:
    line = format_fdr_families([])
    assert "calibration n=0 rejects=0" in line
    assert "discovery n=0 rejects=0" in line
    assert "bound n=0" in line
