import math
from pathlib import Path

from quant_fund.config import load_config
from quant_fund.research.agent import run_research


def test_research_recovers_synthetic_oracle(tmp_path: Path) -> None:
    cfg = load_config("configs/research.yaml")
    cfg.data.root = tmp_path
    cfg.data.synthetic_n_assets = 36
    cfg.data.synthetic_n_days = 220
    cfg.validation.train_bars = 80
    cfg.validation.val_bars = 20
    cfg.validation.test_bars = 20
    nb = run_research(cfg)
    assert nb.synthetic is True
    assert nb.data_source == "SYNTHETIC"
    assert nb.firm == "Artificial Hedge"
    assert "Sharpe" not in nb.disclaimer
    by = {r["name"]: r for r in nb.rankers}
    assert "oracle_raw" in by
    oracle = by["oracle_raw"]
    assert oracle["mean_ic"] > 0.15
    assert oracle["p_ic"] < 0.05
    assert "sharpe" not in oracle
    assert "volatility" in nb.families
    assert "reinforcement" in nb.families
    assert "drawdown" in nb.families
    assert "conformal" in nb.families
    conf = nb.families["conformal"]
    assert "sharpe" not in str(conf).lower()
    assert conf.get("aci", {}).get("coverage", 0.0) >= 0.85
    raw_cov = conf.get("gaussian_raw", {}).get("coverage", 1.0)
    assert raw_cov < 0.90
    scaled = conf.get("scaled_gaussian", {}).get("coverage")
    if scaled is not None:
        assert float(scaled) >= 0.80
    rl_feats = [str(f).lower() for f in nb.families.get("reinforcement", {}).get("features", [])]
    qb_feats = [str(f).lower() for f in nb.families.get("quantile_bandit", {}).get("features", [])]
    assert not any("planted" in f for f in rl_feats)
    assert not any("planted" in f for f in qb_feats)
    jp = nb.families.get("jackknife_plus", {})
    assert float(jp.get("coverage_floor", 0.8)) == 0.8
    assert float(jp.get("coverage", 0.0)) >= 0.78
    caps = nb.families.get("interval_risk", {})
    assert "bind_wide" in caps and "bind_tight" in caps
    wide_b = float(caps.get("bind_wide", float("nan")))
    tight_b = float(caps.get("bind_tight", float("nan")))
    assert str(caps.get("weight_rule")) == "equal_weight_per_date"
    if math.isfinite(wide_b) and math.isfinite(tight_b):
        assert wide_b >= tight_b
        if float(caps.get("frac_binding", 1.0)) < 1.0 - 1e-12:
            assert wide_b > tight_b
    sota = (
        "evalues",
        "jackknife_plus",
        "crc",
        "weighted_conformal",
        "interval_risk",
        "quantile_bandit",
    )
    metric = {
        "evalues": "e_final",
        "jackknife_plus": "coverage",
        "crc": "risk",
        "weighted_conformal": "coverage",
        "interval_risk": "frac_binding",
        "quantile_bandit": "mean_regret_vs_oracle",
    }
    for fam in sota:
        assert fam in nb.families
        blob = nb.families[fam]
        assert blob
        assert "sharpe" not in str(blob).lower()
        val = blob.get(metric[fam])
        assert val is not None
        assert math.isfinite(float(val))
    dumped = (tmp_path / "metadata" / "research" / "latest.md").read_text()
    assert "SYNTHETIC" in dumped
    assert "scientific lab" in dumped.lower()
    assert "sharpe" not in dumped.lower()
    h1 = next(h for h in nb.hypotheses if h.id == "H1_ranking_oracle")
    assert h1.reject_raw is True
    h7 = next(h for h in nb.hypotheses if h.id == "H7_aci_coverage")
    assert "nominal" in h7.statement.lower() or "α" in h7.statement
    mond = conf.get("mondrian_aci", {})
    assert mond.get("coverage", 0.0) >= 0.85
    assert mond.get("worst_x_coverage", 0.0) >= 0.75
    h8 = next(h for h in nb.hypotheses if h.id == "H8_mondrian_high_vol")
    assert "high-vol" in h8.statement.lower() or "X" in h8.statement
    h10 = next(h for h in nb.hypotheses if h.id == "H10_jackknife_coverage")
    assert "2α" in h10.statement or "2a" in h10.statement.lower()
    assert h10.family == "bound"
    assert h10.reject_fdr is False
    assert not math.isfinite(h10.p_value)
    assert h10.meets_floor == (
        float(jp.get("coverage", 0.0)) + 1e-12 >= float(jp.get("coverage_floor", 0.8))
    )
    h5 = next(h for h in nb.hypotheses if h.id == "H5_drawdown_brier")
    assert h5.p_value not in (0.0, 1.0) or not math.isfinite(h5.p_value)
    h13 = next(h for h in nb.hypotheses if h.id == "H13_interval_caps")
    assert h13.p_value != 0.0
    assert "H5-style" not in h13.test
    if str(caps.get("weight_rule")) == "equal_weight_per_date":
        assert "equal-weight" in h13.statement.lower()
    cal = {h.id for h in nb.hypotheses if h.family == "calibration"}
    disc = {h.id for h in nb.hypotheses if h.family == "discovery"}
    assert cal.isdisjoint(disc)
    assert "H10_jackknife_coverage" not in cal and "H10_jackknife_coverage" not in disc
    for hid in (
        "H9_eprocess_aci",
        "H10_jackknife_coverage",
        "H11_crc_var",
        "H12_weighted_cqr",
        "H13_interval_caps",
        "H14_quantile_thompson",
    ):
        next(h for h in nb.hypotheses if h.id == hid)
