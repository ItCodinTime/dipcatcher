"""``qm`` CLI surface: bs-price, iv, greeks, validate-greeks, hrp, gex-decide."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from quant_fund.cli.main import app


def _run(args: list[str]) -> tuple[int, dict]:
    res = CliRunner().invoke(app, ["qm", *args])
    if res.exit_code != 0:
        return res.exit_code, {}
    return 0, json.loads(res.stdout)


def test_qm_bs_price_outputs_bounds() -> None:
    code, out = _run(["bs-price", "--spot", "100", "--strike", "100", "--sigma", "0.2"])
    assert code == 0
    assert out["lower_bound"] <= out["price"] <= out["upper_bound"]
    assert out["price"] > 0
    assert out["research_only"] is True and out["live_pnl_claim"] is False


def test_qm_iv_recovers_sigma() -> None:
    # bs_price(100,100,1y,5%,0,20%) ≈ 10.45; invert through the CLI.
    code, out = _run(["iv", "--market-price", "10.450583572185565"])
    assert code == 0
    assert abs(out["implied_vol"] - 0.2) < 1e-6


def test_qm_greeks_shape() -> None:
    code, out = _run(["greeks"])
    assert code == 0
    for k in ("delta", "gamma", "vega", "theta", "rho"):
        assert k in out
    assert 0.0 < out["delta"] < 1.0
    assert out["live_pnl_claim"] is False


def test_qm_validate_greeks_passes() -> None:
    res = CliRunner().invoke(app, ["qm", "validate-greeks"])
    assert res.exit_code == 0
    # A human-readable table precedes the JSON receipt line.
    out = json.loads(res.stdout.strip().splitlines()[-1])
    assert out["ok"] is True
    assert out["worst_rel_err"] < 2e-4


def test_qm_hrp_outputs() -> None:
    code, out = _run(["hrp"])
    assert code == 0
    hrp = out["hrp"]
    assert abs(sum(hrp) - 1.0) < 1e-6
    assert all(w >= 0 for w in hrp)
    assert abs(sum(out["ivp"]) - 1.0) < 1e-6


def test_qm_gex_decide_follow_and_fade() -> None:
    code, out = _run(["gex-decide", "--gex", "-1e9", "--sofar", "0.004"])
    assert code == 0
    assert out["action"] == "LONG" and out["leg"] == "follow"
    assert out["broker"] is None
    code, out = _run(["gex-decide", "--gex", "1e9", "--sofar", "0.004", "--no-fade"])
    assert out["action"] == "FLAT"
    # Missing required options fail closed.
    res = CliRunner().invoke(app, ["qm", "gex-decide"])
    assert res.exit_code != 0
