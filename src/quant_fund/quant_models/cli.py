"""CLI for quant-models engines. Research only; no live IBKR / MES."""

from __future__ import annotations

import json

import typer

qm_app = typer.Typer(
    help="quant-models engines (BSM, HRP, GEX decide). Research only; no live broker."
)


@qm_app.command("bs-price")
def bs_price_cmd(
    spot: float = typer.Option(100.0),
    strike: float = typer.Option(100.0),
    tau: float = typer.Option(1.0, "--tau"),
    rate: float = typer.Option(0.05, "--rate"),
    div: float = typer.Option(0.0, "--div"),
    sigma: float = typer.Option(0.20, "--sigma"),
    kind: str = typer.Option("call", "--kind"),
) -> None:
    from quant_fund.quant_models.black_scholes import bs_price, price_bounds

    price = float(bs_price(spot, strike, tau, rate, div, sigma, kind))
    lo, hi = price_bounds(spot, strike, tau, rate, div, kind)
    typer.echo(
        json.dumps(
            {
                "price": price,
                "lower_bound": float(lo),
                "upper_bound": float(hi),
                "research_only": True,
                "live_pnl_claim": False,
            }
        )
    )


@qm_app.command("iv")
def iv_cmd(
    spot: float = typer.Option(100.0),
    strike: float = typer.Option(100.0),
    tau: float = typer.Option(1.0, "--tau"),
    rate: float = typer.Option(0.05, "--rate"),
    div: float = typer.Option(0.0, "--div"),
    market_price: float = typer.Option(..., "--market-price"),
    kind: str = typer.Option("call", "--kind"),
) -> None:
    from quant_fund.quant_models.black_scholes import implied_volatility

    iv = implied_volatility(spot, strike, tau, rate, div, market_price, kind)
    typer.echo(json.dumps({"implied_vol": iv, "research_only": True, "live_pnl_claim": False}))


@qm_app.command("greeks")
def greeks_cmd(
    spot: float = typer.Option(100.0),
    strike: float = typer.Option(100.0),
    tau: float = typer.Option(0.25, "--tau"),
    rate: float = typer.Option(0.042, "--rate"),
    div: float = typer.Option(0.005, "--div"),
    sigma: float = typer.Option(0.28, "--sigma"),
    kind: str = typer.Option("call", "--kind"),
) -> None:
    from quant_fund.quant_models.greeks import scaled_greeks

    g = scaled_greeks(spot, strike, tau, rate, div, sigma, kind)
    g.update({"research_only": True, "live_pnl_claim": False})
    typer.echo(json.dumps(g))


@qm_app.command("validate-greeks")
def validate_greeks_cmd() -> None:
    from quant_fund.quant_models.greeks import validate_greeks

    worst = validate_greeks(verbose=True)
    ok = worst < 2e-4
    typer.echo(json.dumps({"worst_rel_err": worst, "ok": ok, "research_only": True}))
    if not ok:
        raise typer.Exit(code=1)


@qm_app.command("hrp")
def hrp_cmd() -> None:
    from quant_fund.quant_models.hrp import (
        generate_ldp_example,
        hrp_weights,
        inverse_variance_weights,
    )

    panel, _cols = generate_ldp_example()
    cov = panel.cov().to_numpy()
    corr = panel.corr().to_numpy()
    hrp = hrp_weights(cov, corr)
    ivp = inverse_variance_weights(cov)
    typer.echo(
        json.dumps(
            {
                "hrp": hrp.tolist(),
                "ivp": ivp.tolist(),
                "hrp_top5_pct": float(sum(sorted(hrp, reverse=True)[:5]) * 100),
                "research_only": True,
                "live_pnl_claim": False,
            }
        )
    )


@qm_app.command("gex-decide")
def gex_decide_cmd(
    gex: float = typer.Option(..., "--gex", help="Previous-close net gamma; sign is the regime"),
    sofar: float = typer.Option(..., "--sofar", help="Open-to-now log return"),
    equity: float = typer.Option(1_000_000.0, "--equity"),
    spot: float = typer.Option(5000.0, "--spot"),
    leverage: float = typer.Option(1.0, "--leverage"),
    no_fade: bool = typer.Option(False, "--no-fade"),
) -> None:
    from quant_fund.quant_models.gex import last_hour_decide

    d = last_hour_decide(gex, sofar, equity, spot, leverage=leverage, fade_long_gamma=not no_fade)
    typer.echo(
        json.dumps(
            {
                "action": d.action,
                "contracts": d.contracts,
                "leg": d.leg,
                "why": d.why,
                "broker": None,
                "research_only": True,
                "live_pnl_claim": False,
            }
        )
    )
