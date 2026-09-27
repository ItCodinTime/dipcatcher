# Causal liquidity and capacity checks

The next-open backtests use the **signal close's** ADV and volatility for the
following open's execution costs and participation limit. The next day's ADV
and volatility are not known at the open. If signal-close ADV is missing or
nonpositive, the attempted order is rejected. Invalid participation limits
(including a runtime assignment of zero or above one) also reject the order.
The reference backtest and the accelerated replay follow the same rule.

Use `capacity_sensitivity` to replay identical decisions with different
starting capital and conservative haircuts on the known ADV estimate:

```python
from quant_fund.backtest.engine import capacity_sensitivity

report = capacity_sensitivity(
    bars, weights, config,
    nav_levels=(100_000.0, 1_000_000.0, 10_000_000.0),
    adv_haircuts=(1.0, 0.5),
)
for case in report["cases"]:
    print(case["initial_nav"], case["adv_haircut"],
          case["executed_notional_fraction"], case["end_nav"])
```

Each cell reruns order sizing, risk limits, participation caps, cash checks,
and market-impact costs. The traded-notional fraction shows when an intended
allocation cannot be filled at a larger scale. `risk_or_liquidity_rejects`
includes both risk-gate and unavailable-liquidity rejects. The configured
transaction-cost model must be enabled. These are research diagnostics, not a
forecast of live liquidity, achievable returns, or fund capacity. Verify
that any supplied `adv` and `vol_20` fields themselves were computed using
information available by the signal close; this backtest cannot prove the
lineage of upstream data.
