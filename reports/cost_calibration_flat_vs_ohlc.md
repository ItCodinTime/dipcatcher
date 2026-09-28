# Cost calibration — flat half-spread vs OHLC estimators

SYNTHETIC execution diagnostic only. Not market evidence. `live_pnl_claim=false`. Research-headline return ratios and equity curves are absent by construction.

- schema: `cost_calibration.v1`
- seed: `7`
- half-spread floor (bps): `1.0`
- planted relative full spread: `0.002`
- Corwin–Schultz closed-form relative: `0.0008280850981386401`
- Corwin–Schultz effective half-spread bps (floored): `4.1404254906932`

## Trial results

```
      spread_estimator |  half_spread_bps_floor |             commission |                 spread |             total_cost |                n_fills
-----------------------+------------------------+------------------------+------------------------+------------------------+-----------------------
                  flat |                   1.00 |              1855.4643 |              1855.4643 |              3710.9286 |                    115
        corwin_schultz |                   1.00 |              1850.7817 |              7477.6730 |              9328.4547 |                    115
          abdi_ranaldo |                   1.00 |              1855.4643 |              1855.4643 |              3710.9286 |                    115
                  roll |                   1.00 |              1854.8503 |              2944.5378 |              4799.3881 |                    116
```

## Notes

- Default path is flat `half_spread_bps`; calibrated estimators are opt-in.
- Calibrated cost is `max(floor, estimator_half_spread_bps)`.
- `run_backtest_fast` refuses any non-flat `spread_estimator`.
- Calibrated estimators: corwin_schultz, abdi_ranaldo, roll.
