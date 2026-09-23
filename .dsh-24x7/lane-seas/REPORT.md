# dip_seas — calendar-seasonality challenger report

**Model**: `dip_seas` — calendar-conditioned empirical distribution on the
trailing `garch_window` (750) close-to-close returns.

- Slot from the **target bar's** UTC `event_time`: daily shards (`bar_interval_ns
  == 86400e9`) → `dow` (7 buckets, Mon=0); otherwise → `hour4`
  (`hour_utc // 4`, 6 buckets; aligns with 00/08/16-UTC funding and session
  rotation). Trailing returns are bucketed by the calendar slot of the bar they
  land on; only bars `<= i` enter the pool (strictly causal).
- Slot quantiles at `LGBM_TAUS` shrink toward pooled empirical quantiles:
  `q = (n_s·q_slot + 40·q_global)/(n_s + 40)`. Slots with `<15` obs or
  non-finite quantiles fall back to the pooled fit (0 fallback rows on all 10
  shards — pools are ~125/slot on 4h, ~107/slot on 1d).
- **CRPS**: `crps_from_quantiles(y, LGBM_TAUS, q)` — the quantile-pinball
  trapezoid identity with the arena's disclosed exponential tail completion
  (same scoring family as `dip_qar`/`dip_stack`). Cross-check:
  `crps_empirical` on the deterministic 512-point quantile-grid sample agrees
  to mean |diff| ≤ 5.2e-7 per shard (meta field `crps_empirical_512_*`).
- Fully deterministic, no RNG. New file only: `scripts/_seas_col.py`.

## Provenance

Computed remote-first on `me@100.116.120.51` (`D:\dipcatcher`) after verifying
all 10 remote shard SHA256s match local; results scp'd back. Cross-machine
determinism spot-check (`h4f_btcusdt`): pinball + `target_time_ns`
bit-identical; CRPS max |diff| 1.7e-18 (last-ulp BLAS accumulation).

Artifacts (all under `.dsh-24x7/lane-seas/`):
- columns: `{d1,h4f}_{asset}_{1d,4h}_deep.seascol.npz` (10)
- spliced shards: `spliced/*.seas.npz` (10, 21 model columns each)
- receipts: `merge_d1.json` (+`.losses.npz`), `merge_h4f.json` (+`.losses.npz`)

## Results — pooled mean CRPS (1500 rows each, n_complete=1500)

| arena | dip_seas pooled CRPS | rank | MCS @0.10 | dip_seas NaN (crps/pin) |
|-------|---------------------|------|-----------|--------------------------|
| d1    | 0.015073            | 4/21 | **in**    | 0 / 0                    |
| h4f   | 0.005309            | 6/21 | **in**    | 0 / 0                    |

d1 ahead of: dip_qar, dip_conf_t, dip_blend, dip_empirical, dip_student_t,
dip_stack, dip_skt, dip_empirical_long, dip_gmm_k, dip_ewma_emp, dip_ewma_t,
dip_gauss, dip_lgbm_q — and all 4 published targets (timesfm 0.015917,
chronos2 0.016390, bolt 0.017857, kronos 0.064231; DM p < 1.2e-5 vs each).
Behind: dip_fhs (0.014945), dip_garch_t (0.014981), dip_regime (0.015065).

h4f ahead of: dip_blend, dip_empirical_long, dip_gmm_k, dip_empirical, dip_qar,
dip_skt, dip_conf_t, dip_student_t, dip_ewma_t, dip_gauss, dip_lgbm_q — and all
4 targets. Behind: dip_fhs, dip_garch_t, dip_stack, dip_regime, dip_ewma_emp.

dip_seas pooled pinball — d1: τ05=0.003217, τ50=0.010006, τ95=0.003730;
h4f: τ05=0.001072, τ50=0.003425, τ95=0.001639.

## Per-cell mean-CRPS delta vs `dip_garch_t` (seas − garch; negative = seas better)

| asset   | d1 seas   | d1 garch  | d1 delta   | h4f seas  | h4f garch | h4f delta  |
|---------|-----------|-----------|------------|-----------|-----------|------------|
| BNBUSDT | 0.012144  | 0.011898  | +0.000247  | 0.004311  | 0.004234  | +0.000077  |
| BTCUSDT | 0.012006  | 0.012068  | −0.000062  | 0.003657  | 0.003631  | +0.000026  |
| ETHUSDT | 0.016601  | 0.016651  | −0.000050  | 0.004914  | 0.004905  | +0.000009  |
| SOLUSDT | 0.018115  | 0.017825  | +0.000290  | 0.005994  | 0.005948  | +0.000046  |
| XRPUSDT | 0.016496  | 0.016463  | +0.000033  | 0.007667  | 0.007383  | +0.000284  |
| pooled  | 0.015073  | 0.014981  | +0.000092 (+0.61%) | 0.005309 | 0.005220 | +0.000088 (+1.69%) |

## Honest verdict

dip_seas is a legitimate mid-pack challenger: zero NaN rows, inside the
MCS @0.10 on both arenas, and decisively better than every published
foundation target. It does **not** beat the leading vol-modeling challengers —
pooled CRPS is +0.61% (d1) / +1.69% (h4f) worse than `dip_garch_t`, and it
also trails `dip_fhs` and `dip_regime` in both arenas. Calendar conditioning
of the empirical distribution adds no edge over the vol-clustering baselines
on this grid; wins on BTC/ETH d1 are real but small and don't generalize.
No promotion claim.
