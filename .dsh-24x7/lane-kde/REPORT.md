# dip_kde — Gaussian KDE challenger (Silverman × EWMA-vol-scaled bandwidth)

Date: 2026-09-23 (UTC). Lane: `.dsh-24x7/lane-kde/`. Tool: `scripts/_kde_col.py` (new file only).

## Design (causal, deterministic, no RNG)

Per origin, over the trailing `garch_window` (750) close-to-close returns:

- Kernel: Gaussian. CDF `F(x) = mean_i Φ((x − r_i)/h)` evaluated analytically.
- Bandwidth: `h = 0.9 · min(s, IQR/1.34) · n^(−1/5) · clip(v_recent/v_long, 0.7, 1.4)`,
  where `v_recent` = `ewma_next_sigma` over the last 60 returns, `v_long` = same
  EWMA recursion over the full window. Adaptive smoothing widens h when recent
  vol exceeds long-run vol.
- Quantiles: numerical inversion of the KDE CDF by linear interpolation on a
  fixed 4097-point grid spanning `[min(r) − 4h, max(r) + 4h]`; evaluated at
  `LGBM_TAUS` (∋ `TAUS`) and at 512 equiprobable midpoint levels.
- CRPS: `crps_empirical(y, q_dense)` — the 512 equiprobable KDE quantiles
  treated as an ensemble sample (same convention as dip_skt / dip_conf_t /
  dip_regime). Chosen over `crps_from_quantiles` so the full smoothed shape
  enters the score rather than a coarse-grid trapezoid.
- Edge cases: `n < 30` or `h <= 0` → honest NaN (never triggered: 300/300
  finite CRPS and pinball on every shard).

## Execution

- All 10 shards computed on remote `me@100.116.120.51` (`D:\dipcatcher`,
  `.venv\Scripts\python.exe`) via `run_kde.ps1`; ~72–82 s per shard.
- All 10 shard SHA256 verified identical remote vs local before compute.
- Columns spliced onto shard copies in `lane-kde/spliced/` via
  `splice_challenger_column.py` (dip_kde already in KNOWN_MODELS; all 10 → 21 cols).
- Merged with `sota_eval_kronos.py --merge-parts … --bars-root data/raw/sources`
  (timestamps reconstructed from hash-verified bars; seed=7, n_boot=1000,
  matching the reference merges).

## Results — pooled CRPS (1500 rows/panel, all complete)

### d1 (5 × 1d deep)

| rank | model | mean CRPS |
|---:|---|---:|
| 1 | dip_fhs | 0.014945 |
| 2 | dip_garch_t | 0.014981 |
| 7 | dip_empirical | 0.015099 |
| **13** | **dip_kde** | **0.015191** |

Pinball dip_kde: τ0.05=0.003261, τ0.5=0.010008, τ0.95=0.003845.
**MCS @0.10: OUT** (p = 0.00999).

### h4f (5 × 4h deep)

| rank | model | mean CRPS |
|---:|---|---:|
| 1 | dip_fhs | 0.005218 |
| 2 | dip_garch_t | 0.005220 |
| 10 | dip_empirical | 0.005325 |
| **12** | **dip_kde** | **0.005327** |

Pinball dip_kde: τ0.05=0.001083, τ0.5=0.003438, τ0.95=0.001653.
**MCS @0.10: IN** (p = 0.2917).

## Key comparisons (pooled CRPS diff, dip_kde − opponent; + = worse)

| opponent | d1 diff | d1 winrate | h4f diff | h4f winrate |
|---|---:|---:|---:|---:|
| dip_empirical | +0.000092 | 0.368 | +0.000002 | 0.368 |
| dip_garch_t | +0.000210 | 0.357 | +0.000106 | 0.387 |
| dip_fhs | +0.000246 | 0.361 | +0.000108 | 0.396 |
| dip_ewma_emp | −0.000001 | 0.411 | +0.000020 | 0.449 |

Per-asset d1 diff vs dip_empirical: BNB +1.9e-05, BTC −2.8e-05, ETH +4.2e-05,
SOL +2.58e-04, XRP +1.67e-04 — the KDE edge on BTC is swamped by SOL/XRP losses.
NaN counts: 0 CRPS, 0 pinball on both panels.

Bandwidth stats (h_min/mean/max, vol-ratio mean) recorded per shard in each
column's `meta_json["kde"]`; e.g. BTC d1 h ≈ 0.0042–0.0062 (mean 0.0044),
vol-ratio ≈ 1.00 — the clip bounds rarely bind on this grid.

## Verdict

**Rejected as a SOTA challenger.** dip_kde is mid-pack: rank 13/21 on d1
(MCS-eliminated, p≈0.01) and rank 12/21 on h4f (MCS-included but +2e-06 CRPS
vs dip_empirical — a wash). It does not beat dip_empirical pooled on either
panel; the Gaussian smoothing loses the fat-tail mass the raw ECDF keeps,
especially on SOL/XRP daily. Silverman bandwidth ≈ 0.9σ·n^(−1/5) is close to
optimal for a Gaussian truth, but crypto returns are heavy-tailed — the KDE
tails remain Gaussian-thin around the data range. Remains a valid,
hash-bound, deterministic nonparametric-smooth entry in the arena (beats
timesfm/chronos2/bolt/kronos and dip_gauss/dip_lgbm_q/dip_ewma_t on both
panels), but it adds nothing over dip_empirical.

## Artifacts

- `scripts/_kde_col.py` — column generator (new file; no existing files edited)
- `.dsh-24x7/lane-kde/{shard}.kde.npz` — 10 column artifacts (bars_sha256-bound)
- `.dsh-24x7/lane-kde/spliced/{shard}.v2aug.npz` — 10 spliced shards (21 cols)
- `.dsh-24x7/lane-kde/merge_d1.json` + `.losses.npz` — pooled daily receipt
- `.dsh-24x7/lane-kde/merge_h4f.json` + `.losses.npz` — pooled 4h receipt
- Remote copies remain at `D:\dipcatcher\.dsh-24x7\lane-kde\` and `D:\dipcatcher\run_kde.ps1`
