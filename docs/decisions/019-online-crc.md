# ADR-019: Online conformal risk control (CRC + ACI)

## Status

Accepted

## Date

2026-09-15

## Context

Batch CRC (ADR-012) chooses one expansion \(\lambda\) on a calibration window so expected 0-1 hit risk is at most \(\alpha\). `AdaptiveConformal` (ADR-008) is Gibbs–Candès ACI on residual *coverage*, not on a monotone tail loss. Live VaR / drawdown bounds need the CRC threshold to move when the loss process shifts, without stacking a cross-section into a fake time series. Neural nets stay blocked (ADR-007).

## Options considered

- Recompute batch CRC each day on a growing window — exchangeable, but looks ahead across dates and is not sequential.
- ACI on \(\alpha_t\) wrapping CQR — valid for interval miscoverage, silent on \(\mathbb{E}[L(\lambda)]\).
- Gibbs–Candès-style integrator on the CRC expansion (adaptive risk control / dual ACI): \(\lambda_{t+1}=\max(0,\,\lambda_t+\gamma(L_t-\alpha))\).

## Decision

`OnlineCRC` lives in `quant_fund.models.online_crc` and does not edit `crc.py` or `conformal.py`. It initializes \(\lambda\) with batch CRC, then updates **once per timestamp**. Names that share a date contribute their mean hit \(L_t=1\{\mathrm{loss}>\mathrm{base}+\lambda_t\}\) and do not each step \(\lambda\). After a miss, \(\lambda\) rises and the bound expands; long-run mean hit risk tracks \(\alpha\). Lab scores are `mean_risk`, nominal \(\alpha\), and \(n\). No Sharpe.

## Consequences

Expected hit-risk can follow \(\alpha\) under non-exchangeable losses, which batch CRC cannot. The wrapper still only expands a base bound; it does not repair a misspecified Gaussian. PIT / Kupiec of the raw bound stay specification diagnostics. Promotion still requires those raw scores to remain visible.
