# ADR-004: In-house DCC(1,1) on arch univariates

## Status

Accepted

## Date

2026-09-15

## Context

`arch` is univariate. `pymgarch` is 0.1.x. R `rmgarch` is the academic reference but is not a Python dependency we want.

## Decision

Implement Gaussian DCC(1,1) two-stage QML in `quant_fund.models.covariance.dcc`, wrapping `arch` for stage 1. Test \(a,b>0\), \(a+b<1\), PSD of \(R_t\), and recovery on a synthetic CCC/DCC process.

## Consequences

Gaussian DCC(1,1) two-stage QML is in-house (ADR-004). Student-t DCC is a catalog estimator (Wave 126) and a named optimizer path (Wave 127). Cappiello–Engle–Sheppard scalar ADCC is a catalog estimator (Wave 128) and a named optimizer path (Wave 129). This is not a full rmgarch clone.

Wave 116 implements stage 1 with `arch` / `GARCHVol` Gaussian GARCH(1,1). The prior EWMA residual path is not retained as a silent fallback.

Wave 122 exposes named `dcc_student_t` and `adcc` entry points that fail closed (`unspecified_dcc_spec:*`) instead of silently running `dcc_gaussian`. `/models` lists `dcc_gaussian` as implemented and `dcc_student_t` / `adcc` as unspecified.

Wave 123 returns Engle (2002) one-step-ahead \(H_{t+1}\) (`covariance_object=one_step_ahead`) rather than in-sample last \(H_t\). Last \(z_t\) enters \(Q_{t+1}\); \(D_{t+1}\) is the univariate GARCH one-step sigma.

Wave 124 wires `dcc_gaussian` into `optimize_asof` / `/risk/portfolio` as a named `optimizer.covariance` path. Default remains trailing Ledoit–Wolf plus the GARCH/RGARCH overlay. The DCC path does not overlay \(H_{t+1}\). Student-t DCC / ADCC remain unspecified.

Wave 125 estimates Gaussian DCC on the trailing contiguous complete-case window (`sample=trailing_complete_window`). Incomplete terminal rows fail closed rather than dropping asof \(z_t\). Holes are not concatenated into the sequential QML sample. Trailing return pivots are sorted by `event_time`.

Wave 126 implements Student-t DCC as a catalog estimator: univariate Student-t GARCH stage 1 and a covariance Student-t stage-2 likelihood (`student_t_corr_nll`, \(ν>2\)). It shares the Engle one-step \(H_{t+1}\) and trailing complete-window contract. Cappiello–Engle–Sheppard ADCC remains unspecified.

Wave 127 wires `dcc_student_t` into `optimize_asof` / `/risk/portfolio` as a named `optimizer.covariance` path. Default remains trailing Ledoit–Wolf plus the GARCH/RGARCH overlay. The Student-t path does not overlay \(H_{t+1}\) and must not silently size as Gaussian DCC or Ledoit–Wolf. ADCC remained unspecified until Wave 128.

Wave 128 implements scalar Cappiello–Engle–Sheppard ADCC as a catalog estimator: univariate Gaussian GARCH stage 1 and CES (2006) \(Q_t=(1-a-b)\bar Q-g\bar N+a z_{t-1}z_{t-1}^\top+b Q_{t-1}+g n_{t-1}n_{t-1}^\top\) with \(n_t=I[z_t<0]\odot z_t\) and \(a+b+\kappa g<1\). It shares the one-step \(H_{t+1}\) and trailing complete-window contract. `optimize_asof` stayed unwired in that wave so ADCC could not silently size as Gaussian DCC, Student-t DCC, or Ledoit–Wolf. This is not matrix AG-DCC.

Wave 129 wires `adcc` into `optimize_asof` / `/risk/portfolio` as a named `optimizer.covariance` path. Default remains trailing Ledoit–Wolf plus the GARCH/RGARCH overlay. The ADCC path does not overlay \(H_{t+1}\) and must not silently size as Gaussian DCC, Student-t DCC, or Ledoit–Wolf. This is still scalar CES ADCC, not matrix AG-DCC.

Wave 133 implements Bollerslev (1990) CCC as a catalog estimator: univariate Gaussian GARCH stage 1 and constant \(R=\mathrm{corr}(z)\) with one-step \(H_{t+1}=D_{t+1} R D_{t+1}\) on the trailing complete window. There is no \(Q\) recursion and no \(a,b\) QML. `optimize_asof` stayed unwired in that wave so CCC could not silently size as Gaussian DCC, Student-t DCC, scalar ADCC, or Ledoit–Wolf. This is not matrix AG-DCC.

Wave 134 wires `ccc` into `optimize_asof` / `/risk/portfolio` as a named `optimizer.covariance` path. Default remains trailing Ledoit–Wolf plus the GARCH/RGARCH overlay. The CCC path does not overlay \(H_{t+1}\) and must not silently size as Gaussian DCC, Student-t DCC, scalar ADCC, or Ledoit–Wolf. This is still constant conditional correlation, not matrix AG-DCC.

Wave 135 implements diagonal Cappiello–Engle–Sheppard AG-DCC as a catalog estimator: univariate Gaussian GARCH stage 1 and CES (2006) \(Q_t=(\bar Q-A\bar Q A-B\bar Q B-G\bar N G)+A z_{t-1}z_{t-1}^\top A+B Q_{t-1}B+G n_{t-1}n_{t-1}^\top G\) with diagonal \(A,B,G\). Equal diagonals recover scalar CES ADCC. It shares the one-step \(H_{t+1}\) and trailing complete-window contract. `optimize_asof` stayed unwired in that wave so AG-DCC could not silently size as scalar ADCC, Gaussian DCC, CCC, or Ledoit–Wolf. Unrestricted full-matrix AG-DCC stays unspecified.

Wave 136 wires `agdcc` into `optimize_asof` / `/risk/portfolio` as a named `optimizer.covariance` path. Default remains trailing Ledoit–Wolf plus the GARCH/RGARCH overlay. The AG-DCC path does not overlay \(H_{t+1}\) and must not silently size as scalar ADCC, Gaussian DCC, CCC, or Ledoit–Wolf. This is still diagonal CES AG-DCC, not unrestricted full-matrix AG-DCC.

Wave 137 implements unrestricted Cappiello–Engle–Sheppard AG-DCC as a catalog estimator: univariate Gaussian GARCH stage 1 and CES (2006) \(Q_t=(\bar Q-A\bar Q A^\top-B\bar Q B^\top-G\bar N G^\top)+A z_{t-1}z_{t-1}^\top A^\top+B Q_{t-1}B^\top+G n_{t-1}n_{t-1}^\top G^\top\) with unrestricted \(A,B,G\). Diagonal matrices recover Wave 135. It shares the one-step \(H_{t+1}\) and trailing complete-window contract. `optimize_asof` stayed unwired in that wave so unrestricted AG-DCC could not silently size as diagonal AG-DCC, scalar ADCC, Gaussian DCC, CCC, or Ledoit–Wolf.

Wave 138 wires `agdcc_full` into `optimize_asof` / `/risk/portfolio` as a named `optimizer.covariance` path. Default remains trailing Ledoit–Wolf plus the GARCH/RGARCH overlay. The unrestricted AG-DCC path does not overlay \(H_{t+1}\) and must not silently size as diagonal AG-DCC, scalar ADCC, Gaussian DCC, CCC, or Ledoit–Wolf. Factor stays unwired.
