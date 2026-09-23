$note = @'

## SOTA eval fleet (sota_eval_kronos.py) — receipts landed 2026-09-21 ~23:00

A second evaluation fleet completed in `.dsh-24x7\eval-full\`: 11 daily assets x 300 origins + 5 4h assets (complete-case, `dip_student_t` excluded — its `scipy.stats.t.fit` fails on flat 4h windows) + Kronos seed-robustness runs (seeds 11/23). Published targets scored zero-shot: kronos_small (canonical `Kronos-Tokenizer-base` pairing via `d1fix_*`/`h4fix_*` rerun, spliced into `*.fixed.npz`), chronos2, bolt_small, timesfm. Challengers: 10 dip_* forecasters incl. garch_t, fhs, ewma_emp, lgbm_q, blend.

Merged receipts: `d1_merged.json` (3300 origins), `h4_merged.json` (1500 origins). All four targets are excluded from the MCS at alpha=0.10 at both horizons; MCS retains only `dip_garch_t` + `dip_fhs`. SPA p_lower/p_cons = 0.0005 per target; p_upper ~0.46-0.51 disclosed. This directly addresses the "SOTA blocked by missing apples-to-apples real-market evaluations" item — real Binance data, fixed causal protocol, proper scores, captured receipts + raw loss matrices (inference rerunnable via `--merge-parts` without models).

Ops note: `Start-Process` children spawned from a session-0 (WMI/schtasks) parent hang at 1 thread/0 CPU on this box; spawn workers via `Invoke-CimMethod Win32_Process Create` with a `cmd /c ... > log 2> err` wrapper (see `scripts\spawn_staggered.ps1`). ssh-session process trees die on session teardown.
'@
Add-Content -Path D:\dipcatcher\.dsh-24x7\PROGRESS.md -Value $note
Write-Output "appended"
