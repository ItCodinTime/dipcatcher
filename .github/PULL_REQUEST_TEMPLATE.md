## Summary

<!-- What changed and why. Keep bullets terse; note the evidence class
     (research / backtest / simulated paper / SYNTHETIC) for results. -->

#### Test plan

- [ ] `make lint` clean (ruff check + format)
- [ ] `make typecheck` clean (`uv run mypy src/fx1` for fx1 changes)
- [ ] Tests pass (`make test` / `make fx1-test` as applicable)
- [ ] No secrets, credentials, or live-P&L claims in the diff
- [ ] Receipts/verifier docs updated if this lands a research claim

<!-- Honesty contract: research results are proper scores only — never
     Sharpe/Sortino/Calmar/P&L/NAV headlines. SYNTHETIC results stay
     labeled SYNTHETIC. See AGENTS.md. -->
