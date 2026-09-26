# Security

## Reporting a vulnerability

Do not open a public issue for security problems. Email the maintainer
directly (see the commit author address) or use GitHub private vulnerability
reporting on this repository.

## Operational policy

- Never commit secrets. The env-var surface is enumerated in `.env.example`;
  a staged-diff `secret-scan` pre-commit hook is enforced.
- `MOONSHOT_API_KEY` / `FX1_SIGNING_KEY` / `QUANT_API_KEY` /
  `QUANT_VENDOR_API_KEY` stay in the environment or the secret store —
  tooling (`fx1 doctor`, `dipcatcher doctor`) reports presence flags only.
- Dependencies are locked via `uv.lock` (`uv sync --frozen`) and patched
  through Dependabot; CI runs `bandit` and `pip-audit` on every change.
