# Helper scripts

Operational scripts that aren't part of the installed packages. Prefer the
`fx1` / `dipcatcher` (`quant`) CLIs for day-to-day work — these are one-off
or bench utilities.

| Script | Purpose |
| --- | --- |
| `gmm_smoke.py` | GMM estimator smoke run |
| `merge_gmm.py` | Merge GMM score outputs |
| `new3_smoke.py` | NEW3 release smoke run |
| `qar_smoke.py` | QAR estimator smoke run |
| `run_gmm_pass.py` / `run_rest_pass.py` / `run_skt_pass.py` | Full bench passes |
| `skt_smoke.py` | SKT estimator smoke run |
| `secret_scan.py` | Pre-commit hook: rejects staged secrets |

Run any of them with `uv run python scripts/<name>.py` inside the synced env.
