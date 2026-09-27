# Platform matrix

`.github/workflows/matrix.yml` runs the lab test suite
(`tests/{unit,property,regression,end_to_end}`, `-m "not network"`) across:

| OS | Python | Status |
|---|---|---|
| ubuntu-latest | 3.12, 3.13, 3.14 | matrix |
| macos-latest (arm64) | 3.12, 3.13, 3.14 | matrix |
| windows-latest | 3.12, 3.13, 3.14 | matrix |

The workflow is additive: it never feeds `ci.yml` required checks, uses
`fail-fast: false`, and shards each combo into 4 jobs (~1.5k tests each) by
greedy-balancing the `pytest --collect-only` per-file test counts.

## Supported band

`pyproject.toml` declares `requires-python = ">=3.12"`:

- **3.11 is out of band.** The project cannot even `uv sync --frozen` on 3.11
  (resolver rejects the interpreter), and the codebase is free to use 3.12
  syntax (`type` aliases, generic parameter lists). Widening to 3.11 would
  require a `target-version`/`python_version` downgrade plus a syntax audit —
  out of scope for this lane.
- **3.14 is in band** (`>=3.12` allows it) and covered by the matrix. Locked
  wheels for cp314 (numpy 2.5.x, pandas 3.x, torch 2.14, polars, pyarrow, …)
  are verified by CI rather than assumed; see the PR for the evidence run.

## Reproducing locally

```bash
# Whole suite, same selection as CI:
uv run pytest -q -m "not network"

# One shard locally (macOS/Linux; POSIX shell):
uv run pytest --collect-only -q -m "not network" \
  | grep -E '^tests/.*: [0-9]+$' | LC_ALL=C sort -t: -k2 -nr > /tmp/inv.txt
awk -F': ' -v k=4 -v want=0 '{min=0;for(i=1;i<k;i++)if(sum[i]<sum[min])min=i;sum[min]+=$NF;if(min==want)print $1}' \
  /tmp/inv.txt > /tmp/shard.txt
uv run pytest -q -m "not network" $(tr '\n' ' ' < /tmp/shard.txt)
```

## Known platform quirks found

- **The locked LightGBM macOS wheel needs OpenMP at runtime.** The matrix
  installs Homebrew `libomp` before collection and checks its dylib exists.
- **Git Bash passes Windows runner paths to AWK with backslashes.** AWK
  prints selected paths to stdout; the shell redirects them to the shard
  file so backslashes are not parsed as AWK string escapes.
- **Native C parity on Windows requires GCC.** The matrix checks the compiler
  and adds its directory to `PATH`. The [GitHub-hosted Windows image](https://github.com/actions/runner-images/blob/main/images/windows/Windows2025-Readme.md)
  lists GCC and MSYS2, while noting that MSYS2 is not on `PATH`; the check
  fails clearly if the image changes.
- **`chmod(0o000)` does not make a file unreadable on Windows.** Windows
  permissions are ACL-based; `os.chmod`/`Path.chmod` can only toggle the
  read-only attribute, and a read-only file still reads fine. Two
  fail-closed tests that simulate an unreadable research receipt are skipped
  on Windows (`tests/unit/pipeline/test_api_cov.py`,
  `test_drift_unreadable_receipt_reports_none_source`,
  `test_research_latest_unreadable_receipt_is_422`).
- **Locale encoding.** Windows runners default to cp1252; the repo's text
  artifacts are UTF-8. The workflow sets `PYTHONUTF8=1`/`PYTHONIOENCODING=utf-8`
  (PEP 540) so Windows interpreters behave like the POSIX runners. PEP 686
  makes UTF-8 the default in CPython 3.15, so this pin ages out naturally.
- **`/tmp/*` string literals** in a few microstructure tests are dict values
  for honesty-validation helpers, never touched on disk — portable.
- Atomic writes in `src/quant_fund` already use
  `NamedTemporaryFile(delete=False)` + `os.replace`, which is the correct
  Windows-safe pattern (no reopen-by-name while open, replace over existing).
- `git` subprocess usage (`utils/reproducibility.py`, reproducibility tests)
  works on Windows runners. `.gitattributes` forces LF checkout for tracked
  text while retaining the LFS binary rules, so sealed config byte hashes
  match the Git blobs on every runner.

## Out of scope / risks seen

- `tests/fx1` stays on its own `fx1.yml` lane (ubuntu, py3.12) — not in the
  default `testpaths` and not matrixed here.
- Live-trading/broker paths are untouched by design (hard rule); nothing in
  the matrix exercises them.
- Numerical parity across BLAS vendors is asserted by the tests themselves;
  platform-specific tolerance loosening would be a bug report, not a patch.
