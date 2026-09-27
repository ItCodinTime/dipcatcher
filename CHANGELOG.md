# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
The version source is `fx1.__version__`.

## [Unreleased]

### Changed

- README opens on dipcatcher: purpose, badges, architecture diagram, and a
  quick start whose default path is labeled synthetic research.
- fx-1 is documented as the in-tree package (corpus, eval, training
  manifests). No trained checkpoint ships in the repository.

### Fixed

- `quant_fund.__version__` follows `fx1.__version__` instead of a second
  literal (`1.0.0` against distribution `0.4.0`).

## [0.4.0] - 2026-09-26

Package version `0.4.0` was set on 2026-09-25 in `src/fx1/__init__.py`.
Entries below are recent conventional commits that landed on that version.

### Added

- Hugging Face OHLCV-1m US minute-bar source (`feat(data)`, #87).
- fx-1 package: corpus builder, eval bank, training-run manifests, model
  cards, SBOM, doctor, and `fx1 --version`.
- Dip Quality bench receipt over 11 crypto series
  (`receipts/dip_bench_crypto_1d_20260925.json`).
- Blocking honesty-inheritance test and corpus-contract smoke in the fx1
  workflow.
- `make fx1-gate` for local parity with that workflow.

### Changed

- Distribution version is read from `fx1.__version__`. Source distributions
  are limited to the buildable tree.
- Coverage floor raised from 70 to 80 (`fail_under` in `pyproject.toml`).
- `warn_return_any` is enforced on `src/quant_fund` and `src/fx1`.
- `make sync` installs all extras, matching CI.

### Fixed

- Worktree fingerprint no longer treats Git LFS pointer drift as a dirty
  tree on Python 3.13 CI (#88).
- Hosted-backend `urlopen` annotated for the Bandit gate.
