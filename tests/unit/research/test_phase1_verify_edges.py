"""research/phase1_verify edge paths: receipt-rejection branches across the
benchmark manifest/report and tournament manifest/phase verifiers, index
guards, and the low-level JSON/hash helpers."""

from __future__ import annotations

import gzip
import hashlib
import json
import shutil
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from quant_fund.research.catalog import BENCHMARK_CATALOG_VERSION
from quant_fund.research.net_tournament import prepare_tournament, run_tournament
from quant_fund.research.phase1_verify import (
    verify_phase1_index,
    verify_phase1_run,
)
from quant_fund.research.real_benchmark import (
    _seal,
    prepare_benchmark,
    score_benchmark,
)
from quant_fund.utils.reproducibility import git_revision, git_worktree_sha256

pytestmark = pytest.mark.synthetic


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False))


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _reseal(path: Path, **changes) -> None:
    payload = _load(path)
    payload.pop("receipt_sha256", None)
    payload.update(changes)
    _write(path, _seal(payload))


def _reseal_gz(path: Path, **changes) -> None:
    with gzip.open(path, "rt") as fh:
        payload = json.load(fh)
    payload.pop("receipt_sha256", None)
    payload.update(changes)
    with gzip.open(path, "wt") as fh:
        json.dump(_seal(payload), fh)


@contextmanager
def _mutated(path: Path, **changes):
    """Reseal `path` with `changes`, yield, then restore the original file."""
    original = path.read_bytes()
    _reseal(path, **changes)
    try:
        yield
    finally:
        path.write_bytes(original)


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("phase1_edges")
    dates = [datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=n) for n in range(240)]
    rng = np.random.default_rng(73)
    frames = []
    for name in ("A", "B", "C", "D"):
        prices = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.015, len(dates))))
        frames.append(
            pl.DataFrame(
                {
                    "security_id": [name] * len(dates),
                    "event_time": dates,
                    "available_time": dates,
                    "ingested_time": dates,
                    "source": ["test_contract"] * len(dates),
                    "close": prices,
                    "open": prices * np.exp(rng.normal(0, 0.003, len(dates))),
                    "volume": [1_000_000] * len(dates),
                }
            )
        )
    data = root / "bars.parquet"
    pl.concat(frames).write_parquet(data)
    protocol = {
        "dataset_path": "bars.parquet",
        "dataset_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
        "source_url": "https://example.com/test-contract",
        "usage_basis": "generated contract test",
        "price_column": "close",
        "price_adjustment": "unadjusted generated fixture",
        "universe_description": "four generated securities",
        "survivorship_bias": True,
        "availability_basis": "reconstructed",
        "holdout_previously_inspected": True,
        "train_start": "2020-01-01",
        "train_end": "2020-03-31",
        "validation_start": "2020-04-05",
        "validation_end": "2020-05-31",
        "test_start": "2020-06-05",
        "test_end": "2020-08-20",
        "min_train_rows": 100,
        "min_score_dates": 30,
    }
    config = root / "benchmark.json"
    _write(config, protocol)
    benchmark = root / "benchmark"
    prepare_benchmark(config, benchmark)
    score_benchmark(benchmark, "validation")
    score_benchmark(benchmark, "test")
    spec = {
        "execution": {},
        "trials": [
            {"name": "mom", "family": "momentum"},
            {"name": "rev", "family": "reversal", "lookback": 1},
        ],
        "benchmark": {"name": "equal", "family": "equal_weight"},
        "open_column": "open",
        "volume_column": "volume",
        "price_basis": "raw_price_return",
        "n_boot": 99,
        "block_sessions": 5,
        "seed": 17,
    }
    spec_path = root / "slate.json"
    _write(spec_path, spec)
    tournament = root / "tournament"
    prepare_tournament(benchmark, spec_path, tournament)
    run_tournament(tournament, "validation")
    run_tournament(tournament, "test")
    index_dir = root / "research"
    index_dir.mkdir()
    runs_meta = []
    for kind, name, source in (
        ("real_benchmark", "benchmark", config),
        ("net_tournament", "tournament", spec_path),
    ):
        directory = root / name
        runs_meta.append(
            {
                "kind": kind,
                "path": f"../{name}",
                "config_path": f"../{source.name}",
                "config_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                **{
                    f"{phase}_sha256": _load(directory / f"{phase}.json")["receipt_sha256"]
                    for phase in ("manifest", "validation", "test")
                },
            }
        )
    index = _seal(
        {
            "kind": "phase1_evidence_index",
            "schema_version": 1,
            "created_at": datetime.now(UTC).isoformat(),
            "git_revision": git_revision(),
            "git_worktree_sha256": git_worktree_sha256(),
            "benchmark_catalog_version": BENCHMARK_CATALOG_VERSION,
            "runs": runs_meta,
        }
    )
    _write(index_dir / "index.json", index)
    return root


@pytest.fixture
def fresh(runs, tmp_path):
    dest = tmp_path / "copy"
    shutil.copytree(runs, dest)
    return dest


def _errors(result: dict) -> list[str]:
    return result["errors"]


class TestLowLevel:
    def test_duplicate_keys_rejected(self, tmp_path) -> None:
        p = tmp_path / "bad.json"
        p.write_text('{"a": 1, "a": 2}')
        from quant_fund.research.phase1_verify import _receipt

        errors: list[str] = []
        assert _receipt(p, errors) is None
        assert errors

    def test_nonfinite_token_rejected(self) -> None:
        from quant_fund.research.phase1_verify import _reject_nonfinite

        with pytest.raises(ValueError):
            _reject_nonfinite("NaN")

    def test_timestamp_and_sha_helpers(self) -> None:
        from quant_fund.research.phase1_verify import _sha256, _timestamp

        assert _timestamp("2020-01-01T00:00:00+00:00") is True
        assert _timestamp("not-a-time") is False
        assert _timestamp(123) is False
        assert _sha256("a" * 64) is True
        assert _sha256("zz") is False

    def test_manifest_not_object(self, fresh) -> None:
        (fresh / "benchmark/manifest.json").write_text("[1]")
        assert _errors(verify_phase1_run(fresh / "benchmark"))


class TestBenchmarkManifest:
    def test_schema_and_timestamp(self, fresh) -> None:
        path = fresh / "benchmark/manifest.json"
        with _mutated(path, schema_version=2):
            assert any(
                "benchmark: unsupported schema" in e
                for e in _errors(verify_phase1_run(fresh / "benchmark"))
            )
        with _mutated(path, created_at="x"):
            assert any(
                "benchmark: invalid timestamp" in e
                for e in _errors(verify_phase1_run(fresh / "benchmark"))
            )

    def test_code_sha_mismatch(self, fresh) -> None:
        with _mutated(fresh / "benchmark/manifest.json", code_sha256="0" * 64):
            assert any(
                "code SHA-256 differs" in e for e in _errors(verify_phase1_run(fresh / "benchmark"))
            )

    def test_runtime_mismatch(self, fresh) -> None:
        with _mutated(fresh / "benchmark/manifest.json", runtime={"python": "0.0"}):
            assert any(
                "runtime differs" in e for e in _errors(verify_phase1_run(fresh / "benchmark"))
            )

    def test_protocol_not_object(self, fresh) -> None:
        with _mutated(fresh / "benchmark/manifest.json", protocol="x"):
            assert any(
                "invalid protocol" in e for e in _errors(verify_phase1_run(fresh / "benchmark"))
            )

    def test_holdout_disclosure_mismatch(self, fresh) -> None:
        with _mutated(
            fresh / "benchmark/manifest.json",
            holdout_status="uninspected_by_declaration",
        ):
            assert any(
                "holdout disclosure" in e for e in _errors(verify_phase1_run(fresh / "benchmark"))
            )

    def test_limitations_missing(self, fresh) -> None:
        with _mutated(fresh / "benchmark/manifest.json", limitations=None):
            assert any(
                "limitations missing" in e for e in _errors(verify_phase1_run(fresh / "benchmark"))
            )


class TestBenchmarkReport:
    def test_phase_and_link_guards(self, fresh) -> None:
        path = fresh / "benchmark/validation.json"
        with _mutated(path, phase="test"):
            assert any(
                "phase mismatch" in e for e in _errors(verify_phase1_run(fresh / "benchmark"))
            )
        with _mutated(path, manifest_sha256="0" * 64):
            assert any(
                "manifest link mismatch" in e
                for e in _errors(verify_phase1_run(fresh / "benchmark"))
            )

    def test_claim_and_field_guards(self, fresh) -> None:
        path = fresh / "benchmark/validation.json"
        with _mutated(path, claim="Sharpe ratio leaderboard"):
            assert any(
                "invalid research claim" in e
                for e in _errors(verify_phase1_run(fresh / "benchmark"))
            )
        with _mutated(path, holdout_status="x"):
            assert any(
                "differs from manifest" in e
                for e in _errors(verify_phase1_run(fresh / "benchmark"))
            )


class TestTournamentManifest:
    def test_schema_timestamp(self, fresh) -> None:
        with _mutated(fresh / "tournament/manifest.json", schema_version=2):
            assert any(
                "tournament: unsupported schema" in e
                for e in _errors(verify_phase1_run(fresh / "tournament"))
            )

    def test_embedded_benchmark_mismatch(self, fresh) -> None:
        m = _load(fresh / "tournament/manifest.json")
        parent = dict(m["benchmark_manifest"])
        parent["holdout_status"] = "tampered"
        with _mutated(fresh / "tournament/manifest.json", benchmark_manifest=parent):
            assert any(
                "embedded benchmark manifest differs" in e
                for e in _errors(verify_phase1_run(fresh / "tournament"))
            )

    def test_spec_and_slate_guards(self, fresh) -> None:
        path = fresh / "tournament/manifest.json"
        with _mutated(path, spec="x"):
            assert any(
                "invalid specification" in e
                for e in _errors(verify_phase1_run(fresh / "tournament"))
            )
        with _mutated(path, candidate_count=99):
            assert any(
                "candidate_count mismatch" in e
                for e in _errors(verify_phase1_run(fresh / "tournament"))
            )
        bad = list(_load(path)["candidates"])
        bad[0] = {**bad[0], "name": "tampered"}
        with _mutated(path, candidates=bad):
            assert any(
                "candidate slate mismatch" in e
                for e in _errors(verify_phase1_run(fresh / "tournament"))
            )
        with _mutated(path, benchmark={"name": "x"}):
            assert any(
                "baseline mismatch" in e for e in _errors(verify_phase1_run(fresh / "tournament"))
            )

    def test_rule_scenarios_limitations(self, fresh) -> None:
        path = fresh / "tournament/manifest.json"
        with _mutated(path, selection_rule="x"):
            assert any(
                "selection rule mismatch" in e
                for e in _errors(verify_phase1_run(fresh / "tournament"))
            )
        with _mutated(path, scenarios={}):
            assert any(
                "impact scenarios mismatch" in e
                for e in _errors(verify_phase1_run(fresh / "tournament"))
            )
        with _mutated(path, limitations=["x", ""]):
            assert any(
                "limitations missing" in e for e in _errors(verify_phase1_run(fresh / "tournament"))
            )


class TestTournamentPhase:
    def test_attempt_guards(self, fresh) -> None:
        path = fresh / "tournament/validation.attempt.json"
        with _mutated(path, phase="test"):
            assert any(
                "attempt phase mismatch" in e
                for e in _errors(verify_phase1_run(fresh / "tournament"))
            )
        with _mutated(path, manifest_sha256="0" * 64):
            assert any(
                "attempt manifest link mismatch" in e
                for e in _errors(verify_phase1_run(fresh / "tournament"))
            )
        with _mutated(path, candidates=["x"]):
            assert any(
                "attempt candidate slate mismatch" in e
                for e in _errors(verify_phase1_run(fresh / "tournament"))
            )

    def test_report_guards(self, fresh) -> None:
        path = fresh / "tournament/validation.json"
        with _mutated(path, phase="test"):
            assert any(
                "phase mismatch" in e for e in _errors(verify_phase1_run(fresh / "tournament"))
            )
        with _mutated(path, claim="x"):
            assert any(
                "invalid research claim" in e
                for e in _errors(verify_phase1_run(fresh / "tournament"))
            )

    def test_selected_outside_slate(self, fresh) -> None:
        with _mutated(fresh / "tournament/validation.json", selected="ghost"):
            assert any(
                "outside frozen slate" in e
                for e in _errors(verify_phase1_run(fresh / "tournament"))
            )

    def test_scenarios_and_trials(self, fresh) -> None:
        path = fresh / "tournament/validation.json"
        with _mutated(path, scenarios={"configured": {}}):
            assert any(
                "scenarios missing or changed" in e
                for e in _errors(verify_phase1_run(fresh / "tournament"))
            )
        report = _load(path)
        sc = dict(report["scenarios"])
        sc["configured"] = {**sc["configured"], "trials": {}}
        with _mutated(path, scenarios=sc):
            assert any(
                "trial slate incomplete" in e
                for e in _errors(verify_phase1_run(fresh / "tournament"))
            )

    def test_trial_status_and_ledgers(self, fresh) -> None:
        path = fresh / "tournament/validation.json"
        report = _load(path)
        sc = dict(report["scenarios"])
        trials = dict(sc["configured"]["trials"])
        first = next(iter(trials))
        trials[first] = {"status": "unknown"}
        sc["configured"] = {**sc["configured"], "trials": trials}
        with _mutated(path, scenarios=sc):
            assert any(
                "invalid trial status" in e
                for e in _errors(verify_phase1_run(fresh / "tournament"))
            )

        report = _load(path)
        sc = dict(report["scenarios"])
        trials = dict(sc["configured"]["trials"])
        first = next(iter(trials))
        bad = dict(trials[first])
        bad.pop("daily", None)
        trials[first] = bad
        sc["configured"] = {**sc["configured"], "trials": trials}
        with _mutated(path, scenarios=sc):
            assert any(
                "ledger missing" in e for e in _errors(verify_phase1_run(fresh / "tournament"))
            )

    def test_comparison_and_flags(self, fresh) -> None:
        path = fresh / "tournament/validation.json"
        report = _load(path)
        sc = dict(report["scenarios"])
        sc["configured"] = {**sc["configured"]}
        sc["configured"].pop("comparison", None)
        with _mutated(path, scenarios=sc):
            assert any(
                "comparison missing" in e for e in _errors(verify_phase1_run(fresh / "tournament"))
            )
        report = _load(path)
        sc = dict(report["scenarios"])
        comp = dict(sc["configured"]["comparison"])
        comp["tested_candidates"] = []
        sc["configured"] = {**sc["configured"], "comparison": comp}
        with _mutated(path, scenarios=sc):
            assert any(
                "inference slate mismatch" in e
                for e in _errors(verify_phase1_run(fresh / "tournament"))
            )
        with _mutated(path, complete=False):
            assert any(
                "completeness flag mismatch" in e
                for e in _errors(verify_phase1_run(fresh / "tournament"))
            )
        with _mutated(path, selected_holdout_adjusted_rejection="yes"):
            assert any(
                "adjusted rejection flag" in e
                for e in _errors(verify_phase1_run(fresh / "tournament"))
            )
        gate = not _load(path)["economic_evidence_gate"]
        with _mutated(path, economic_evidence_gate=gate):
            assert any(
                "economic evidence gate" in e
                for e in _errors(verify_phase1_run(fresh / "tournament"))
            )


class TestIndex:
    def test_index_kind_schema_timestamp(self, fresh) -> None:
        path = fresh / "research/index.json"
        with _mutated(path, kind="x"):
            assert any("kind mismatch" in e for e in _errors(verify_phase1_index(path)))
        with _mutated(path, schema_version=2):
            assert any("unsupported schema" in e for e in _errors(verify_phase1_index(path)))
        with _mutated(path, created_at="x"):
            assert any("invalid timestamp" in e for e in _errors(verify_phase1_index(path)))
        with _mutated(path, git_worktree_sha256="x"):
            assert any("git_worktree" in e for e in _errors(verify_phase1_index(path)))

    def test_duplicate_runs_and_kind(self, fresh) -> None:
        path = fresh / "research/index.json"
        index = _load(path)
        index.pop("receipt_sha256")
        index["runs"] = [index["runs"][0], index["runs"][0]]
        _write(path, _seal(index))
        assert any("duplicate run" in e for e in _errors(verify_phase1_index(path)))
        index = _load(path)
        index.pop("receipt_sha256")
        index["runs"][0]["path"] = "../tournament"  # real_benchmark slot → tournament dir
        index["runs"][0]["config_sha256"] = index["runs"][1]["config_sha256"]
        index["runs"][0]["config_path"] = index["runs"][1]["config_path"]
        _write(path, _seal(index))
        assert any("kind mismatch" in e for e in _errors(verify_phase1_index(path)))

    def test_missing_index_file(self, tmp_path) -> None:
        out = verify_phase1_index(tmp_path / "none.json")
        assert out["valid"] is False

    _pristine: dict[str, bytes] = {}

    def _rewrite_index(self, path: Path, mutate) -> None:
        key = str(path)
        self._pristine.setdefault(key, path.read_bytes())
        index = json.loads(self._pristine[key])
        index.pop("receipt_sha256", None)
        mutate(index)
        _write(path, _seal(index))

    def test_entry_object_and_kind_guards(self, fresh) -> None:
        path = fresh / "research/index.json"
        self._rewrite_index(path, lambda i: i["runs"].append("x"))
        assert any("entry is not an object" in e for e in _errors(verify_phase1_index(path)))
        self._rewrite_index(path, lambda i: i["runs"].append({"kind": "weird"}))
        assert any("unknown run kind" in e for e in _errors(verify_phase1_index(path)))

    def test_absolute_paths_rejected(self, fresh) -> None:
        path = fresh / "research/index.json"
        self._rewrite_index(
            path,
            lambda i: i["runs"][0].update(path="/abs/run"),
        )
        assert any("run path must be relative" in e for e in _errors(verify_phase1_index(path)))
        self._rewrite_index(
            path,
            lambda i: i["runs"][0].update(config_path="/abs/cfg"),
        )
        assert any("config path must be relative" in e for e in _errors(verify_phase1_index(path)))

    def test_receipt_fields_missing(self, fresh) -> None:
        path = fresh / "research/index.json"
        self._rewrite_index(path, lambda i: i["runs"][0].pop("manifest_sha256", None))
        assert any("manifest_sha256 missing" in e for e in _errors(verify_phase1_index(path)))
        self._rewrite_index(path, lambda i: i["runs"][0].pop("config_sha256", None))
        assert any("config_sha256 missing" in e for e in _errors(verify_phase1_index(path)))

    def test_config_not_object(self, fresh) -> None:
        path = fresh / "research/index.json"
        cfg = fresh / "benchmark.json"
        cfg.write_text("[1,2]")
        digest = hashlib.sha256(cfg.read_bytes()).hexdigest()
        self._rewrite_index(path, lambda i: i["runs"][0].update(config_sha256=digest))
        assert any("config unavailable or invalid" in e for e in _errors(verify_phase1_index(path)))

    def test_config_differs_from_frozen(self, fresh) -> None:
        path = fresh / "research/index.json"
        cfg = fresh / "slate.json"
        spec = json.loads(cfg.read_text())
        spec["n_boot"] = 7
        cfg.write_text(json.dumps(spec))
        digest = hashlib.sha256(cfg.read_bytes()).hexdigest()
        self._rewrite_index(path, lambda i: i["runs"][1].update(config_sha256=digest))
        assert any(
            "config differs from frozen slate" in e for e in _errors(verify_phase1_index(path))
        )

    def test_config_protocol_mismatch(self, fresh) -> None:
        path = fresh / "research/index.json"
        cfg = fresh / "benchmark.json"
        proto = json.loads(cfg.read_text())
        proto["min_train_rows"] = 5
        cfg.write_text(json.dumps(proto))
        digest = hashlib.sha256(cfg.read_bytes()).hexdigest()
        self._rewrite_index(path, lambda i: i["runs"][0].update(config_sha256=digest))
        assert any(
            "config differs from frozen protocol" in e for e in _errors(verify_phase1_index(path))
        )

    def test_source_hashes_missing(self, fresh) -> None:
        path = fresh / "research/index.json"
        mpath = fresh / "tournament/manifest.json"
        manifest = _load(mpath)
        manifest.pop("receipt_sha256", None)
        manifest.pop("code_sha256", None)
        sealed = _seal(manifest)
        _write(mpath, sealed)
        index = _load(path)
        index.pop("receipt_sha256", None)
        index["runs"][1]["manifest_sha256"] = sealed["receipt_sha256"]
        _write(path, _seal(index))
        errors = _errors(verify_phase1_index(path))
        assert any("source code hashes missing" in e for e in errors)

    def test_runs_empty(self, fresh) -> None:
        path = fresh / "research/index.json"
        self._rewrite_index(path, lambda i: i.update(runs=[]))
        out = verify_phase1_index(path)
        assert out["valid"] is False
        assert any("nonempty list" in e for e in out["errors"])
