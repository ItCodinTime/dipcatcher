"""CLI coverage: research_cmds synthetic benches, research100_cli,
lightspeed/cli entry points, cli/main lazy re-exports + observe hook.
All runs are SYNTHETIC-labeled, tiny parametrized, deterministic."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from quant_fund.cli.app import app
from quant_fund.lightspeed.cli import ls_app
from quant_fund.research.research100_cli import research100_app

RUNNER = CliRunner()
CFG = Path(__file__).resolve().parents[3] / "configs" / "research.yaml"


class TestResearchCmds:
    def test_rankic_writes_receipt(self, tmp_path: Path) -> None:
        res = RUNNER.invoke(
            app,
            [
                "rankic",
                "--panels",
                "linear_signal",
                "--n-assets",
                "8",
                "--n-dates",
                "60",
                "--horizons",
                "1",
                "--seed",
                "5",
                "--out-dir",
                str(tmp_path),
            ],
        )
        assert res.exit_code == 0, res.output
        assert "SYNTHETIC" in res.output
        receipts = list(tmp_path.glob("rankic_eval_*.json"))
        assert receipts, res.output
        body = json.loads(receipts[0].read_text())
        assert body

    def test_rankic_bad_panel_fails_closed(self, tmp_path: Path) -> None:
        res = RUNNER.invoke(
            app,
            [
                "rankic",
                "--panels",
                "no_such_panel",
                "--n-dates",
                "60",
                "--out-dir",
                str(tmp_path),
            ],
        )
        assert res.exit_code != 0

    def test_verify_identities_and_receipt(self, tmp_path: Path) -> None:
        receipt = tmp_path / "ids.json"
        res = RUNNER.invoke(
            app,
            [
                "verify-identities",
                "--out",
                str(receipt),
                "--trials",
                "2",
                "--seed",
                "3",
            ],
        )
        assert res.exit_code == 0, res.output
        assert "SYNTHETIC" in res.output
        assert receipt.exists()

        res = RUNNER.invoke(app, ["verify-receipt", str(receipt)])
        assert res.exit_code == 0, res.output
        result = json.loads(res.output)
        assert result["valid"] is True
        assert result["kind"] == "identity_sweep"

    def test_verify_receipt_rejects_corrupt_file(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text('{"kind": "fake", "receipt_sha256": "0"}')
        res = RUNNER.invoke(app, ["verify-receipt", str(bad)])
        assert res.exit_code != 0

    def test_fleet_minimal(self, tmp_path: Path) -> None:
        res = RUNNER.invoke(
            app,
            [
                "fleet",
                "--config",
                str(CFG),
                "--models",
                "empirical",
                "--shards",
                "iid_gaussian",
                "--n-train",
                "96",
                "--n-eval",
                "48",
                "--out-dir",
                str(tmp_path),
            ],
        )
        assert res.exit_code == 0, res.output
        assert list(tmp_path.glob("fleet_eval_*.json"))

    def test_fleet_receipt_v2_and_bad_version(self, tmp_path: Path) -> None:
        base = [
            "fleet",
            "--config",
            str(CFG),
            "--models",
            "empirical",
            "--shards",
            "iid_gaussian",
            "--n-train",
            "96",
            "--n-eval",
            "48",
        ]
        res = RUNNER.invoke(app, base + ["--out-dir", str(tmp_path), "--receipt-version", "2"])
        assert res.exit_code == 0, res.output
        res = RUNNER.invoke(app, base + ["--out-dir", str(tmp_path), "--receipt-version", "9"])
        assert res.exit_code != 0

    def test_vol_bench_minimal(self, tmp_path: Path) -> None:
        res = RUNNER.invoke(
            app,
            [
                "vol-bench",
                "--config",
                str(CFG),
                "--models",
                "rv_roll",
                "--shards",
                "garch_vol",
                "--horizons",
                "1",
                "--min-history",
                "64",
                "--n-origins",
                "10",
                "--out-dir",
                str(tmp_path),
            ],
        )
        assert res.exit_code == 0, res.output
        assert list(tmp_path.glob("vol_bench_*.json"))

    def test_vol_bench_bad_model_fails(self, tmp_path: Path) -> None:
        res = RUNNER.invoke(
            app,
            [
                "vol-bench",
                "--config",
                str(CFG),
                "--models",
                "bogus_model",
                "--horizons",
                "1",
                "--n-origins",
                "10",
                "--out-dir",
                str(tmp_path),
            ],
        )
        assert res.exit_code != 0

    def test_capacity_dev_guard_and_run(self, tmp_path: Path) -> None:
        res = RUNNER.invoke(app, ["capacity", "--books", "uniform", "--out-dir", str(tmp_path)])
        assert res.exit_code != 0  # --dev required
        res = RUNNER.invoke(
            app,
            [
                "capacity",
                "--books",
                "uniform",
                "--n-dates",
                "20",
                "--n-names",
                "8",
                "--aum-grid",
                "1e6",
                "--dev",
                "--out-dir",
                str(tmp_path),
            ],
        )
        assert res.exit_code == 0, res.output
        assert list(tmp_path.glob("capacity_eval_*.json"))


class TestResearch100Cli:
    def test_catalog_lists_entries(self) -> None:
        res = RUNNER.invoke(research100_app, ["catalog"])
        assert res.exit_code == 0
        body = json.loads(res.output)
        assert isinstance(body, (list, dict))

    def test_catalog_one_id(self) -> None:
        res = RUNNER.invoke(research100_app, ["catalog", "R008"])
        assert res.exit_code == 0
        assert "title" in res.output or "authors" in res.output

    def test_catalog_bad_id_fails(self) -> None:
        res = RUNNER.invoke(research100_app, ["catalog", "ZZZZ"])
        assert res.exit_code != 0

    def test_verify_completes(self, tmp_path: Path) -> None:
        res = RUNNER.invoke(research100_app, ["verify", "--output", str(tmp_path / "v.json")])
        assert res.exit_code == 0, res.output
        assert (tmp_path / "v.json").exists()

    def test_benchmark_requires_synthetic(self, tmp_path: Path) -> None:
        res = RUNNER.invoke(research100_app, ["benchmark"])
        assert res.exit_code != 0
        res = RUNNER.invoke(
            research100_app,
            ["benchmark", "--synthetic", "--seed", "100", "--output", str(tmp_path / "b.json")],
        )
        assert res.exit_code == 0, res.output
        assert (tmp_path / "b.json").exists()


class TestLightspeedCli:
    def test_specs(self) -> None:
        res = RUNNER.invoke(ls_app, ["specs"])
        assert res.exit_code == 0
        assert "TQQQ" in res.output or "family" in res.output

    def test_demo_research_payload(self) -> None:
        res = RUNNER.invoke(ls_app, ["demo", "--seed", "3"])
        assert res.exit_code == 0, res.output
        body = json.loads(res.output)
        assert body["research_only"] is True
        assert body["live_pnl_claim"] is False
        assert body["metalabel_in_unit_interval"] is True
        assert 0.0 <= body["metalabel_last_multiplier"] <= 1.0
        assert abs(sum(body["tqqq_last"].values()) - 1.0) < 1e-6


class TestCliMain:
    def test_lazy_reexports_resolve(self) -> None:
        import quant_fund.cli.main as main_mod

        assert callable(main_mod.dump_resolved)
        assert callable(main_mod.load_config)
        assert callable(main_mod.run_doctor)
        assert callable(main_mod.configure_logging)
        assert callable(main_mod.get_logger)
        # re-exported registry names exist
        for name in ("research", "fleet", "vol_bench", "capacity"):
            assert name in main_mod.__all__

    def test_observe_hook_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import quant_fund.cli.main as main_mod

        called: list[bool] = []

        def _fake() -> None:
            called.append(True)

        monkeypatch.setenv("DIPCATCHER_OBSERVE", "1")
        import quant_fund.observe.install as oi

        monkeypatch.setattr(oi, "install_passive_hooks", _fake)
        main_mod._maybe_install_observation_hooks()
        assert called

        called.clear()
        monkeypatch.setenv("DIPCATCHER_OBSERVE", "0")
        main_mod._maybe_install_observation_hooks()
        assert not called

        called.clear()
        monkeypatch.delenv("DIPCATCHER_OBSERVE", raising=False)
        main_mod._maybe_install_observation_hooks()
        assert not called

    def test_app_module_import(self) -> None:
        mod = importlib.import_module("quant_fund.cli.main")
        assert mod.app is not None
