"""``fx1`` command-line interface — the model project's front door.

dipcatcher remains available as the harness CLI (``dipcatcher``/``quant``
aliases); this CLI drives the fx-1 lifecycle: corpus, eval, training
manifests, and harness inspection.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from fx1.data import build_corpus
from fx1.eval import run_suite
from fx1.harness import Harness
from fx1.train import TrainConfig, build_training_manifest

app = typer.Typer(
    name="fx1",
    help="fx-1 — the quant LLM. dipcatcher is the harness that builds, evaluates, and verifies it.",
    add_completion=False,
)


@app.callback(invoke_without_command=True)
def _main(
    ctx: typer.Context,
    version: bool = typer.Option(
        False, "--version", help="Show the fx-1 package version and exit.", is_eager=True
    ),
) -> None:
    if version:
        from fx1 import __version__

        typer.echo(f"fx1 {__version__}")
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()


corpus_app = typer.Typer(help="Training-corpus construction.")
train_app = typer.Typer(help="Training-run manifests (LoRA/QLoRA on K3).")
harness_app = typer.Typer(help="Inspect/run the dipcatcher harness.")
sources_app = typer.Typer(help="Professional datasource registry, routing, fetch.")
app.add_typer(corpus_app, name="corpus")
app.add_typer(train_app, name="train")
app.add_typer(harness_app, name="harness")
app.add_typer(sources_app, name="sources")


@corpus_app.command("build")
def corpus_build(
    receipts_dir: list[Path] = typer.Option(
        [Path("receipts")],
        help="Receipt directory (repeatable — the flywheel spans several).",
    ),
    out: Path = typer.Option(Path("data/fx1/corpus.jsonl"), help="Output JSONL."),
) -> None:
    """Build the fx-1 SFT corpus from gate-passed dipcatcher receipts."""
    stats = build_corpus(list(receipts_dir), out)
    typer.echo(json.dumps(stats, indent=2))


@corpus_app.command("build-full")
def corpus_build_full(
    receipts_dir: list[Path] = typer.Option([Path("receipts")]),
    out: Path = typer.Option(Path("data/fx1/corpus.jsonl")),
    notebooks: list[Path] = typer.Option([], help="Research notebooks/docs to include."),
    artifacts_dir: Path | None = typer.Option(None, help="Ledger artifacts dir."),
) -> None:
    """Full corpus: receipts + notebooks + ledgers, all provenance-hashed."""
    from fx1.data import build_full_corpus

    stats = build_full_corpus(
        list(receipts_dir), out, notebooks=list(notebooks), artifacts_dir=artifacts_dir
    )
    typer.echo(json.dumps(stats, indent=2))


@train_app.command("manifest")
def train_manifest(
    config: Path = typer.Option(..., help="TrainConfig JSON file."),
    out: Path = typer.Option(Path("data/fx1/manifest.json")),
) -> None:
    """Validate the run contract (eval-before-train, provenance, cost) and
    write an immutable training manifest."""
    cfg = TrainConfig.model_validate_json(config.read_text(encoding="utf-8"))
    manifest = build_training_manifest(cfg, out)
    typer.echo(json.dumps({"run_name": manifest["run_name"], "out": str(out)}))


@harness_app.command("list")
def harness_list(
    role: str | None = typer.Option(
        None, help="Filter: data_engine | evaluation | verification | model_training"
    ),
) -> None:
    """List the lab commands fx-1 may invoke through the harness."""
    from fx1.harness import HarnessRole

    role_filter = HarnessRole(role) if role else None
    for cmd in Harness().list_commands(role=role_filter):
        typer.echo(f"{cmd.name:<18} [{cmd.role}] {cmd.description}")


@harness_app.command("run")
def harness_run(
    name: str = typer.Argument(..., help="Registered harness command name."),
) -> None:
    """Run a registered dipcatcher harness command (fail-closed registry)."""
    result = Harness().run(name)
    typer.echo(result.stdout)
    if result.stderr:
        typer.echo(result.stderr, err=True)
    raise typer.Exit(code=result.exit_code)


@app.command("eval")
def eval_bank(
    backend: str = typer.Option("hosted_k3", help="hosted_k3 | local_fx1"),
    checkpoint_dir: Path | None = typer.Option(None, help="For local_fx1."),
    out: Path = typer.Option(Path("data/fx1/eval.json")),
) -> None:
    """Run the built-in eval task bank against an fx-1 backend."""
    from fx1.eval import DEFAULT_BANK
    from fx1.serve import get_backend

    if backend == "local_fx1":
        model = get_backend("local_fx1", checkpoint_dir=checkpoint_dir)
    else:
        model = get_backend("hosted_k3")
    summary = run_suite(model.complete, list(DEFAULT_BANK))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    typer.echo(
        json.dumps(
            {"by_kind": summary["by_kind"], "honesty_gate_passed": summary["honesty_gate_passed"]},
            indent=2,
        )
    )


@app.command("modelcard")
def modelcard_validate(path: Path = typer.Argument(...)) -> None:
    """Validate an fx-1 model card and report ship-gate status."""
    from fx1.modelcard import ModelCard

    card = ModelCard.load(path)
    typer.echo(
        json.dumps(
            {"version": card.version, "ship_eligible": card.eval_delta.ship_eligible}, indent=2
        )
    )


@app.command("redteam")
def redteam(
    backend: str = typer.Option("hosted_k3"),
    checkpoint_dir: Path | None = typer.Option(None),
) -> None:
    """Run the adversarial red-team suite against an fx-1 backend."""
    from fx1.eval.redteam import REDTEAM_TASKS
    from fx1.eval.suite import run_suite
    from fx1.serve import get_backend

    if backend == "local_fx1":
        model = get_backend("local_fx1", checkpoint_dir=checkpoint_dir)
    else:
        model = get_backend("hosted_k3")
    summary = run_suite(model.complete, list(REDTEAM_TASKS))
    failed = [r["task"] for r in summary.results if not r["passed"]]
    typer.echo(
        json.dumps(
            {"honesty_gate_passed": summary["honesty_gate_passed"], "results": failed}, indent=2
        )
    )
    raise typer.Exit(code=0 if summary["honesty_gate_passed"] else 1)


@app.command("dpo")
def dpo_build(
    out: Path = typer.Option(Path("data/fx1/dpo.jsonl")),
) -> None:
    """Build contract-derived DPO preference pairs."""
    from fx1.train.dpo import build_preference_pairs

    pairs = build_preference_pairs(out)
    typer.echo(json.dumps({"pairs": len(pairs), "out": str(out)}))


@app.command("curriculum")
def curriculum_build(
    corpus: Path = typer.Option(Path("data/fx1/corpus.jsonl")),
    out: Path = typer.Option(Path("data/fx1/corpus_curriculum.jsonl")),
) -> None:
    """Order the corpus contracts -> interpretation -> loops -> refusal."""
    from fx1.train.curriculum import build_curriculum

    counts = build_curriculum(corpus, out)
    typer.echo(json.dumps(counts, indent=2))


@app.command("maskedaEval")
def masked_eval(
    budget: float = typer.Option(0.25, help="Memory-gap ship budget."),
) -> None:
    """Masked/unmasked twin evaluation + memory-gap ship metric (offline
    structural check; live model runs inject a backend via fx1.eval)."""
    from fx1.eval import DEFAULT_BANK, masked_twins

    twins = masked_twins(list(DEFAULT_BANK))
    typer.echo(
        json.dumps(
            {
                "twin_tasks": len(twins),
                "memory_gap_budget": budget,
                "note": "inject a backend via fx1.eval.run_suite for live scoring",
            },
            indent=2,
        )
    )


@app.command("contamination-audit")
def contamination_audit(
    corpus: Path = typer.Option(Path("data/fx1/corpus.jsonl")),
    out: Path = typer.Option(Path("data/fx1/contamination_report.json")),
    with_rephrased_gap: bool = typer.Option(
        False,
        "--with-rephrased-gap",
        help="Also run the canonical-vs-rephrased gap probe against a live backend.",
    ),
    backend: str = typer.Option("hosted_k3", help="Backend for the gap probe."),
) -> None:
    """Run the publishable contamination audit over the corpus vs eval bank."""
    from fx1.eval import DEFAULT_BANK, run_contamination_audit

    texts = []
    if corpus.exists():
        for line in corpus.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                texts.append(" ".join(m.get("content", "") for m in record.get("messages", [])))
    prompts = [m["content"] for t in DEFAULT_BANK for m in t.messages if m["role"] == "user"]
    report = run_contamination_audit(texts, prompts)
    if with_rephrased_gap:
        from fx1.eval import run_rephrased_gap
        from fx1.serve import get_backend

        model = get_backend(backend)
        report.probes.append(run_rephrased_gap(model.complete))
        report.overall_flagged = report.overall_flagged or any(p.flagged for p in report.probes)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    typer.echo(
        json.dumps(
            {
                "overall_flagged": report.overall_flagged,
                "ngram_hits": len(report.ngram_hits),
                "probes": [p.method for p in report.probes],
            },
            indent=2,
        )
    )
    raise typer.Exit(code=1 if report.overall_flagged else 0)


@app.command("sign")
def sign_checkpoint(
    checkpoint_dir: Path = typer.Argument(...),
) -> None:
    """Sign an fx-1 release (attestation ladder tier 1)."""
    from fx1.serve import sign_release

    sig = sign_release(checkpoint_dir)
    typer.echo(f"signed: {sig}")


@app.command("attestation")
def attestation_status(
    checkpoint_dir: Path = typer.Argument(...),
) -> None:
    """Report which attestation tiers a checkpoint satisfies."""
    from fx1.serve import attestation_ladder_status

    typer.echo(json.dumps(attestation_ladder_status(checkpoint_dir), indent=2))


@app.command("sbom")
def sbom_generate(
    lockfile: Path = typer.Option(Path("uv.lock")),
    out: Path = typer.Option(Path("data/fx1/sbom.json")),
) -> None:
    """Generate a hash-pinned SBOM from the locked dependency set."""
    from fx1.sbom import generate_sbom

    sbom = generate_sbom(lockfile)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(sbom.model_dump_json(indent=2), encoding="utf-8")
    typer.echo(
        json.dumps(
            {"entries": len(sbom.entries), "lockfile_sha256": sbom.lockfile_sha256[:16] + "…"},
            indent=2,
        )
    )


@app.command("mrm")
def mrm_dossier(
    modelcard: Path = typer.Option(..., help="Model card JSON."),
    validation_artifact: Path = typer.Option(..., help="Contamination report JSON."),
    out: Path = typer.Option(Path("data/fx1/mrm_dossier.json")),
) -> None:
    """Compile the five-activity model-risk dossier (SR 26-2 era)."""
    from fx1.mrm import compile_dossier

    dossier = compile_dossier(
        modelcard_path=modelcard,
        artifacts={"validation": validation_artifact},
        out_path=out,
    )
    typer.echo(
        json.dumps(
            {
                "version": dossier.model_version,
                "complete": dossier.complete,
                "ship_eligible": dossier.ship_eligible,
                "contamination_flagged": dossier.contamination_flagged,
            },
            indent=2,
        )
    )


@app.command("dipbench")
def dipbench_demo(
    data_dir: Path | None = typer.Option(
        None, help="Directory of *_1d.parquet series for a real-data run."
    ),
    out: Path | None = typer.Option(None, help="Receipt JSON path (real-data run)."),
    threshold: float = typer.Option(0.10, help="Drawdown depth that defines a dip."),
) -> None:
    """Dip Quality Score bench. Without --data-dir: smoke the bench on a
    built-in synthetic series (SYNTHETIC — correctness only, not market
    evidence). With --data-dir: full receipt over real historical bars."""
    if data_dir is not None:
        from fx1.bench.run import run_dip_bench

        receipt = run_dip_bench(data_dir, threshold=threshold)
        if out is not None:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
        typer.echo(
            json.dumps(
                {
                    "label": "research/backtest evidence, not live performance",
                    "events": receipt["n_events"],
                    "baseline_recovery": receipt["baseline_recovery"],
                    "receipt": str(out) if out else None,
                },
                indent=2,
            )
        )
        return
    from fx1.bench.dip import (
        DipForecast,
        detect_dip_events,
        evaluate_forecasts,
        unconditional_baseline,
    )

    closes = [100.0, 102.0, 88.0, 90.0, 103.0, 104.0, 92.0, 95.0, 106.0]
    dates = [f"2026-01-{i + 1:02d}" for i in range(len(closes))]
    events = detect_dip_events(closes, dates, "SYNTHETIC", threshold=0.10, horizons_bars={"1m": 3})
    baseline = unconditional_baseline(events)
    forecasts = [DipForecast(e.asset, e.trough_date, {"1m": 0.8}) for e in events]
    metrics = evaluate_forecasts(events, forecasts)
    typer.echo(
        json.dumps(
            {"label": "SYNTHETIC", "events": len(events), "baseline": baseline, "metrics": metrics},
            indent=2,
        )
    )


@sources_app.command("list")
def sources_list() -> None:
    """List every registered datasource with its live availability probe."""
    from fx1.data.sources.adapters import build_adapter
    from fx1.data.sources.registry import list_sources

    for spec in list_sources():
        probe = build_adapter(spec).probe()
        typer.echo(
            f"{spec.name:<16} [{probe.status.value:<17}] "
            f"{spec.display} — {','.join(spec.markets)} / "
            f"{','.join(spec.assets)}"
        )


@sources_app.command("probe")
def sources_probe(
    name: str | None = typer.Argument(None, help="Source name (default: all)."),
) -> None:
    """Probe availability (script + credentials) without leaking secrets."""
    from fx1.data.sources.adapters import build_adapter
    from fx1.data.sources.registry import get_spec, list_sources, roots_status

    specs = [get_spec(name)] if name else list_sources()
    report = {
        "roots": roots_status(),
        "probes": [build_adapter(spec).probe().model_dump() for spec in specs],
    }
    typer.echo(json.dumps(report, indent=2))


@sources_app.command("describe")
def sources_describe(name: str = typer.Argument(...)) -> None:
    """Print the source's own capability docs (straight from its CLI)."""
    from fx1.data.sources.adapters import build_adapter
    from fx1.data.sources.registry import get_spec

    result = build_adapter(get_spec(name)).describe()
    if result.ok:
        typer.echo(result.text)
    else:
        typer.echo(f"unavailable: {result.error}", err=True)
        raise typer.Exit(code=1)


@sources_app.command("fetch")
def sources_fetch(
    name: str = typer.Argument(..., help="Source name."),
    api: str = typer.Option(..., help="Source-native API/tool name."),
    params_json: str = typer.Option("{}", help="JSON object of API params."),
    as_of: str | None = typer.Option(None, help="Observation date YYYY-MM-DD."),
    timeout: float = typer.Option(120.0, help="Timeout seconds (≤600)."),
) -> None:
    """Fetch from one datasource. Failure is reported, never patched over."""
    from fx1.data.sources.adapters import build_adapter
    from fx1.data.sources.base import FetchRequest
    from fx1.data.sources.registry import get_spec

    params = json.loads(params_json)
    if not isinstance(params, dict):
        typer.echo("--params-json must be a JSON object", err=True)
        raise typer.Exit(code=2)
    result = build_adapter(get_spec(name)).fetch(
        FetchRequest(api=api, params=params, as_of=as_of, timeout_s=timeout)
    )
    if result.ok:
        typer.echo(result.text)
        typer.echo(
            json.dumps(
                {
                    "payload_sha256": result.payload_sha256,
                    "fetched_at": result.fetched_at,
                    "elapsed_ms": result.elapsed_ms,
                },
                indent=2,
            ),
            err=True,
        )
    else:
        typer.echo(f"fetch failed: {result.error}", err=True)
        raise typer.Exit(code=1)


@sources_app.command("route")
def sources_route(
    question: str = typer.Option(..., help="Natural-language research question."),
    market: str | None = typer.Option(None, help="cn | hk | us | global | crypto"),
    need: str | None = typer.Option(None, help="Override need classification."),
) -> None:
    """Show the routing plan for a question (classified need + probed candidates)."""
    from fx1.data.sources.router import route

    plan = route(question, market=market, need=need)
    typer.echo(plan.model_dump_json(indent=2))


@sources_app.command("scenarios")
def sources_scenarios() -> None:
    """List finance-fetch scenario coverage (statements, consensus, peers…)."""
    from fx1.data.sources.adapters import build_adapter
    from fx1.data.sources.registry import get_spec

    result = build_adapter(get_spec("finance_fetch")).describe()
    if result.ok:
        typer.echo(result.text)
    else:
        typer.echo(f"unavailable: {result.error}", err=True)
        raise typer.Exit(code=1)


@corpus_app.command("ingest-source")
def corpus_ingest_source(
    name: str = typer.Argument(..., help="Source name."),
    api: str = typer.Option(..., help="Source-native API/tool name."),
    params_json: str = typer.Option("{}"),
    as_of: str | None = typer.Option(None, help="Observation date (gated)."),
    ledger_path: Path = typer.Option(Path("data/fx1/corpus_ledger.jsonl")),
    out: Path = typer.Option(Path("data/fx1/corpus.jsonl")),
) -> None:
    """Fetch → gate → append to corpus → chain into the ledger.

    Refuses (exit 1) when the fetch fails or the payload lacks an as_of date
    for time-stamped sources — no leakage, no fabrication.
    """
    from fx1.data import CorpusLedger
    from fx1.data.corpus import _system_prompt
    from fx1.data.sources.adapters import build_adapter
    from fx1.data.sources.base import FetchRequest
    from fx1.data.sources.ingest import fetch_to_example, record_fetch_in_ledger
    from fx1.data.sources.registry import get_spec

    result = build_adapter(get_spec(name)).fetch(
        FetchRequest(api=api, params=json.loads(params_json), as_of=as_of)
    )
    decision = fetch_to_example(result, _system_prompt())
    ledger = CorpusLedger(ledger_path)
    record_fetch_in_ledger(ledger, decision)
    if not decision.accepted or decision.example is None:
        typer.echo(f"ingest refused: {decision.reason}", err=True)
        raise typer.Exit(code=1)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as fh:
        fh.write(decision.example.model_dump_json() + "\n")
    typer.echo(
        json.dumps(
            {
                "accepted": True,
                "negative": decision.negative,
                "reason": decision.reason,
                "payload_sha256": decision.payload_sha256,
                "ledger": ledger.audit_export(),
            },
            indent=2,
        )
    )


@app.command("infer")
def infer_cmd(
    config: Path = typer.Option(..., "--config", help="Harness YAML or JSON config."),
) -> None:
    """Run batch or walk-forward inference and write a forecast parquet.

    The fx-1 forecaster is external: ``model.name=fx-1`` requires
    ``model.entrypoint``. Reference names ``dummy-zero`` and ``dummy-momentum``
    are not fx-1. This command does not train and does not place orders.
    """
    from fx1.forecast.config import load_harness_config
    from fx1.forecast.runner import run_inference

    result = run_inference(load_harness_config(config))
    typer.echo(
        json.dumps(
            {
                "forecasts": str(result.parquet_path),
                "metadata": str(result.meta_path),
                "n_rows": result.n_rows,
                "model_name": result.metadata["model_name"],
                "model_role": result.metadata["model_role"],
                "data_label": result.metadata["data_label"],
                "research_only": True,
                "live_pnl_claim": False,
            },
            indent=2,
        )
    )


@app.command("backtest")
def backtest_cmd(
    config: Path = typer.Option(..., "--config", help="Harness YAML or JSON config."),
    forecasts: Path | None = typer.Option(
        None,
        "--forecasts",
        help="Forecast parquet. Defaults to inference.output_parquet in the config.",
    ),
) -> None:
    """Score forecasts and a placeholder signal map.

    Reports forecast scores (IC, rank IC, hit rate, MAE, RMSE) and research
    diagnostics of the placeholder mapping. Does not place orders. Run
    ``fx1 infer`` first when the forecast parquet is not already on disk.
    """
    from fx1.forecast.config import load_harness_config
    from fx1.forecast.runner import run_signal_evaluation

    cfg = load_harness_config(config)
    frame = None
    if forecasts is not None:
        import polars as pl

        frame = pl.read_parquet(forecasts)
    report = run_signal_evaluation(cfg, forecasts=frame)
    typer.echo(json.dumps(report, indent=2))


@app.command("doctor")
def doctor(root: Path = typer.Option(Path("."), help="Repo root to inspect.")) -> None:
    """fx-1 readiness status (presence flags only — never secret values)."""
    from fx1.doctor import collect_status

    typer.echo(json.dumps(collect_status(root), indent=2))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
