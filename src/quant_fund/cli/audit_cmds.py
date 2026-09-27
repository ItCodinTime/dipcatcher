"""Explicit audit-ledger commands.

Recording is a separate step from simulated execution. These commands do not
submit orders and do not change the paper broker.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from quant_fund.audit.cli import (
    generate_signing_key,
    record_paper,
    record_receipt,
    run_verify,
    trace_number,
)
from quant_fund.audit.errors import AuditError
from quant_fund.cli._app import app


@app.command("verify-ledger")
def verify_ledger_cmd(
    ledger: Path = typer.Argument(..., help="Directory with entries.jsonl and checkpoints.jsonl"),
    trust_pub: Path | None = typer.Option(
        None,
        "--trust-pub",
        help="Pinned Ed25519 public key. Detects a log rewritten under another key.",
    ),
    expect_size: int | None = typer.Option(
        None,
        "--expect-size",
        help="Witness tree size. Detects a tip deleted together with its checkpoint.",
    ),
    expect_root: str | None = typer.Option(
        None,
        "--expect-root",
        help="Witness Merkle root (hex) for --expect-size.",
    ),
    sigstore_identity: str | None = typer.Option(None, "--sigstore-identity"),
    sigstore_issuer: str | None = typer.Option(None, "--sigstore-issuer"),
    sigstore_offline: bool = typer.Option(False, "--sigstore-offline"),
) -> None:
    """Verify a hash-chained audit ledger and its signed Merkle roots."""
    code = run_verify(
        ledger,
        trust_pub=trust_pub,
        expect_size=expect_size,
        expect_root=expect_root,
        sigstore_identity=sigstore_identity,
        sigstore_issuer=sigstore_issuer,
        sigstore_offline=sigstore_offline,
    )
    raise typer.Exit(code=code)


@app.command("audit-record")
def audit_record(
    ledger: Path = typer.Option(..., "--ledger", help="Audit ledger directory"),
    signing_key: Path | None = typer.Option(None, "--signing-key", help="Ed25519 private key hex"),
    generate_key: Path | None = typer.Option(
        None,
        "--generate-key",
        help="Write a new Ed25519 key here (mode 0600) and use it to sign",
    ),
    receipt: Path | None = typer.Option(None, "--receipt", help="Research receipt JSON to commit"),
    paper_dir: Path | None = typer.Option(
        None,
        "--paper-dir",
        help="Simulated paper directory with orders.parquet",
    ),
    sign_every: int = typer.Option(1, "--sign-every", min=1),
) -> None:
    """Append a research receipt or simulated paper rows to the audit ledger."""
    key = signing_key
    if generate_key is not None:
        public = generate_signing_key(generate_key)
        if key is None:
            key = generate_key
        if receipt is None and paper_dir is None:
            typer.echo(public)
            return
    if key is None:
        raise typer.BadParameter("pass --signing-key or --generate-key")
    try:
        if receipt is not None:
            typer.echo(json.dumps(record_receipt(ledger, key, receipt, sign_every=sign_every)))
        if paper_dir is not None:
            typer.echo(json.dumps(record_paper(ledger, key, paper_dir, sign_every=sign_every)))
    except AuditError as exc:
        typer.echo(str(exc))
        for item in exc.errors:
            typer.echo(item)
        raise typer.Exit(code=1) from exc
    if receipt is None and paper_dir is None:
        raise typer.BadParameter("pass --receipt and/or --paper-dir, or --generate-key")


@app.command("audit-trace")
def audit_trace(
    ledger: Path = typer.Option(..., "--ledger", help="Audit ledger directory"),
    receipt: Path = typer.Option(..., "--receipt", help="Research receipt JSON"),
    metric: str = typer.Option(..., "--metric", help="Dotted path of the published number"),
    trust_pub: Path | None = typer.Option(None, "--trust-pub"),
    verify_receipt: bool = typer.Option(
        False,
        "--verify-receipt",
        help="Embed the existing verify-research result. Does not change linkage.",
    ),
) -> None:
    """Trace one published number to its receipt digest, code hashes, and ledger entry."""
    try:
        report = trace_number(
            ledger,
            receipt,
            metric,
            trust_pub=trust_pub,
            verify_receipt=verify_receipt,
        )
    except AuditError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from exc
    raise typer.Exit(code=0 if report.get("linked") else 1)
