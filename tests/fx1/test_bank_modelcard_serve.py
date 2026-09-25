"""Tests for the eval bank, model cards, and inference backends."""

from pathlib import Path

import pytest

from fx1.eval import DEFAULT_BANK, run_suite
from fx1.modelcard import EvalDelta, ModelCard
from fx1.serve import get_backend
from fx1.serve.backends import HostedK3Backend, LocalFx1Backend


def test_bank_composition():
    kinds = {t.kind for t in DEFAULT_BANK}
    assert kinds == {"honesty", "domain", "general"}
    assert sum(1 for t in DEFAULT_BANK if t.kind == "honesty") >= 5


def test_bank_catches_noncompliant_model():
    # A base model that blurts Sharpe headlines must fail the honesty gate.
    bad = lambda msgs: "Sure — the sharpe: 2.4 is the headline."  # noqa: E731
    summary = run_suite(bad, [t for t in DEFAULT_BANK if t.kind == "honesty"])
    assert summary["honesty_gate_passed"] is False


def _delta(**kw) -> EvalDelta:
    base = {
        "domain_pass_rate_base": 0.5,
        "domain_pass_rate_candidate": 0.7,
        "general_pass_rate_base": 0.9,
        "general_pass_rate_candidate": 0.9,
        "honesty_gate_candidate": True,
    }
    base.update(kw)
    return EvalDelta(**base)


def _card(**kw) -> ModelCard:
    defaults = {
        "version": "fx-1.v0.1",
        "corpus_sha256": "a" * 64,
        "corpus_receipt_range": "b5942241..f0e1d2c3",
        "training_manifest_sha256": "b" * 64,
        "eval_delta": _delta(),
    }
    defaults.update(kw)
    return ModelCard(**defaults)


def test_modelcard_roundtrip_and_ship_gate(tmp_path: Path):
    card = _card()
    path = tmp_path / "modelcard.json"
    card.save(path)
    loaded = ModelCard.load(path)
    assert loaded.version == "fx-1.v0.1"
    assert loaded.eval_delta.ship_eligible


def test_ship_gate_blocks_regression_and_dishonesty():
    assert not _delta(general_pass_rate_candidate=0.8).ship_eligible
    assert not _delta(honesty_gate_candidate=False).ship_eligible
    assert not _delta(domain_pass_rate_candidate=0.5).ship_eligible


def test_modelcard_never_live():
    with pytest.raises(ValueError, match="research-scoped"):
        _card(live_pnl_claim=True)
    with pytest.raises(ValueError, match="research-scoped"):
        _card(research_only=False)


def test_hosted_backend_requires_env_key(monkeypatch):
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="MOONSHOT_API_KEY"):
        HostedK3Backend()


def test_local_backend_fail_closed(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="model card"):
        LocalFx1Backend(tmp_path)
    _card(eval_delta=_delta(honesty_gate_candidate=False)).save(tmp_path / "modelcard.json")
    with pytest.raises(RuntimeError, match="ship gate"):
        LocalFx1Backend(tmp_path)


def test_backend_factory_fail_closed(tmp_path: Path):
    with pytest.raises(KeyError, match="unknown backend"):
        get_backend("openai")
    _card().save(tmp_path / "modelcard.json")
    backend = get_backend("local_fx1", checkpoint_dir=tmp_path)
    assert isinstance(backend, LocalFx1Backend)
