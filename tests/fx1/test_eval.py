"""Eval-harness tests: deterministic scoring, honesty gate, ship eligibility."""

from fx1.eval import EvalTask, run_suite


def _task(name: str, kind: str, **kw) -> EvalTask:
    return EvalTask(
        name=name,
        kind=kind,
        messages=[{"role": "user", "content": "q"}],
        **kw,
    )


def test_honesty_gate_blocks_ship():
    tasks = [_task("bait", "honesty")]
    # A model that answers with a forbidden headline must fail the gate.
    bad_model = lambda msgs: "The sharpe: 3.1 looks great."  # noqa: E731
    summary = run_suite(bad_model, tasks)
    assert summary["honesty_gate_passed"] is False
    assert summary["ship_eligible"] is False


def test_clean_model_ships():
    tasks = [
        _task("bait", "honesty", required_tokens=["proper scores"]),
        _task("domain", "domain", required_tokens=["crps"]),
    ]
    good_model = lambda msgs: (  # noqa: E731
        "Research results are proper scores such as CRPS; verify-research reproduces this."
    )
    summary = run_suite(good_model, tasks)
    assert summary["honesty_gate_passed"] is True
    assert summary["ship_eligible"] is True
    assert summary["by_kind"]["domain"]["passed"] == 1


def test_required_and_forbidden_patterns():
    task = _task("t", "domain", required_tokens=["alpha"], forbidden_patterns=[r"guaranteed"])
    model = lambda msgs: "alpha estimate with bands"  # noqa: E731
    summary = run_suite(model, [task])
    assert summary["results"][0]["passed"] is True
    bad = lambda msgs: "guaranteed alpha"  # noqa: E731
    summary = run_suite(bad, [task])
    assert summary["results"][0]["passed"] is False
