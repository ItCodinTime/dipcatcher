"""CLI surface smoke coverage for ``src/quant_fund/cli/main.py``.

Every registered command and sub-command must render ``--help`` cleanly —
this exercises the option/callback registration code paths, which are the
bulk of the module. A handful of fail-fast invocations additionally assert
missing-input errors are typed and clean (usage/exit codes), never raw
tracebacks.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from quant_fund.cli.main import app

runner = CliRunner()


def _command_name(cmd) -> str:
    return cmd.name or cmd.callback.__name__.replace("_", "-")


ROOT_COMMANDS = sorted(_command_name(c) for c in app.registered_commands)
GROUPS = {
    (g.name or g.typer_instance.info.name): sorted(
        _command_name(c) for c in g.typer_instance.registered_commands
    )
    for g in app.registered_groups
}


def test_root_help_lists_everything() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert result.output.strip()
    for name in ROOT_COMMANDS:
        assert name in result.output
    for name in GROUPS:
        assert name in result.output


@pytest.mark.parametrize("name", ROOT_COMMANDS)
def test_command_help(name: str) -> None:
    result = runner.invoke(app, [name, "--help"])
    assert result.exit_code == 0, f"{name} --help exited {result.exit_code}"
    assert result.output.strip(), f"{name} --help produced no output"


@pytest.mark.parametrize("group", sorted(GROUPS))
def test_group_help(group: str) -> None:
    result = runner.invoke(app, [group, "--help"])
    assert result.exit_code == 0, f"{group} --help exited {result.exit_code}"
    for sub in GROUPS[group]:
        assert sub in result.output


@pytest.mark.parametrize(
    "argv",
    [(group, sub) for group, subs in sorted(GROUPS.items()) for sub in subs],
    ids=lambda a: f"{a[0]}-{a[1]}",
)
def test_subcommand_help(argv: tuple[str, str]) -> None:
    result = runner.invoke(app, [*argv, "--help"])
    assert result.exit_code == 0, f"{' '.join(argv)} --help exited {result.exit_code}"
    assert result.output.strip()


@pytest.mark.parametrize(
    "argv",
    [
        ("no-such-command",),
        ("train", "no-such-sub"),
        ("paper", "--config", "definitely/missing/paper.yaml"),
        ("doctor", "--config", "definitely/missing/research.yaml"),
        ("verify-research", "--path", "definitely/missing/receipt.json"),
    ],
    ids=lambda a: "-".join(a),
)
def test_missing_inputs_fail_cleanly(argv: tuple[str, ...]) -> None:
    """Bad command names / missing configs exit nonzero without a traceback."""
    result = runner.invoke(app, list(argv))
    assert result.exit_code != 0, f"{' '.join(argv)} unexpectedly succeeded"
    combined = result.output + (str(result.exception) if result.exception else "")
    assert "Traceback" not in combined, f"{' '.join(argv)} raised a raw traceback"
