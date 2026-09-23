"""CLI for the SLP3 Appendix A discrete HMM. Research only."""

from __future__ import annotations

import json

import typer

hmm_app = typer.Typer(help="Jurafsky & Martin SLP3 Appendix A HMM (Eisner ice cream).")


def _obs(text: str) -> list[int]:
    return [int(p.strip()) for p in text.split(",") if p.strip()]


@hmm_app.command("eisner")
def eisner_cmd(obs: str = typer.Option("3,1,3", "--obs")) -> None:
    """Forward likelihood + Viterbi path on the Fig. A.2 ice-cream HMM."""
    from quant_fund.hmm.discrete import forward, viterbi
    from quant_fund.hmm.eisner import EISNER, STATE_NAMES, ice_cream

    o = ice_cream(_obs(obs))
    alpha, p = forward(EISNER, o)
    path, path_p = viterbi(EISNER, o)
    typer.echo(
        json.dumps(
            {
                "obs": _obs(obs),
                "likelihood": p,
                "alpha_t1": alpha[0].tolist(),
                "viterbi_path": [STATE_NAMES[i] for i in path.tolist()],
                "viterbi_path_prob": path_p,
                "source": "https://web.stanford.edu/~jurafsky/slp3/A.pdf",
                "research_only": True,
            },
            indent=2,
        )
    )


@hmm_app.command("train")
def train_cmd(
    obs: str = typer.Option("3,1,3,2,1,2,3", "--obs"),
    n_states: int = typer.Option(2, "--n-states"),
    n_iter: int = typer.Option(15, "--n-iter"),
) -> None:
    """Baum–Welch on ice-cream counts. Prints likelihood history."""
    from quant_fund.hmm.discrete import baum_welch
    from quant_fund.hmm.eisner import ice_cream

    counts = _obs(obs)
    o = ice_cream(counts)
    model, hist = baum_welch(o, n_states=n_states, n_obs=3, n_iter=n_iter)
    typer.echo(
        json.dumps(
            {
                "obs": counts,
                "likelihood_history": hist,
                "non_decreasing": all(hist[i] <= hist[i + 1] + 1e-12 for i in range(len(hist) - 1)),
                "A": model.A.tolist(),
                "B": model.B.tolist(),
                "pi": model.pi.tolist(),
                "research_only": True,
            },
            indent=2,
        )
    )
