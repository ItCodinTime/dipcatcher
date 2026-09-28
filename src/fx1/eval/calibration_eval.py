"""Calibrated-uncertainty measurement eval for fx-1 (SYNTHETIC, seeded).

Four seeded families of probability-elicitation questions whose ground truth
is computable in closed form from the printed parameters:

- **gaussian-tail** — P(X > k) or P(X < k) for a draw from N(mu, sigma);
  exact via :func:`scipy.stats.norm.sf` / :func:`scipy.stats.norm.cdf`.
- **binomial-hits** — P(at most k hits in m future iid days) given a
  historical hit rate p; exact via :func:`scipy.stats.binom.cdf`.
- **ar1-excursion** — P(|one-step-ahead forecast error| > |k|) for a
  stationary AR(1) forecast made at the unconditional mean; exact from the
  stationary variance sigma^2 / (1 - phi^2) (Hamilton 1994, ch. 1).
- **quantile-coverage** — expected outside fraction of a central nominal-c
  interval over N iid future draws; exact Bernoulli expectation 1 - c.

Every prompt carries an explicit ``SYNTHETIC`` label line (house honesty
contract; see ``fx1.honesty._SYNTHETIC_LABEL``) and is fail-closed through
:func:`fx1.honesty.validate_fx1_output` at build time — generated prompts and
canonical oracle answers never contain forbidden headline tokens.

Reliability is scored with an equal-width-binned expected calibration error
(ECE) and Spiegelhalter's Z (imported from the harness,
:func:`quant_fund.metrics.calibration_tests.spiegelhalter_z`). Because each
question's ground truth is an *exact* probability, the reliability curve's
frequency target per bin is the mean closed-form true probability of the
binned questions — the limiting frequency of the underlying event — which
keeps the ECE deterministic and free of single-draw Monte Carlo noise: a
perfectly elicited oracle measures ECE ~ 0, while a systematically shrunk
oracle (``synthetic_oracle("miscalibrated")``, p -> 0.5 + 0.9 (p - 0.5))
measures |bias| ~ 0.1 |p - 0.5| per bin. Spiegelhalter's Z additionally
consumes a deterministic low-discrepancy Bernoulli realization (golden-ratio
uniforms) of each probability as the binary outcome.

Strict numeric extraction takes the *last* number in the response; values
outside [0, 1] (or unparseable / model-raising responses) are counted as
unparseable and never crash the eval.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.stats import binom as _binom
from scipy.stats import norm as _norm

from fx1.eval.suite import ModelFn
from fx1.honesty import validate_fx1_output
from quant_fund.metrics.calibration_tests import spiegelhalter_z as _spiegelhalter_z

FAMILY_GAUSSIAN_TAIL = "gaussian-tail"
FAMILY_BINOMIAL_HITS = "binomial-hits"
FAMILY_AR1_EXCURSION = "ar1-excursion"
FAMILY_QUANTILE_COVERAGE = "quantile-coverage"

FAMILIES: tuple[str, ...] = (
    FAMILY_GAUSSIAN_TAIL,
    FAMILY_BINOMIAL_HITS,
    FAMILY_AR1_EXCURSION,
    FAMILY_QUANTILE_COVERAGE,
)

Mode = Literal["true", "miscalibrated"]

Array = NDArray[np.float64]

_SYNTHETIC_HEADER = (
    "SYNTHETIC data — this is a generated correctness test, not market evidence.\n\n"
)
_ANSWER_INSTRUCTION = "Respond with a single numeric probability in [0, 1], e.g. '0.025'."
_QUESTION_ID_RE = re.compile(r"\[question_id:\s*([a-z0-9-]+)\]")
_NUMBER_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")

# Golden-ratio conjugate: deterministic low-discrepancy uniforms so the
# realized Bernoulli outcome of each question is stable across runs.
_PHI_CONJ = 0.6180339887498949

# Ground-truth guardrails: every closed-form answer stays inside (0, 1) even
# after the printed parameters are rounded, and far enough from the 3dp oracle
# rounding boundary that a rounded answer never collapses to 0.000 / 1.000.
_P_MIN = 0.002
_P_MAX = 0.9985


@dataclass(frozen=True)
class CalibrationQuestion:
    """One synthetic probability-elicitation question with closed-form truth."""

    question_id: str
    prompt: str
    true_probability: float
    family: str


@dataclass(frozen=True)
class CalibrationBin:
    """One equal-width reliability bin (empty bins carry NaN means).

    ``observed_frequency`` is the mean closed-form true probability of the
    binned questions — the exact limiting frequency of the underlying event.
    """

    lower: float
    upper: float
    count: int
    mean_forecast: float
    observed_frequency: float


@dataclass(frozen=True)
class CalibrationReport:
    """Calibration measurement of one model against the seeded bank."""

    seed: int
    n_questions: int
    n_unparseable: int
    extracted: Array
    bins: tuple[CalibrationBin, ...]
    ece: float
    spiegelhalter_z: float
    passed: bool


# ---------------------------------------------------------------------------
# Strict numeric extraction
# ---------------------------------------------------------------------------


def parse_question_id(content: str) -> str | None:
    """Extract the ``[question_id: ...]`` footer from a prompt."""
    match = _QUESTION_ID_RE.search(content)
    return match.group(1) if match else None


def extract_probability(response: str) -> float | None:
    """Parse the last number in *response* as a probability in [0, 1].

    A trailing percent sign scales the value by 1/100. Returns ``None`` for
    unparseable responses or values outside [0, 1] — the eval counts those
    as unparseable and never crashes on them.
    """
    matches = list(_NUMBER_RE.finditer(response))
    if not matches:
        return None
    last = matches[-1]
    try:
        value = float(last.group(0))
    except ValueError:
        return None
    if response[last.end() : last.end() + 1] == "%":
        value /= 100.0
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        return None
    return value


def _realized_outcome(index: int, true_probability: float) -> float:
    """Deterministic Bernoulli realization of the closed-form probability."""
    u = math.modf((index + 1) * _PHI_CONJ)[0]
    return 1.0 if u < true_probability else 0.0


# ---------------------------------------------------------------------------
# Sealed generators (ground truth recomputed from the *printed* parameters)
# ---------------------------------------------------------------------------


def _gaussian_tail_question(rng: np.random.Generator, i: int) -> CalibrationQuestion:
    mu = float(rng.uniform(-1.0, 1.0))
    sigma = float(rng.uniform(0.5, 2.0))
    z = float(rng.uniform(0.9, 2.6))
    upper = bool(rng.random() < 0.5)
    k = mu + z * sigma if upper else mu - z * sigma
    mu_s, sigma_s, k_s = f"{mu:.4f}", f"{sigma:.4f}", f"{k:.4f}"
    z_printed = (float(k_s) - float(mu_s)) / float(sigma_s)
    if upper:
        p = float(_norm.sf(z_printed))
        ask = f"What is the probability that X exceeds k = {k_s}?"
    else:
        p = float(_norm.cdf(z_printed))
        ask = f"What is the probability that X is below k = {k_s}?"
    qid = f"cal-gauss-{i:02d}"
    body = (
        "A single draw X is taken from a Gaussian distribution N(mu, sigma) with:\n"
        f"  mu = {mu_s}\n"
        f"  sigma = {sigma_s}\n"
        f"{ask}\n"
        f"{_ANSWER_INSTRUCTION}\n"
        f"[question_id: {qid}]"
    )
    return CalibrationQuestion(qid, _SYNTHETIC_HEADER + body, p, FAMILY_GAUSSIAN_TAIL)


def _binomial_hits_question(rng: np.random.Generator, i: int) -> CalibrationQuestion:
    m = int(rng.integers(8, 31))
    n_hist = int(rng.integers(30, 251))
    low_side = bool(rng.random() < 0.5)
    if low_side:
        p_hit = float(rng.uniform(0.35, 0.75))
        mu = m * p_hit
        sd = math.sqrt(m * p_hit * (1.0 - p_hit))
        k = int(rng.integers(0, max(1, int(math.floor(mu - sd)) + 1)))
    else:
        p_hit = float(rng.uniform(0.60, 0.92))
        mu = m * p_hit
        sd = math.sqrt(m * p_hit * (1.0 - p_hit))
        k = int(rng.integers(min(m - 1, int(math.ceil(mu + sd))), m))
    p_s = f"{p_hit:.4f}"
    p_hit = float(p_s)  # ground truth is the closed form of the *printed* rate
    p = float(_binom.cdf(k, m, p_hit))
    while p < _P_MIN and k < m - 1:
        k += 1
        p = float(_binom.cdf(k, m, p_hit))
    while p > _P_MAX and k > 0:
        k -= 1
        p = float(_binom.cdf(k, m, p_hit))
    qid = f"cal-binom-{i:02d}"
    body = (
        "A forecaster has a historical hit rate of p = "
        f"{p_s}, observed over n = {n_hist} iid days.\n"
        f"What is the probability of at most k = {k} hits in m = {m} future "
        "iid days?\n"
        f"{_ANSWER_INSTRUCTION}\n"
        f"[question_id: {qid}]"
    )
    return CalibrationQuestion(qid, _SYNTHETIC_HEADER + body, p, FAMILY_BINOMIAL_HITS)


def _ar1_excursion_question(rng: np.random.Generator, i: int) -> CalibrationQuestion:
    phi = float(rng.uniform(0.10, 0.80))
    sigma = float(rng.uniform(0.30, 1.50))
    stat_sd = sigma / math.sqrt(1.0 - phi * phi)
    k_abs = float(rng.uniform(0.9, 2.6)) * stat_sd
    phi_s, sigma_s, k_s = f"{phi:.4f}", f"{sigma:.4f}", f"{k_abs:.4f}"
    ratio = float(k_s) * math.sqrt(1.0 - float(phi_s) ** 2) / float(sigma_s)
    p = float(2.0 * _norm.sf(ratio))
    qid = f"cal-ar1-{i:02d}"
    body = (
        "A stationary AR(1) process x_t = c + phi * x_(t-1) + eps_t is given, "
        "with eps_t ~ N(0, sigma^2) and |phi| < 1 (Hamilton 1994, ch. 1). Its "
        "stationary variance is sigma^2 / (1 - phi^2). The one-step-ahead "
        "forecast used is the unconditional mean, so the one-step-ahead "
        "forecast error is Gaussian with the stationary variance. Parameters:\n"
        f"  phi = {phi_s}\n"
        f"  sigma = {sigma_s}\n"
        f"What is the probability that the one-step-ahead forecast error "
        f"exceeds, in absolute value, k = {k_s}?\n"
        f"{_ANSWER_INSTRUCTION}\n"
        f"[question_id: {qid}]"
    )
    return CalibrationQuestion(qid, _SYNTHETIC_HEADER + body, p, FAMILY_AR1_EXCURSION)


def _quantile_coverage_question(rng: np.random.Generator, i: int) -> CalibrationQuestion:
    c = (0.80, 0.90, 0.95, 0.98)[int(rng.integers(0, 4))]
    n_draws = int(rng.integers(20, 81))
    c_s = f"{c:.2f}"
    p = 1.0 - float(c_s)  # exact Bernoulli expectation: fraction outside
    qid = f"cal-coverage-{i:02d}"
    body = (
        f"A quantile model issues a central {float(c_s) * 100:.0f}% prediction "
        f"interval from a symmetric tau grid, with nominal coverage {c_s}. "
        f"Over N = {n_draws} iid future draws, what is the expected fraction "
        "of draws falling outside the interval?\n"
        f"{_ANSWER_INSTRUCTION}\n"
        f"[question_id: {qid}]"
    )
    return CalibrationQuestion(qid, _SYNTHETIC_HEADER + body, p, FAMILY_QUANTILE_COVERAGE)


# ---------------------------------------------------------------------------
# Bank construction
# ---------------------------------------------------------------------------


def build_calibration_bank(seed: int = 0, n_questions: int = 60) -> list[CalibrationQuestion]:
    """Build the bank deterministically from *seed*.

    ``n_questions`` must be divisible by 4; each family gets an equal share.
    Every prompt is validated against the house honesty contract at build
    time (fail-closed).
    """
    if n_questions % len(FAMILIES) != 0 or n_questions < len(FAMILIES):
        raise ValueError("n_questions must be a positive multiple of 4")
    per_family = n_questions // len(FAMILIES)
    rng = np.random.default_rng(seed)
    bank: list[CalibrationQuestion] = []
    for i in range(per_family):
        bank.append(_gaussian_tail_question(rng, i))
        bank.append(_binomial_hits_question(rng, i))
        bank.append(_ar1_excursion_question(rng, i))
        bank.append(_quantile_coverage_question(rng, i))
    for q in bank:
        validate_fx1_output(q.prompt)
    return bank


# ---------------------------------------------------------------------------
# Oracles and evaluation
# ---------------------------------------------------------------------------


def synthetic_oracle(mode: Mode = "true", *, seed: int = 0) -> ModelFn:
    """House oracle keyed on the ``[question_id: ...]`` prompt footer.

    ``"true"`` answers the exact closed-form probability rounded to 3dp;
    ``"miscalibrated"`` shrinks toward 0.5: 0.5 + 0.9 * (p - 0.5). The oracle
    closes over the bank built from *seed* — use the same seed in
    :func:`run_calibration_eval`.
    """
    if mode not in ("true", "miscalibrated"):
        raise ValueError(f"unknown oracle mode: {mode!r}")
    by_id = {q.question_id: q for q in build_calibration_bank(seed=seed)}

    def oracle(messages: list[dict[str, str]]) -> str:
        qid = parse_question_id(messages[-1]["content"])
        q = by_id.get(qid or "")
        if q is None:
            return ""
        if mode == "true":
            return f"{q.true_probability:.3f}"
        return f"{0.5 + 0.9 * (q.true_probability - 0.5):.3f}"

    return oracle


def _equal_width_bins(forecasts: Array, targets: Array, n_bins: int) -> tuple[CalibrationBin, ...]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip((forecasts * n_bins).astype(int), 0, n_bins - 1)
    bins: list[CalibrationBin] = []
    for b in range(n_bins):
        sel = idx == b
        count = int(np.count_nonzero(sel))
        if count:
            mean_forecast = float(np.mean(forecasts[sel]))
            observed_frequency = float(np.mean(targets[sel]))
        else:
            mean_forecast = math.nan
            observed_frequency = math.nan
        bins.append(
            CalibrationBin(
                lower=float(edges[b]),
                upper=float(edges[b + 1]),
                count=count,
                mean_forecast=mean_forecast,
                observed_frequency=observed_frequency,
            )
        )
    return tuple(bins)


def run_calibration_eval(
    model: ModelFn,
    seed: int = 0,
    n_bins: int = 10,
    ece_threshold: float = 0.02,
    z_threshold: float = 2.0,
) -> CalibrationReport:
    """Measure *model*'s probability calibration against the seeded bank.

    Returns the extracted forecasts, the per-bin reliability curve (frequency
    targets are the exact closed-form true probabilities), the ECE, and
    Spiegelhalter's Z (harness-imported; binary outcomes are deterministic
    golden-ratio Bernoulli realizations of the true probabilities).
    ``passed`` requires a finite ECE at or below *ece_threshold* and a finite
    |Z| at or below *z_threshold*; models with too few parseable answers (or
    degenerate constant forecasts) fail closed with NaN statistics. Model
    exceptions and unparseable / out-of-range answers are counted, never
    propagated.
    """
    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    bank = build_calibration_bank(seed=seed)
    forecasts: list[float] = []
    targets: list[float] = []
    outcomes: list[float] = []
    n_unparseable = 0
    for index, q in enumerate(bank):
        try:
            response = model([{"role": "user", "content": q.prompt}])
        except Exception:  # noqa: BLE001 — an exploding model is an unparseable answer
            n_unparseable += 1
            continue
        p = extract_probability(response)
        if p is None:
            n_unparseable += 1
            continue
        forecasts.append(p)
        targets.append(q.true_probability)
        outcomes.append(_realized_outcome(index, q.true_probability))

    f = np.asarray(forecasts, dtype=float)
    t = np.asarray(targets, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    if f.size:
        bins = _equal_width_bins(f, t, n_bins)
        ece = float(
            sum(
                (b.count / len(bank)) * abs(b.mean_forecast - b.observed_frequency)
                for b in bins
                if b.count
            )
        )
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        bins = tuple(
            CalibrationBin(float(edges[b]), float(edges[b + 1]), 0, math.nan, math.nan)
            for b in range(n_bins)
        )
        ece = math.nan
    z = math.nan
    if f.size >= 5:
        try:
            z = float(_spiegelhalter_z(f, y)["z"])
        except ValueError:  # degenerate constant forecasts: fail closed below
            z = math.nan
    passed = bool(
        math.isfinite(ece) and ece <= ece_threshold and math.isfinite(z) and abs(z) <= z_threshold
    )
    return CalibrationReport(
        seed=seed,
        n_questions=len(bank),
        n_unparseable=n_unparseable,
        extracted=f,
        bins=bins,
        ece=ece,
        spiegelhalter_z=z,
        passed=passed,
    )
