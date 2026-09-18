"""Authoritative catalog of Dipcatcher research benchmark families.

Dual honesty catalogs (research scorecard vs paper analytics export)
-------------------------------------------------------------------
- **Research family / scorecard blobs** (bench outputs, H-table payloads):
  ``FORBIDDEN_RESEARCH_METRIC_KEYS`` / ``family_blob_forbidden_metrics_absent``
  enforce headline hygiene — no sharpe / sortino / calmar / pnl / nav *key tokens*.
- **Paper/backtest ``analytics_export``** (see ``metrics.analytics``): equity /
  stress diagnostics may intentionally carry ``pnl`` / ``nav`` keys, but
  ``live_pnl_claim`` must stay false and ``validate_analytics_export`` fails closed
  on ``live_pnl_claim=true``. Do **not** apply the research forbidden-key scanner
  to the full export blob — different catalogs, different contracts.
"""

from __future__ import annotations

import math

BENCHMARK_CATALOG_VERSION = 2
RESEARCH_RECEIPT_SCHEMA_VERSION = 1

REQUIRED_BENCHMARK_FAMILIES = frozenset(
    {
        "ranking",
        "alpha",
        "volatility",
        "distribution",
        "regime",
        "tail",
        "drawdown",
        "liquidity",
        "reinforcement",
        "conformal",
        "evalues",
        "jackknife_plus",
        "crc",
        "weighted_conformal",
        "interval_risk",
        "quantile_bandit",
        "cv_plus",
        "cpcv",
        "localized_conformal",
        "conformal_rank",
        "online_crc",
        "portfolio_conformal",
        "northset",
    }
)

# Optional families may appear in a receipt (and are then fully soft-verified)
# but are never required by the benchmark catalog. ``candle_order_book`` is the
# optional candlestick+L2 family emitted by the ``candle-book`` CLI; without
# this allow-list the verify.py candle honesty dispatch could never run.
OPTIONAL_BENCHMARK_FAMILIES = frozenset({"candle_order_book"})

BENCHMARK_FAMILY_ORDER = (
    "ranking",
    "alpha",
    "volatility",
    "distribution",
    "regime",
    "tail",
    "drawdown",
    "liquidity",
    "reinforcement",
    "conformal",
    "evalues",
    "jackknife_plus",
    "crc",
    "weighted_conformal",
    "interval_risk",
    "quantile_bandit",
    "cv_plus",
    "cpcv",
    "localized_conformal",
    "conformal_rank",
    "online_crc",
    "portfolio_conformal",
    "northset",
)

# Headline P&L / ratio key *tokens* that must never appear in research family /
# scorecard blobs (not the paper analytics_export schema — that catalog allows
# equity/stress pnl/nav diagnostics under live_pnl_claim=false).
# Matching is by underscore-token on mapping keys (case-insensitive), so
# ``flag_high_sharpe`` / ``shock_down_pnl`` / ``calmar_diagnostic`` all fail closed.
FORBIDDEN_RESEARCH_METRIC_KEYS = frozenset(
    {
        "sharpe",
        "sortino",
        "calmar",
        "pnl",
        "nav",
    }
)


def _iter_mapping_keys(obj: object) -> list[str]:
    """Collect nested mapping keys (dicts only; list elements walked)."""
    keys: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            keys.append(str(k))
            keys.extend(_iter_mapping_keys(v))
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            keys.extend(_iter_mapping_keys(item))
    return keys


def family_blob_forbidden_metrics_absent(payload: object) -> bool:
    """Return True iff *payload* has no forbidden research-headline metric keys.

    Fail closed: any mapping key whose underscore tokens include sharpe / sortino /
    calmar / pnl / nav marks the blob unclean. Values are not scanned (keys only).

    Scope: research family / scorecard blobs only. Paper ``analytics_export`` may
    contain equity ``nav_*`` / stress ``*_pnl`` diagnostics; validate those with
    ``validate_analytics_export`` (live_pnl_claim fail-closed), not this helper.
    """
    for key in _iter_mapping_keys(payload):
        parts = str(key).lower().replace("-", "_").split("_")
        if any(tok in FORBIDDEN_RESEARCH_METRIC_KEYS for tok in parts if tok):
            return False
    return True


def family_blob_executed(payload: object) -> bool:
    """True iff family payload is truthy (scorecard ``executed`` twin)."""
    return bool(payload)


def family_blob_nonempty(payload: object) -> bool:
    """True iff family payload is truthy (scorecard ``nonempty`` twin)."""
    return bool(payload)


def family_blob_has_finite_observation(payload: object) -> bool:
    """True iff *payload* recursively contains at least one finite observation.

    Lifted from research.agent ``_benchmark_scorecard`` nested ``has_observation``
    (Day Wave 44). Semantics:
    - dict / list / tuple → any child is an observation
    - bool → True
    - int (incl. numpy integer scalars via ``.item()``) → True
    - float (incl. numpy floating) → ``math.isfinite``
    - str → nonempty strip and not ``nan`` / ``none`` (case-insensitive)
    - else → False

    Empty ``{}`` / all-NaN blobs → False. Used by soft scorecard forge verify.
    """
    if isinstance(payload, dict):
        return any(family_blob_has_finite_observation(item) for item in payload.values())
    if isinstance(payload, (list, tuple)):
        return any(family_blob_has_finite_observation(item) for item in payload)
    if isinstance(payload, bool):
        return True
    if isinstance(payload, int):
        return True
    if isinstance(payload, float):
        return math.isfinite(payload)
    # Numpy scalars are not always subclasses of int/float — recurse via .item().
    if (
        hasattr(payload, "dtype")
        and hasattr(payload, "item")
        and not isinstance(payload, (bytes, bytearray, memoryview))
    ):
        try:
            return family_blob_has_finite_observation(payload.item())
        except (ValueError, TypeError, AttributeError):
            return False
    return (
        isinstance(payload, str)
        and bool(payload.strip())
        and payload.lower() not in {"nan", "none"}
    )


# Soft VaR-battery key hygiene (Day Wave 18 research diagnostic).
# When a nonempty tail / bench_tail blob exposes Kupiec keys (kupiec_p or
# kupiec_lr), Christoffersen CC (+ preferred ind) keys must also be present so
# Day Wave 16 cannot silently regress. Values may be honest NaN — presence only.
# Not a live capital / promotion gate; research receipt verify fail-closed on
# missing key *presence* only.
KUPIEC_MARKER_KEYS = frozenset({"kupiec_p", "kupiec_lr"})
TAIL_VAR_BATTERY_REQUIRED_WHEN_KUPIEC = (
    "christoffersen_cc_p",
    "christoffersen_cc_lr",
    "christoffersen_ind_p",
    "christoffersen_ind_lr",
)
REQUIRED_CHRISTOFFERSEN_CC_KEYS = frozenset(
    {
        "christoffersen_cc_p",
        "christoffersen_cc_lr",
    }
)
PREFERRED_CHRISTOFFERSEN_IND_KEYS = frozenset(
    {
        "christoffersen_ind_p",
        "christoffersen_ind_lr",
    }
)


def tail_var_battery_missing_keys(payload: object) -> list[str]:
    """Return missing Christoffersen keys when a nonempty blob has Kupiec markers.

    Contract (Day Wave 18 soft honesty — research diagnostic only):
    - Empty ``{}`` or non-dict → no required keys (tiny panels may omit battery).
    - Nonempty dict without ``kupiec_p`` / ``kupiec_lr`` → skip (no Kupiec ⇒ no
      battery contract).
    - Nonempty dict with either Kupiec marker (even NaN) → require CC + ind key
      *presence*; values may be NaN. Missing keys returned in catalog order for
      ``tail_var_battery_incomplete:<key>`` errors in ``verify_research_artifact``.
    """
    if not isinstance(payload, dict) or not payload:
        return []
    keys = {str(k) for k in payload}
    if not (keys & KUPIEC_MARKER_KEYS):
        return []
    return [key for key in TAIL_VAR_BATTERY_REQUIRED_WHEN_KUPIEC if key not in keys]


def tail_var_battery_keys_present(payload: object) -> bool:
    """Return True iff soft VaR-battery key presence holds (see missing_keys)."""
    return not tail_var_battery_missing_keys(payload)


# Soft distribution CRPS e-process key hygiene (Day Wave 20 research diagnostic).
# When a nonempty distribution blob exposes DM CRPS markers (``dm_crps_p`` and/or
# ``dm_crps_scaled_p``), the matching Day Wave 18/19 ``e_dm_crps_*`` companion keys
# must also be present so e-process wiring cannot silently regress. Values may be
# honest NaN (or False/0 sentinels on e-process failure) — presence only.
# Not a live capital / promotion gate; research receipt verify fail-closed on
# missing key *presence* only. Empty ``{}`` and blobs without DM markers skip.
DM_CRPS_MARKER_KEY = "dm_crps_p"
DM_CRPS_SCALED_MARKER_KEY = "dm_crps_scaled_p"
DIST_CRPS_EPROCESS_REQUIRED_WHEN_DM = (
    "e_dm_crps_final",
    "e_dm_crps_reject",
    "e_dm_crps_n",
)
DIST_CRPS_SCALED_EPROCESS_REQUIRED_WHEN_DM = (
    "e_dm_crps_scaled_final",
    "e_dm_crps_scaled_reject",
    "e_dm_crps_scaled_n",
)


def dist_crps_eprocess_missing_keys(payload: object) -> list[str]:
    """Return missing e_dm_crps_* keys when a nonempty blob has DM CRPS markers.

    Contract (Day Wave 20 soft honesty — research diagnostic only):
    - Empty ``{}`` or non-dict → no required keys.
    - Nonempty dict without ``dm_crps_p`` / ``dm_crps_scaled_p`` → skip.
    - ``dm_crps_p`` present (even NaN) → require unscaled ``e_dm_crps_*`` presence.
    - ``dm_crps_scaled_p`` present (even NaN) → require scaled ``e_dm_crps_scaled_*``.
    - Values may be NaN / False / 0; missing keys returned in catalog order for
      ``dist_crps_eprocess_incomplete:<key>`` errors in ``verify_research_artifact``.
    """
    if not isinstance(payload, dict) or not payload:
        return []
    keys = {str(k) for k in payload}
    missing: list[str] = []
    if DM_CRPS_MARKER_KEY in keys:
        missing.extend(key for key in DIST_CRPS_EPROCESS_REQUIRED_WHEN_DM if key not in keys)
    if DM_CRPS_SCALED_MARKER_KEY in keys:
        missing.extend(key for key in DIST_CRPS_SCALED_EPROCESS_REQUIRED_WHEN_DM if key not in keys)
    return missing


def dist_crps_eprocess_keys_present(payload: object) -> bool:
    """Return True iff soft distribution CRPS e-process key presence holds."""
    return not dist_crps_eprocess_missing_keys(payload)


# Soft ES-battery key hygiene (Day Wave 21 research diagnostic).
# When a nonempty tail / bench_tail blob exposes ES/VaR forecast markers
# (``es_95`` OR ``realized_es`` OR ``var_95``), Acerbi–Székely Z1/Z2 + mean
# Fissler–Ziegel + ``es_hit_count`` keys must also be present so Day Wave 15
# cannot silently regress. Values may be honest NaN — presence only.
# Orthogonal to VaR-battery (Kupiec ⇒ Christoffersen): ES markers ⇒ Acerbi/FZ.
# Not a live capital / promotion gate; research receipt verify fail-closed on
# missing key *presence* only. Empty ``{}`` skips.
ES_MARKER_KEYS = frozenset({"es_95", "realized_es", "var_95"})
TAIL_ES_BATTERY_REQUIRED_WHEN_ES_MARKERS = (
    "acerbi_szekely_z1",
    "acerbi_szekely_z2",
    "fissler_ziegel_mean",
    "es_hit_count",
)
# Back-compat alias (Wave20 Kupiec-gate name); prefer WHEN_ES_MARKERS.


def tail_es_battery_missing_keys(payload: object) -> list[str]:
    """Return missing Acerbi/FZ keys when a nonempty blob has ES/VaR markers.

    Contract (Day Wave 21 soft honesty — research diagnostic only):
    - Empty ``{}`` or non-dict → no required keys (tiny panels may omit battery).
    - Nonempty dict without ``es_95`` / ``realized_es`` / ``var_95`` → skip
      (no ES/VaR forecast markers ⇒ no ES-battery contract).
    - Nonempty dict with any ES marker (even NaN) → require Acerbi Z1/Z2 +
      ``fissler_ziegel_mean`` + ``es_hit_count`` key *presence*; values may be
      NaN. Missing keys returned in catalog order for
      ``tail_es_battery_incomplete:<key>`` errors in ``verify_research_artifact``.
    - Does **not** double-require VaR-battery / Christoffersen keys — keep
      orthogonal (Kupiec ⇒ Christoffersen; ES markers ⇒ Acerbi/FZ). When both
      Kupiec and ES markers are present, both batteries apply independently.
    """
    if not isinstance(payload, dict) or not payload:
        return []
    keys = {str(k) for k in payload}
    if not (keys & ES_MARKER_KEYS):
        return []
    return [key for key in TAIL_ES_BATTERY_REQUIRED_WHEN_ES_MARKERS if key not in keys]


def tail_es_battery_keys_present(payload: object) -> bool:
    """Return True iff soft ES-battery key presence holds (see missing_keys)."""
    return not tail_es_battery_missing_keys(payload)


# Soft H4b H-table consistency (Day Wave 26 research diagnostic).
# When ``families["tail"]`` exposes a *finite* ``christoffersen_cc_p``, the
# notebook must mint ``H4b_var_christoffersen_cc`` (calibration) — same gate as
# Day Wave 17 ``_finite_number`` mint. Non-finite / missing cc_p → skip.
# Research-receipt fail-closed; not a live capital / promotion gate.
H4B_HYPOTHESIS_ID = "H4b_var_christoffersen_cc"
H4B_EXPECTED_FAMILY = "calibration"


def _finite_scalar(value: object) -> bool:
    """Return True iff *value* is a finite real number (bools rejected)."""
    if value is None or isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    number = float(value)
    return math.isfinite(number)


def tail_has_finite_christoffersen_cc_p(payload: object) -> bool:
    """True iff *payload* is a dict with finite ``christoffersen_cc_p``."""
    if not isinstance(payload, dict):
        return False
    return _finite_scalar(payload.get("christoffersen_cc_p"))


def hypotheses_include_h4b(hypotheses: object, *, require_calibration: bool = True) -> bool:
    """True iff *hypotheses* contains ``H4b_var_christoffersen_cc``.

    When *require_calibration* is True (default), the matching row must also
    have ``family == "calibration"`` (Day Wave 17 mint contract).
    """
    if not isinstance(hypotheses, list):
        return False
    for hyp in hypotheses:
        if not isinstance(hyp, dict):
            continue
        if hyp.get("id") != H4B_HYPOTHESIS_ID:
            continue
        return not (require_calibration and hyp.get("family") != H4B_EXPECTED_FAMILY)
    return False


def h4b_hypothesis_consistency_errors(tail: object, hypotheses: object) -> list[str]:
    """Return soft-verify errors for H4b notebook consistency (Day Wave 26).

    Contract (research diagnostic only):
    - Non-dict / missing / non-finite ``christoffersen_cc_p`` → no errors (skip).
    - Finite ``christoffersen_cc_p`` without ``H4b_var_christoffersen_cc``
      (calibration) → ``hypothesis_h4b_missing_despite_finite_christoffersen_cc_p``.
    """
    if not tail_has_finite_christoffersen_cc_p(tail):
        return []
    if hypotheses_include_h4b(hypotheses, require_calibration=True):
        return []
    return ["hypothesis_h4b_missing_despite_finite_christoffersen_cc_p"]


# Soft H4 Kupiec H-table consistency (Day Wave 29 research diagnostic).
# When ``families["tail"]`` exposes a *finite* ``kupiec_p``, the notebook must
# mint ``H4_var_kupiec`` (calibration) — same gate as agent ``_finite_number``
# mint. Non-finite / missing kupiec_p → skip.
# Research-receipt fail-closed; not a live capital / promotion gate.
H4_HYPOTHESIS_ID = "H4_var_kupiec"
H4_EXPECTED_FAMILY = "calibration"


def tail_has_finite_kupiec_p(payload: object) -> bool:
    """True iff *payload* is a dict with finite ``kupiec_p``."""
    if not isinstance(payload, dict):
        return False
    return _finite_scalar(payload.get("kupiec_p"))


def hypotheses_include_h4(hypotheses: object, *, require_calibration: bool = True) -> bool:
    """True iff *hypotheses* contains ``H4_var_kupiec``.

    When *require_calibration* is True (default), the matching row must also
    have ``family == "calibration"`` (agent mint contract).
    """
    if not isinstance(hypotheses, list):
        return False
    for hyp in hypotheses:
        if not isinstance(hyp, dict):
            continue
        if hyp.get("id") != H4_HYPOTHESIS_ID:
            continue
        return not (require_calibration and hyp.get("family") != H4_EXPECTED_FAMILY)
    return False


def h4_hypothesis_consistency_errors(tail: object, hypotheses: object) -> list[str]:
    """Return soft-verify errors for H4 notebook consistency (Day Wave 29).

    Contract (research diagnostic only):
    - Non-dict / missing / non-finite ``kupiec_p`` → no errors (skip).
    - Finite ``kupiec_p`` without ``H4_var_kupiec``
      (calibration) → ``hypothesis_h4_missing_despite_finite_kupiec_p``.
    """
    if not tail_has_finite_kupiec_p(tail):
        return []
    if hypotheses_include_h4(hypotheses, require_calibration=True):
        return []
    return ["hypothesis_h4_missing_despite_finite_kupiec_p"]


# Soft H3 vol-DM H-table consistency (Day Wave 30 research diagnostic).
# When ``families["volatility"]`` exposes a *finite* ``dm_p``, the notebook must
# mint ``H3_vol_dm`` (discovery) — same gate as agent ``_finite_number`` mint.
# Non-finite / missing dm_p → skip.
# Research-receipt fail-closed; not a live capital / promotion gate.
H3_HYPOTHESIS_ID = "H3_vol_dm"
H3_EXPECTED_FAMILY = "discovery"


def volatility_has_finite_dm_p(payload: object) -> bool:
    """True iff *payload* is a dict with finite ``dm_p``."""
    if not isinstance(payload, dict):
        return False
    return _finite_scalar(payload.get("dm_p"))


def hypotheses_include_h3(hypotheses: object, *, require_discovery: bool = True) -> bool:
    """True iff *hypotheses* contains ``H3_vol_dm``.

    When *require_discovery* is True (default), the matching row must also
    have ``family == "discovery"`` (agent mint contract).
    """
    if not isinstance(hypotheses, list):
        return False
    for hyp in hypotheses:
        if not isinstance(hyp, dict):
            continue
        if hyp.get("id") != H3_HYPOTHESIS_ID:
            continue
        return not (require_discovery and hyp.get("family") != H3_EXPECTED_FAMILY)
    return False


def h3_hypothesis_consistency_errors(volatility: object, hypotheses: object) -> list[str]:
    """Return soft-verify errors for H3 notebook consistency (Day Wave 30).

    Contract (research diagnostic only):
    - Non-dict / missing / non-finite ``dm_p`` → no errors (skip).
    - Finite ``dm_p`` without ``H3_vol_dm``
      (discovery) → ``hypothesis_h3_missing_despite_finite_dm_p``.
    """
    if not volatility_has_finite_dm_p(volatility):
        return []
    if hypotheses_include_h3(hypotheses, require_discovery=True):
        return []
    return ["hypothesis_h3_missing_despite_finite_dm_p"]


# Soft H1/H2 oracle ranking H-table consistency (Day Wave 31 research diagnostic).
# Oracle lives in ``notebook["rankers"]`` (name ``oracle_raw``), NOT families.
# Skip ranker names starting with ``_`` (agent model_rankers filter).
# Finite ``p_ic`` → require ``H1_ranking_oracle`` (discovery); finite ``ls_p`` →
# require ``H2_decile_mono`` (discovery). Gates are independent (mirror agent mint).
# Non-finite / missing / no oracle_raw → skip each gate independently.
# Research-receipt fail-closed; not a live capital / promotion gate.
H1_HYPOTHESIS_ID = "H1_ranking_oracle"
H1_EXPECTED_FAMILY = "discovery"
H2_HYPOTHESIS_ID = "H2_decile_mono"
H2_EXPECTED_FAMILY = "discovery"


def rankers_oracle_raw(rankers: object) -> dict | None:
    """Return the ``oracle_raw`` ranker dict from *rankers*, or None.

    Skips non-dicts and names starting with ``_`` (same filter as agent
    ``model_rankers``). First matching ``name == "oracle_raw"`` wins.
    """
    if not isinstance(rankers, list):
        return None
    for row in rankers:
        if not isinstance(row, dict):
            continue
        name = row.get("name")
        if not isinstance(name, str) or name.startswith("_"):
            continue
        if name == "oracle_raw":
            return row
    return None


def oracle_has_finite_p_ic(payload: object) -> bool:
    """True iff *payload* is a dict with finite ``p_ic`` (oracle_raw row)."""
    if not isinstance(payload, dict):
        return False
    return _finite_scalar(payload.get("p_ic"))


def oracle_has_finite_ls_p(payload: object) -> bool:
    """True iff *payload* is a dict with finite ``ls_p`` (oracle_raw row)."""
    if not isinstance(payload, dict):
        return False
    return _finite_scalar(payload.get("ls_p"))


def hypotheses_include_h1(hypotheses: object, *, require_discovery: bool = True) -> bool:
    """True iff *hypotheses* contains ``H1_ranking_oracle``.

    When *require_discovery* is True (default), the matching row must also
    have ``family == "discovery"`` (agent mint contract).
    """
    if not isinstance(hypotheses, list):
        return False
    for hyp in hypotheses:
        if not isinstance(hyp, dict):
            continue
        if hyp.get("id") != H1_HYPOTHESIS_ID:
            continue
        return not (require_discovery and hyp.get("family") != H1_EXPECTED_FAMILY)
    return False


def hypotheses_include_h2(hypotheses: object, *, require_discovery: bool = True) -> bool:
    """True iff *hypotheses* contains ``H2_decile_mono``.

    When *require_discovery* is True (default), the matching row must also
    have ``family == "discovery"`` (agent mint contract).
    """
    if not isinstance(hypotheses, list):
        return False
    for hyp in hypotheses:
        if not isinstance(hyp, dict):
            continue
        if hyp.get("id") != H2_HYPOTHESIS_ID:
            continue
        return not (require_discovery and hyp.get("family") != H2_EXPECTED_FAMILY)
    return False


def h1_hypothesis_consistency_errors(rankers: object, hypotheses: object) -> list[str]:
    """Return soft-verify errors for H1 notebook consistency (Day Wave 31).

    Contract (research diagnostic only):
    - No ``oracle_raw`` / non-dict / missing / non-finite ``p_ic`` → no errors (skip).
    - Finite ``p_ic`` without ``H1_ranking_oracle``
      (discovery) → ``hypothesis_h1_missing_despite_finite_p_ic``.
    """
    oracle = rankers_oracle_raw(rankers)
    if oracle is None or not oracle_has_finite_p_ic(oracle):
        return []
    if hypotheses_include_h1(hypotheses, require_discovery=True):
        return []
    return ["hypothesis_h1_missing_despite_finite_p_ic"]


def h2_hypothesis_consistency_errors(rankers: object, hypotheses: object) -> list[str]:
    """Return soft-verify errors for H2 notebook consistency (Day Wave 31).

    Contract (research diagnostic only):
    - No ``oracle_raw`` / non-dict / missing / non-finite ``ls_p`` → no errors (skip).
    - Finite ``ls_p`` without ``H2_decile_mono``
      (discovery) → ``hypothesis_h2_missing_despite_finite_ls_p``.
    """
    oracle = rankers_oracle_raw(rankers)
    if oracle is None or not oracle_has_finite_ls_p(oracle):
        return []
    if hypotheses_include_h2(hypotheses, require_discovery=True):
        return []
    return ["hypothesis_h2_missing_despite_finite_ls_p"]


# Data-snooping battery over the ranker universe (Reality Check / SPA / StepM /
# MCS). When ``families["ranking"]["data_snooping"]`` exposes a *finite*
# consistent SPA p-value, the notebook must mint ``H46_ranking_data_snooping``
# (discovery) — same gate as the agent ``_finite_number`` mint. Non-finite /
# missing / no block → skip. Research-receipt fail-closed; not a live capital
# gate and never a live Sharpe claim.
H99_HYPOTHESIS_ID = "H99_ranking_data_snooping"
H99_EXPECTED_FAMILY = "discovery"
_DATA_SNOOPING_P_KEYS = ("reality_check_p", "spa_p_lower", "spa_p_consistent", "spa_p_upper")


def ranking_data_snooping_blob(ranking: object) -> dict | None:
    """Return the nested ``data_snooping`` dict from a ranking family blob."""
    if not isinstance(ranking, dict):
        return None
    blob = ranking.get("data_snooping")
    return blob if isinstance(blob, dict) and blob else None


def data_snooping_has_finite_spa_p(blob: object) -> bool:
    """True iff the data-snooping blob exposes a finite consistent SPA p-value."""
    if not isinstance(blob, dict):
        return False
    return _finite_scalar(blob.get("spa_p_consistent"))


def hypotheses_include_h99(hypotheses: object, *, require_discovery: bool = True) -> bool:
    """True iff *hypotheses* contains ``H46_ranking_data_snooping`` (discovery)."""
    if not isinstance(hypotheses, list):
        return False
    for hyp in hypotheses:
        if not isinstance(hyp, dict):
            continue
        if hyp.get("id") != H99_HYPOTHESIS_ID:
            continue
        return not (require_discovery and hyp.get("family") != H99_EXPECTED_FAMILY)
    return False


def h99_hypothesis_consistency_errors(ranking: object, hypotheses: object) -> list[str]:
    """Return soft-verify errors for H46 data-snooping notebook consistency.

    Contract (research diagnostic only):
    - No ``data_snooping`` block / non-dict / non-finite ``spa_p_consistent``
      → no errors (skip).
    - Finite consistent SPA p without ``H46_ranking_data_snooping`` (discovery)
      → ``hypothesis_h99_missing_despite_finite_spa_p``.
    """
    blob = ranking_data_snooping_blob(ranking)
    if blob is None or not data_snooping_has_finite_spa_p(blob):
        return []
    if hypotheses_include_h99(hypotheses, require_discovery=True):
        return []
    return ["hypothesis_h99_missing_despite_finite_spa_p"]


def ranking_data_snooping_honesty_errors(ranking: object) -> list[str]:
    """Soft-verify nested ``ranking.data_snooping`` p-values and ordering.

    When the ranking family carries a data-snooping block:

    - ``research_only is True`` and ``claim == "research_diagnostic_only"``
    - Reality Check / SPA p-values ∈ [0, 1] when finite
    - SPA ordering ``p_lower <= p_consistent <= p_upper`` when all finite
    - ``n_trials`` / ``n_obs`` positive; ``stepm_n_rejected == len(rejected)``;
      ``mcs_n_included == len(included)``
    - per-trial adjusted / MCS p-values ∈ [0, 1] when finite

    Missing block / non-dict → skip. Research diagnostic only; never a live
    Sharpe / promotion gate.
    """
    blob = ranking_data_snooping_blob(ranking)
    if blob is None:
        return []
    errs: list[str] = []
    if blob.get("research_only") is not True:
        errs.append("data_snooping_research_only_missing_or_false")
    if blob.get("claim") != "research_diagnostic_only":
        errs.append("data_snooping_claim_not_research_diagnostic_only")
    for key in _DATA_SNOOPING_P_KEYS:
        val = blob.get(key)
        if val is None:
            continue
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            errs.append(f"data_snooping_{key}_non_numeric")
            continue
        x = float(val)
        if x != x:
            continue
        if not 0.0 <= x <= 1.0:
            errs.append(f"data_snooping_{key}_out_of_unit_interval")
    # No universal ordering is asserted between the three SPA variants (see
    # SpaResult): each is range-checked above; the consistent variant is the
    # recommended one but can sit below the all-recentered lower variant.
    for key in ("n_trials", "n_obs"):
        val = blob.get(key)
        if val is None:
            continue
        if isinstance(val, bool) or not isinstance(val, (int, float)) or float(val) < 1:
            errs.append(f"data_snooping_{key}_invalid")
    rejected = blob.get("stepm_rejected")
    n_rejected = blob.get("stepm_n_rejected")
    if (
        isinstance(rejected, list)
        and isinstance(n_rejected, (int, float))
        and not isinstance(n_rejected, bool)
        and int(n_rejected) != len(rejected)
    ):
        errs.append("data_snooping_stepm_n_rejected_mismatch")
    included = blob.get("mcs_included")
    n_included = blob.get("mcs_n_included")
    if (
        isinstance(included, list)
        and isinstance(n_included, (int, float))
        and not isinstance(n_included, bool)
        and int(n_included) != len(included)
    ):
        errs.append("data_snooping_mcs_n_included_mismatch")
    for key in ("stepm_adjusted_p", "mcs_p_values"):
        table = blob.get(key)
        if not isinstance(table, dict):
            continue
        for name, val in table.items():
            if val is None:
                continue
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                errs.append(f"data_snooping_{key}_{name}_non_numeric")
                continue
            x = float(val)
            if x != x:
                continue
            if not 0.0 <= x <= 1.0:
                errs.append(f"data_snooping_{key}_{name}_out_of_unit_interval")
    return errs


# Soft H7 ACI coverage H-table consistency (Day Wave 32 research diagnostic).
# When ``families["conformal"]["aci"]`` exposes a *finite* ``kupiec_p``, the
# notebook must mint ``H7_aci_coverage`` (calibration) — same gate as agent
# ``_finite_number`` mint on ACI miss-Kupiec. Non-finite / missing / no aci → skip.
# Research-receipt fail-closed; not a live capital / promotion gate.
H7_HYPOTHESIS_ID = "H7_aci_coverage"
H7_EXPECTED_FAMILY = "calibration"


def conformal_aci_blob(conformal: object) -> dict | None:
    """Return ``families['conformal']['aci']`` dict, or None.

    *conformal* is ``families.get("conformal")`` (the conformal family blob),
    not the full families dict.
    """
    if not isinstance(conformal, dict):
        return None
    aci = conformal.get("aci")
    return aci if isinstance(aci, dict) else None


# Back-compat alias (Day Wave 32 rename payload→blob).
conformal_aci_payload = conformal_aci_blob


def aci_has_finite_kupiec_p(payload: object) -> bool:
    """True iff *payload* is a dict with finite ``kupiec_p`` (ACI row)."""
    if not isinstance(payload, dict):
        return False
    return _finite_scalar(payload.get("kupiec_p"))


def hypotheses_include_h7(hypotheses: object, *, require_calibration: bool = True) -> bool:
    """True iff *hypotheses* contains ``H7_aci_coverage``.

    When *require_calibration* is True (default), the matching row must also
    have ``family == "calibration"`` (agent mint contract).
    """
    if not isinstance(hypotheses, list):
        return False
    for hyp in hypotheses:
        if not isinstance(hyp, dict):
            continue
        if hyp.get("id") != H7_HYPOTHESIS_ID:
            continue
        return not (require_calibration and hyp.get("family") != H7_EXPECTED_FAMILY)
    return False


def h7_hypothesis_consistency_errors(conformal: object, hypotheses: object) -> list[str]:
    """Return soft-verify errors for H7 notebook consistency (Day Wave 32).

    Contract (research diagnostic only):
    - No ``aci`` / non-dict / missing / non-finite ``kupiec_p`` → no errors (skip).
    - Finite ACI ``kupiec_p`` without ``H7_aci_coverage``
      (calibration) → ``hypothesis_h7_missing_despite_finite_aci_kupiec_p``.
    """
    aci = conformal_aci_blob(conformal)
    if aci is None or not aci_has_finite_kupiec_p(aci):
        return []
    if hypotheses_include_h7(hypotheses, require_calibration=True):
        return []
    return ["hypothesis_h7_missing_despite_finite_aci_kupiec_p"]


# Soft H8 Mondrian high-vol H-table consistency (Day Wave 33 research diagnostic).
# When ``families["conformal"]["mondrian_aci"]`` exposes a *finite* ``high_x_kupiec_p``,
# the notebook must mint ``H8_mondrian_high_vol`` (calibration) — same gate as agent
# ``_finite_number`` mint on Mondrian high-X miss-Kupiec. Non-finite / missing /
# no mondrian_aci → skip. Research-receipt fail-closed; not a live capital / promotion gate.
H8_HYPOTHESIS_ID = "H8_mondrian_high_vol"
H8_EXPECTED_FAMILY = "calibration"


def conformal_mondrian_aci_blob(conformal: object) -> dict | None:
    """Return ``families['conformal']['mondrian_aci']`` dict, or None.

    *conformal* is ``families.get("conformal")`` (the conformal family blob),
    not the full families dict.
    """
    if not isinstance(conformal, dict):
        return None
    mond = conformal.get("mondrian_aci")
    return mond if isinstance(mond, dict) else None


def mondrian_aci_has_finite_high_x_kupiec_p(payload: object) -> bool:
    """True iff *payload* is a dict with finite ``high_x_kupiec_p`` (Mondrian row)."""
    if not isinstance(payload, dict):
        return False
    return _finite_scalar(payload.get("high_x_kupiec_p"))


# Brief / back-compat aliases (Day Wave 33; mirror H7 payload alias).
conformal_mondrian_aci_payload = conformal_mondrian_aci_blob
mondrian_has_finite_high_x_kupiec_p = mondrian_aci_has_finite_high_x_kupiec_p


def hypotheses_include_h8(hypotheses: object, *, require_calibration: bool = True) -> bool:
    """True iff *hypotheses* contains ``H8_mondrian_high_vol``.

    When *require_calibration* is True (default), the matching row must also
    have ``family == "calibration"`` (agent mint contract).
    """
    if not isinstance(hypotheses, list):
        return False
    for hyp in hypotheses:
        if not isinstance(hyp, dict):
            continue
        if hyp.get("id") != H8_HYPOTHESIS_ID:
            continue
        return not (require_calibration and hyp.get("family") != H8_EXPECTED_FAMILY)
    return False


def h8_hypothesis_consistency_errors(conformal: object, hypotheses: object) -> list[str]:
    """Return soft-verify errors for H8 notebook consistency (Day Wave 33).

    Contract (research diagnostic only):
    - No ``mondrian_aci`` / non-dict / missing / non-finite ``high_x_kupiec_p`` → no errors (skip).
    - Finite Mondrian ``high_x_kupiec_p`` without ``H8_mondrian_high_vol``
      (calibration) → ``hypothesis_h8_missing_despite_finite_high_x_kupiec_p``.
    """
    mond = conformal_mondrian_aci_blob(conformal)
    if mond is None or not mondrian_aci_has_finite_high_x_kupiec_p(mond):
        return []
    if hypotheses_include_h8(hypotheses, require_calibration=True):
        return []
    return ["hypothesis_h8_missing_despite_finite_high_x_kupiec_p"]


# Soft H11 CRC coverage H-table consistency (Day Wave 34 research diagnostic).
# When ``families["crc"]`` exposes a *finite* ``kupiec_p``, the notebook must mint
# ``H11_crc_var`` (calibration) — same gate as agent ``_finite_number`` mint on CRC
# Kupiec. Non-finite / missing → skip. Research-receipt fail-closed; not a live
# capital / promotion gate.
H11_HYPOTHESIS_ID = "H11_crc_var"
H11_EXPECTED_FAMILY = "calibration"


def crc_has_finite_kupiec_p(crc: object) -> bool:
    """True iff the CRC family exposes a finite Kupiec p-value."""
    return isinstance(crc, dict) and _finite_scalar(crc.get("kupiec_p"))


def hypotheses_include_h11(hypotheses: object, *, require_calibration: bool = True) -> bool:
    """True iff hypotheses contain the CRC calibration hypothesis H11."""
    if not isinstance(hypotheses, list):
        return False
    for hyp in hypotheses:
        if not isinstance(hyp, dict) or hyp.get("id") != H11_HYPOTHESIS_ID:
            continue
        return not (require_calibration and hyp.get("family") != H11_EXPECTED_FAMILY)
    return False


def h11_hypothesis_consistency_errors(crc: object, hypotheses: object) -> list[str]:
    """Return soft-verify errors for H11 notebook consistency (Day Wave 34).

    Contract (research diagnostic only):
    - Non-dict / missing / non-finite ``kupiec_p`` → no errors (skip).
    - Finite CRC ``kupiec_p`` without ``H11_crc_var`` (calibration) →
      ``hypothesis_h11_missing_despite_finite_crc_kupiec_p``.
    """
    if not crc_has_finite_kupiec_p(crc):
        return []
    if hypotheses_include_h11(hypotheses, require_calibration=True):
        return []
    return ["hypothesis_h11_missing_despite_finite_crc_kupiec_p"]


# Soft H12 weighted CQR coverage H-table consistency (Day Wave 35 research diagnostic).
# When ``families["weighted_conformal"]`` exposes a *finite* ``kupiec_p``, the notebook
# must mint ``H12_weighted_cqr`` (calibration) — same gate as agent ``_finite_number``
# mint on weighted CQR Kupiec. Non-finite / missing → skip. Research-receipt fail-closed;
# not a live capital / promotion gate.
H12_HYPOTHESIS_ID = "H12_weighted_cqr"
H12_EXPECTED_FAMILY = "calibration"


def weighted_conformal_has_finite_kupiec_p(wcqr: object) -> bool:
    """True iff the weighted_conformal family exposes a finite Kupiec p-value."""
    return isinstance(wcqr, dict) and _finite_scalar(wcqr.get("kupiec_p"))


def hypotheses_include_h12(hypotheses: object, *, require_calibration: bool = True) -> bool:
    """True iff hypotheses contain the weighted CQR calibration hypothesis H12."""
    if not isinstance(hypotheses, list):
        return False
    for hyp in hypotheses:
        if not isinstance(hyp, dict) or hyp.get("id") != H12_HYPOTHESIS_ID:
            continue
        return not (require_calibration and hyp.get("family") != H12_EXPECTED_FAMILY)
    return False


def h12_hypothesis_consistency_errors(wcqr: object, hypotheses: object) -> list[str]:
    """Return soft-verify errors for H12 notebook consistency (Day Wave 35).

    Contract (research diagnostic only):
    - Non-dict / missing / non-finite ``kupiec_p`` → no errors (skip).
    - Finite weighted_conformal ``kupiec_p`` without ``H12_weighted_cqr``
      (calibration) → ``hypothesis_h12_missing_despite_finite_wcqr_kupiec_p``.
    """
    if not weighted_conformal_has_finite_kupiec_p(wcqr):
        return []
    if hypotheses_include_h12(hypotheses, require_calibration=True):
        return []
    return ["hypothesis_h12_missing_despite_finite_wcqr_kupiec_p"]


# Soft H9 e-process ACI H-table consistency (Day Wave 36 research diagnostic).
# When ``families["evalues"]`` exposes a *finite* ``e_sup``, the notebook must mint
# ``H9_eprocess_aci`` (calibration) — same gate as agent ``_finite_number`` mint on
# e-process e_sup. Non-finite / missing → skip. Research-receipt fail-closed; not a
# live capital / promotion gate.
H9_HYPOTHESIS_ID = "H9_eprocess_aci"
H9_EXPECTED_FAMILY = "calibration"


def evalues_has_finite_e_sup(evalues: object) -> bool:
    """True iff the evalues family exposes a finite e_sup."""
    return isinstance(evalues, dict) and _finite_scalar(evalues.get("e_sup"))


def hypotheses_include_h9(hypotheses: object, *, require_calibration: bool = True) -> bool:
    """True iff hypotheses contain the e-process ACI calibration hypothesis H9."""
    if not isinstance(hypotheses, list):
        return False
    for hyp in hypotheses:
        if not isinstance(hyp, dict) or hyp.get("id") != H9_HYPOTHESIS_ID:
            continue
        return not (require_calibration and hyp.get("family") != H9_EXPECTED_FAMILY)
    return False


def h9_hypothesis_consistency_errors(evalues: object, hypotheses: object) -> list[str]:
    """Return soft-verify errors for H9 notebook consistency (Day Wave 36).

    Contract (research diagnostic only):
    - Non-dict / missing / non-finite ``e_sup`` → no errors (skip).
    - Finite evalues ``e_sup`` without ``H9_eprocess_aci`` (calibration) →
      ``hypothesis_h9_missing_despite_finite_e_sup``.
    """
    if not evalues_has_finite_e_sup(evalues):
        return []
    if hypotheses_include_h9(hypotheses, require_calibration=True):
        return []
    return ["hypothesis_h9_missing_despite_finite_e_sup"]


# Soft H10 Jackknife+ coverage H-table consistency (Day Wave 37 research diagnostic).
# When ``families["jackknife_plus"]`` exposes a *finite* ``coverage``, the notebook
# must mint ``H10_jackknife_coverage`` (bound) — same gate as agent ``_finite_number``
# mint on Jackknife+ coverage. Non-finite / missing → skip. Research-receipt fail-closed;
# not a live capital / promotion gate. Note: expected family is ``bound`` (not calibration).
H10_HYPOTHESIS_ID = "H10_jackknife_coverage"
H10_EXPECTED_FAMILY = "bound"


def jackknife_plus_has_finite_coverage(jp: object) -> bool:
    """True iff the jackknife_plus family exposes a finite coverage."""
    return isinstance(jp, dict) and _finite_scalar(jp.get("coverage"))


def hypotheses_include_h10(hypotheses: object, *, require_bound: bool = True) -> bool:
    """True iff hypotheses contain the Jackknife+ bound hypothesis H10."""
    if not isinstance(hypotheses, list):
        return False
    for hyp in hypotheses:
        if not isinstance(hyp, dict) or hyp.get("id") != H10_HYPOTHESIS_ID:
            continue
        return not (require_bound and hyp.get("family") != H10_EXPECTED_FAMILY)
    return False


def h10_hypothesis_consistency_errors(jp: object, hypotheses: object) -> list[str]:
    """Return soft-verify errors for H10 notebook consistency (Day Wave 37).

    Contract (research diagnostic only):
    - Non-dict / missing / non-finite ``coverage`` → no errors (skip).
    - Finite jackknife_plus ``coverage`` without ``H10_jackknife_coverage``
      (bound) → ``hypothesis_h10_missing_despite_finite_coverage``.
    """
    if not jackknife_plus_has_finite_coverage(jp):
        return []
    if hypotheses_include_h10(hypotheses, require_bound=True):
        return []
    return ["hypothesis_h10_missing_despite_finite_coverage"]


# Soft H15 CV+ floor H-table consistency (Day Wave 38 research diagnostic).
# When ``families["cv_plus"]`` exposes *finite* ``coverage`` *and* finite
# ``coverage_floor``, the notebook must mint ``H15_cv_plus_floor`` (bound) —
# same gate as agent ``_finite_number`` mint on both fields. Non-finite /
# missing either field → skip. Research-receipt fail-closed; not a live capital
# / promotion gate. Note: expected family is ``bound`` (not calibration).
H15_HYPOTHESIS_ID = "H15_cv_plus_floor"
H15_EXPECTED_FAMILY = "bound"


def cv_plus_has_finite_coverage_and_floor(cvp: object) -> bool:
    """True iff the cv_plus family exposes finite coverage and coverage_floor."""
    return (
        isinstance(cvp, dict)
        and _finite_scalar(cvp.get("coverage"))
        and _finite_scalar(cvp.get("coverage_floor"))
    )


def hypotheses_include_h15(hypotheses: object, *, require_bound: bool = True) -> bool:
    """True iff hypotheses contain the CV+ bound hypothesis H15."""
    if not isinstance(hypotheses, list):
        return False
    for hyp in hypotheses:
        if not isinstance(hyp, dict) or hyp.get("id") != H15_HYPOTHESIS_ID:
            continue
        return not (require_bound and hyp.get("family") != H15_EXPECTED_FAMILY)
    return False


def h15_hypothesis_consistency_errors(cvp: object, hypotheses: object) -> list[str]:
    """Return soft-verify errors for H15 notebook consistency (Day Wave 38).

    Contract (research diagnostic only):
    - Non-dict / missing / non-finite ``coverage`` or ``coverage_floor`` → no errors (skip).
    - Finite cv_plus ``coverage`` + ``coverage_floor`` without ``H15_cv_plus_floor``
      (bound) → ``hypothesis_h15_missing_despite_finite_coverage_and_floor``.
    """
    if not cv_plus_has_finite_coverage_and_floor(cvp):
        return []
    if hypotheses_include_h15(hypotheses, require_bound=True):
        return []
    return ["hypothesis_h15_missing_despite_finite_coverage_and_floor"]


# Soft H16–H18 panel Kupiec H-table consistency (Day Wave 39 research diagnostic).
# When a *panel* family blob (localized_conformal / online_crc / portfolio_conformal)
# exposes a *finite* ``kupiec_p`` and ``dgp != "fixture"``, the notebook must mint the
# matching calibration hypothesis (agent mint). Fixture DGP never shares this H-table —
# skip. Non-finite / missing → skip. Research-receipt fail-closed; not a live capital /
# promotion gate. One parameterized helper keeps the three rows thin.
H16_H18_EXPECTED_FAMILY = "calibration"
H16_H18_PANEL_SPECS: tuple[tuple[str, str, str], ...] = (
    (
        "localized_conformal",
        "H16_localized_cqr",
        "hypothesis_h16_missing_despite_finite_kupiec_p",
    ),
    (
        "online_crc",
        "H17_online_crc",
        "hypothesis_h17_missing_despite_finite_kupiec_p",
    ),
    (
        "portfolio_conformal",
        "H18_portfolio_conformal",
        "hypothesis_h18_missing_despite_finite_kupiec_p",
    ),
)
H16_HYPOTHESIS_ID = "H16_localized_cqr"
H17_HYPOTHESIS_ID = "H17_online_crc"
H18_HYPOTHESIS_ID = "H18_portfolio_conformal"


def panel_family_has_finite_kupiec_p(blob: object) -> bool:
    """True iff panel family blob has a finite panel Kupiec p and is not fixture DGP.

    Panel families (H16–H18) may expose either ``kupiec_p`` or the clustered
    variant ``date_clustered_p``; either finite value satisfies the gate.
    """
    if not isinstance(blob, dict):
        return False
    if str(blob.get("dgp") or "") == "fixture":
        return False
    return _finite_scalar(blob.get("kupiec_p")) or _finite_scalar(blob.get("date_clustered_p"))


def hypotheses_include_id(
    hypotheses: object,
    hyp_id: str,
    *,
    require_family: str | None = H16_H18_EXPECTED_FAMILY,
) -> bool:
    """True iff hypotheses contain ``hyp_id`` (optionally requiring family)."""
    if not isinstance(hypotheses, list):
        return False
    for hyp in hypotheses:
        if not isinstance(hyp, dict) or hyp.get("id") != hyp_id:
            continue
        return require_family is None or hyp.get("family") == require_family
    return False


def hypotheses_include_h16(hypotheses: object, *, require_calibration: bool = True) -> bool:
    """True iff hypotheses contain panel localized CQR calibration hypothesis H16."""
    fam = H16_H18_EXPECTED_FAMILY if require_calibration else None
    return hypotheses_include_id(hypotheses, H16_HYPOTHESIS_ID, require_family=fam)


def hypotheses_include_h17(hypotheses: object, *, require_calibration: bool = True) -> bool:
    """True iff hypotheses contain panel online CRC calibration hypothesis H17."""
    fam = H16_H18_EXPECTED_FAMILY if require_calibration else None
    return hypotheses_include_id(hypotheses, H17_HYPOTHESIS_ID, require_family=fam)


def hypotheses_include_h18(hypotheses: object, *, require_calibration: bool = True) -> bool:
    """True iff hypotheses contain panel portfolio conformal calibration hypothesis H18."""
    fam = H16_H18_EXPECTED_FAMILY if require_calibration else None
    return hypotheses_include_id(hypotheses, H18_HYPOTHESIS_ID, require_family=fam)


def h16_h18_panel_kupiec_consistency_errors(families: object, hypotheses: object) -> list[str]:
    """Return soft-verify errors for H16–H18 panel Kupiec notebook consistency (Day Wave 39).

    Contract (research diagnostic only):
    - Non-dict families / missing family / fixture ``dgp`` / non-finite ``kupiec_p`` → skip.
    - Finite panel ``kupiec_p`` without matching H16/H17/H18 (calibration) → error token.
    """
    if not isinstance(families, dict):
        return []
    errors: list[str] = []
    for family_key, hyp_id, err_token in H16_H18_PANEL_SPECS:
        blob = families.get(family_key)
        if not panel_family_has_finite_kupiec_p(blob):
            continue
        if hypotheses_include_id(hypotheses, hyp_id, require_family=H16_H18_EXPECTED_FAMILY):
            continue
        errors.append(err_token)
    return errors


# Soft H19 conformal-rank FDR bound H-table consistency (Day Wave 40 research diagnostic).
# When ``families["conformal_rank"]`` exposes a *finite* ``fdr`` and ``dgp != "fixture"``,
# the notebook must mint ``H19_conformal_rank`` (bound) — same gate as agent mint
# (``_finite_number`` on fdr; fixture DGP never shares this H-table). Non-finite /
# missing / fixture → skip. Alpha defaults to 0.20 at mint time when missing; soft
# verify only gates finite-fdr + non-fixture → bound H19 presence. Research-receipt
# fail-closed; not a live capital / promotion gate. Note: expected family is ``bound``.
H19_HYPOTHESIS_ID = "H19_conformal_rank"
H19_EXPECTED_FAMILY = "bound"


def conformal_rank_has_finite_fdr(topk: object) -> bool:
    """True iff conformal_rank family has finite fdr and is not fixture DGP."""
    if not isinstance(topk, dict):
        return False
    if str(topk.get("dgp") or "") == "fixture":
        return False
    return _finite_scalar(topk.get("fdr"))


def hypotheses_include_h19(hypotheses: object, *, require_bound: bool = True) -> bool:
    """True iff hypotheses contain the conformal-rank bound hypothesis H19."""
    if not isinstance(hypotheses, list):
        return False
    for hyp in hypotheses:
        if not isinstance(hyp, dict) or hyp.get("id") != H19_HYPOTHESIS_ID:
            continue
        return not (require_bound and hyp.get("family") != H19_EXPECTED_FAMILY)
    return False


def h19_hypothesis_consistency_errors(topk: object, hypotheses: object) -> list[str]:
    """Return soft-verify errors for H19 notebook consistency (Day Wave 40).

    Contract (research diagnostic only):
    - Non-dict / missing / fixture ``dgp`` / non-finite ``fdr`` → no errors (skip).
    - Finite conformal_rank ``fdr`` (non-fixture) without ``H19_conformal_rank``
      (bound) → ``hypothesis_h19_missing_despite_finite_fdr``.
    """
    if not conformal_rank_has_finite_fdr(topk):
        return []
    if hypotheses_include_h19(hypotheses, require_bound=True):
        return []
    return ["hypothesis_h19_missing_despite_finite_fdr"]


# Soft H20–H22 Northset notebook consistency (catalog v2 research diagnostic).
# Finite ``ohlc_identity_rate`` → bound H20; finite ``book_uncrossed_rate`` → bound
# H21; finite ``imbalance_top_p_ic`` → discovery H22. Non-finite / missing → skip.
# Research-receipt fail-closed; not a live capital / promotion gate.
H20_HYPOTHESIS_ID = "H20_northset_ohlc"
H20_EXPECTED_FAMILY = "bound"
H21_HYPOTHESIS_ID = "H21_northset_book"
H21_EXPECTED_FAMILY = "bound"
H22_HYPOTHESIS_ID = "H22_northset_imbalance"
H22_EXPECTED_FAMILY = "discovery"


def northset_has_finite_ohlc_identity_rate(blob: object) -> bool:
    """True iff the northset family exposes a finite ohlc_identity_rate."""
    return isinstance(blob, dict) and _finite_scalar(blob.get("ohlc_identity_rate"))


def northset_has_finite_book_uncrossed_rate(blob: object) -> bool:
    """True iff the northset family exposes a finite book_uncrossed_rate."""
    return (
        isinstance(blob, dict)
        and blob.get("book_hypothesis_eligible", True) is not False
        and _finite_scalar(blob.get("book_uncrossed_rate"))
    )


def northset_has_finite_imbalance_p_ic(blob: object) -> bool:
    """True iff the northset family exposes a finite imbalance_top_p_ic."""
    return (
        isinstance(blob, dict)
        and blob.get("book_hypothesis_eligible", True) is not False
        and _finite_scalar(blob.get("imbalance_top_p_ic"))
    )


def hypotheses_include_h20(hypotheses: object, *, require_bound: bool = True) -> bool:
    """True iff hypotheses contain the Northset OHLC bound hypothesis H20."""
    fam = H20_EXPECTED_FAMILY if require_bound else None
    return hypotheses_include_id(hypotheses, H20_HYPOTHESIS_ID, require_family=fam)


def hypotheses_include_h21(hypotheses: object, *, require_bound: bool = True) -> bool:
    """True iff hypotheses contain the Northset book bound hypothesis H21."""
    fam = H21_EXPECTED_FAMILY if require_bound else None
    return hypotheses_include_id(hypotheses, H21_HYPOTHESIS_ID, require_family=fam)


def hypotheses_include_h22(hypotheses: object, *, require_discovery: bool = True) -> bool:
    """True iff hypotheses contain the Northset imbalance discovery hypothesis H22."""
    fam = H22_EXPECTED_FAMILY if require_discovery else None
    return hypotheses_include_id(hypotheses, H22_HYPOTHESIS_ID, require_family=fam)


def h20_hypothesis_consistency_errors(blob: object, hypotheses: object) -> list[str]:
    """Return soft-verify errors for H20 Northset OHLC identity consistency."""
    if not northset_has_finite_ohlc_identity_rate(blob):
        return []
    if hypotheses_include_h20(hypotheses, require_bound=True):
        return []
    return ["hypothesis_h20_missing_despite_finite_ohlc_identity_rate"]


def h21_hypothesis_consistency_errors(blob: object, hypotheses: object) -> list[str]:
    """Return soft-verify errors for H21 Northset uncrossed-book consistency."""
    if not northset_has_finite_book_uncrossed_rate(blob):
        return []
    if hypotheses_include_h21(hypotheses, require_bound=True):
        return []
    return ["hypothesis_h21_missing_despite_finite_book_uncrossed_rate"]


def h22_hypothesis_consistency_errors(blob: object, hypotheses: object) -> list[str]:
    """Return soft-verify errors for H22 Northset imbalance IC consistency."""
    if not northset_has_finite_imbalance_p_ic(blob):
        return []
    if hypotheses_include_h22(hypotheses, require_discovery=True):
        return []
    return ["hypothesis_h22_missing_despite_finite_imbalance_p_ic"]


# Soft H23–H51 Northset expansion (catalog v2). Finite metric → matching H-row.
# Bound: session reconstruct / volume conservation / session chain / OOT sign.
# Discovery: microprice IC, wick-skew IC, OFI IC, GK-vs-Parkinson DM,
# CLV IC, overnight-split vs Parkinson DM, VPIN IC, sweep-reject IC,
# sweep-follow IC, event studies, placebos, matched/liq controls, name cluster,
# two-way cluster, untradeable overnight gap.
# Non-finite / missing → skip.
H23_HYPOTHESIS_ID = "H23_northset_session"
H24_HYPOTHESIS_ID = "H24_northset_volume"
H25_HYPOTHESIS_ID = "H25_northset_microprice"
H26_HYPOTHESIS_ID = "H26_northset_wick"
H27_HYPOTHESIS_ID = "H27_northset_ofi"
H28_HYPOTHESIS_ID = "H28_northset_gk"
H29_HYPOTHESIS_ID = "H29_northset_chain"
H30_HYPOTHESIS_ID = "H30_northset_clv"
H31_HYPOTHESIS_ID = "H31_northset_overnight"
H32_HYPOTHESIS_ID = "H32_northset_vpin"
H33_HYPOTHESIS_ID = "H33_northset_sweep_reject"
H34_HYPOTHESIS_ID = "H34_northset_sweep_follow"
H35_HYPOTHESIS_ID = "H35_northset_reject_event"
H36_HYPOTHESIS_ID = "H36_northset_follow_event"
H37_HYPOTHESIS_ID = "H37_northset_reject_placebo"
H38_HYPOTHESIS_ID = "H38_northset_follow_placebo"
H39_HYPOTHESIS_ID = "H39_northset_reject_cost"
H40_HYPOTHESIS_ID = "H40_northset_follow_cost"
H41_HYPOTHESIS_ID = "H41_northset_reject_stability"
H42_HYPOTHESIS_ID = "H42_northset_follow_stability"
H44_HYPOTHESIS_ID = "H44_northset_reject_control"
H45_HYPOTHESIS_ID = "H45_northset_follow_control"
H46_HYPOTHESIS_ID = "H46_northset_reject_liq_control"
H47_HYPOTHESIS_ID = "H47_northset_follow_liq_control"
H48_HYPOTHESIS_ID = "H48_northset_follow_oot"
H49_HYPOTHESIS_ID = "H49_northset_follow_name_cluster"
H50_HYPOTHESIS_ID = "H50_northset_follow_two_way_cluster"
H51_HYPOTHESIS_ID = "H51_northset_follow_overnight_gap"
NORTHSET_H23_H28_SPECS: tuple[tuple[str, str, str, str], ...] = (
    (
        "session_reconstructs_daily_rate",
        H23_HYPOTHESIS_ID,
        "bound",
        "hypothesis_h23_missing_despite_finite_session_reconstructs_daily_rate",
    ),
    (
        "session_volume_conservation_rate",
        H24_HYPOTHESIS_ID,
        "bound",
        "hypothesis_h24_missing_despite_finite_session_volume_conservation_rate",
    ),
    (
        "microprice_p_ic",
        H25_HYPOTHESIS_ID,
        "discovery",
        "hypothesis_h25_missing_despite_finite_microprice_p_ic",
    ),
    (
        "wick_skew_p_ic",
        H26_HYPOTHESIS_ID,
        "discovery",
        "hypothesis_h26_missing_despite_finite_wick_skew_p_ic",
    ),
    (
        "ofi_p_ic",
        H27_HYPOTHESIS_ID,
        "discovery",
        "hypothesis_h27_missing_despite_finite_ofi_p_ic",
    ),
    (
        "dm_gk_vs_park_p",
        H28_HYPOTHESIS_ID,
        "discovery",
        "hypothesis_h28_missing_despite_finite_dm_gk_vs_park_p",
    ),
    (
        "session_chain_rate",
        H29_HYPOTHESIS_ID,
        "bound",
        "hypothesis_h29_missing_despite_finite_session_chain_rate",
    ),
    (
        "clv_p_ic",
        H30_HYPOTHESIS_ID,
        "discovery",
        "hypothesis_h30_missing_despite_finite_clv_p_ic",
    ),
    (
        "dm_split_vs_park_p",
        H31_HYPOTHESIS_ID,
        "discovery",
        "hypothesis_h31_missing_despite_finite_dm_split_vs_park_p",
    ),
    (
        "vpin_p_ic",
        H32_HYPOTHESIS_ID,
        "discovery",
        "hypothesis_h32_missing_despite_finite_vpin_p_ic",
    ),
    (
        "sweep_reject_signed_p_ic",
        H33_HYPOTHESIS_ID,
        "discovery",
        "hypothesis_h33_missing_despite_finite_sweep_reject_signed_p_ic",
    ),
    (
        "sweep_follow_signed_p_ic",
        H34_HYPOTHESIS_ID,
        "discovery",
        "hypothesis_h34_missing_despite_finite_sweep_follow_signed_p_ic",
    ),
    (
        "sweep_reject_event_p",
        H35_HYPOTHESIS_ID,
        "discovery",
        "hypothesis_h35_missing_despite_finite_sweep_reject_event_p",
    ),
    (
        "sweep_follow_event_p",
        H36_HYPOTHESIS_ID,
        "discovery",
        "hypothesis_h36_missing_despite_finite_sweep_follow_event_p",
    ),
    (
        "sweep_reject_placebo_p",
        H37_HYPOTHESIS_ID,
        "discovery",
        "hypothesis_h37_missing_despite_finite_sweep_reject_placebo_p",
    ),
    (
        "sweep_follow_placebo_p",
        H38_HYPOTHESIS_ID,
        "discovery",
        "hypothesis_h38_missing_despite_finite_sweep_follow_placebo_p",
    ),
    (
        "sweep_reject_cost_adjusted_mean_bps",
        H39_HYPOTHESIS_ID,
        "bound",
        "hypothesis_h39_missing_despite_finite_sweep_reject_cost_adjusted_mean_bps",
    ),
    (
        "sweep_follow_cost_adjusted_mean_bps",
        H40_HYPOTHESIS_ID,
        "bound",
        "hypothesis_h40_missing_despite_finite_sweep_follow_cost_adjusted_mean_bps",
    ),
    (
        "sweep_reject_fold_positive_fraction",
        H41_HYPOTHESIS_ID,
        "bound",
        "hypothesis_h41_missing_despite_finite_sweep_reject_fold_positive_fraction",
    ),
    (
        "sweep_follow_fold_positive_fraction",
        H42_HYPOTHESIS_ID,
        "bound",
        "hypothesis_h42_missing_despite_finite_sweep_follow_fold_positive_fraction",
    ),
    (
        "sweep_reject_control_diff_p",
        H44_HYPOTHESIS_ID,
        "discovery",
        "hypothesis_h44_missing_despite_finite_sweep_reject_control_diff_p",
    ),
    (
        "sweep_follow_control_diff_p",
        H45_HYPOTHESIS_ID,
        "discovery",
        "hypothesis_h45_missing_despite_finite_sweep_follow_control_diff_p",
    ),
    (
        "sweep_reject_liq_control_diff_p",
        H46_HYPOTHESIS_ID,
        "discovery",
        "hypothesis_h46_missing_despite_finite_sweep_reject_liq_control_diff_p",
    ),
    (
        "sweep_follow_liq_control_diff_p",
        H47_HYPOTHESIS_ID,
        "discovery",
        "hypothesis_h47_missing_despite_finite_sweep_follow_liq_control_diff_p",
    ),
    (
        "sweep_follow_oot_holdout_mean_bps",
        H48_HYPOTHESIS_ID,
        "bound",
        "hypothesis_h48_missing_despite_finite_sweep_follow_oot_holdout_mean_bps",
    ),
    (
        "sweep_follow_name_cluster_p",
        H49_HYPOTHESIS_ID,
        "discovery",
        "hypothesis_h49_missing_despite_finite_sweep_follow_name_cluster_p",
    ),
    (
        "sweep_follow_two_way_cluster_p",
        H50_HYPOTHESIS_ID,
        "discovery",
        "hypothesis_h50_missing_despite_finite_sweep_follow_two_way_cluster_p",
    ),
    (
        "sweep_follow_overnight_gap_p",
        H51_HYPOTHESIS_ID,
        "discovery",
        "hypothesis_h51_missing_despite_finite_sweep_follow_overnight_gap_p",
    ),
)


H43_HYPOTHESIS_ID = "H43_northset_session_book_vpin"
H43_EXPECTED_FAMILY = "discovery"


def northset_has_finite_session_book_vpin_p_ic(blob: object) -> bool:
    """True iff northset exposes finite session_book_vpin_p_ic (multi-snap path)."""
    return (
        isinstance(blob, dict)
        and blob.get("session_book_hypothesis_eligible", True) is not False
        and _finite_scalar(blob.get("session_book_vpin_p_ic"))
    )


def hypotheses_include_h43(hypotheses: object, *, require_discovery: bool = True) -> bool:
    """True iff hypotheses contain H43 session-book VPIN discovery row."""
    fam = H43_EXPECTED_FAMILY if require_discovery else None
    return hypotheses_include_id(hypotheses, H43_HYPOTHESIS_ID, require_family=fam)


def h43_hypothesis_consistency_errors(blob: object, hypotheses: object) -> list[str]:
    """Soft-verify: finite session_book_vpin_p_ic → H43 discovery row present."""
    if not northset_has_finite_session_book_vpin_p_ic(blob):
        return []
    if hypotheses_include_h43(hypotheses, require_discovery=True):
        return []
    return ["hypothesis_h43_missing_despite_finite_session_book_vpin_p_ic"]


def northset_h23_h28_consistency_errors(blob: object, hypotheses: object) -> list[str]:
    """Return soft-verify errors for Northset H23–H51 notebook consistency."""
    if not isinstance(blob, dict):
        return []
    errors: list[str] = []
    book_metrics = {"microprice_p_ic", "ofi_p_ic", "vpin_p_ic"}
    for key, hyp_id, family, token in NORTHSET_H23_H28_SPECS:
        if key in book_metrics and blob.get("book_hypothesis_eligible", True) is False:
            continue
        if not _finite_scalar(blob.get(key)):
            continue
        if hypotheses_include_id(hypotheses, hyp_id, require_family=family):
            continue
        errors.append(token)
    return errors


# Soft Wave41 honesty-key receipt verify (Day Wave 42 research diagnostic).
# When nonempty ``jackknife_plus`` / ``cv_plus`` exposes ``coverage`` and/or
# ``coverage_floor`` (values may be NaN — key presence only), the blob must
# also expose ``coverage_guarantee_scope == "marginal_exchangeable"`` (Wave41
# honesty key). Empty ``{}`` / missing coverage keys → skip. Missing key →
# ``coverage_guarantee_scope_missing:<fam>``; wrong value →
# ``coverage_guarantee_scope_invalid:<fam>``. Research-receipt fail-closed;
# not a live capital / promotion gate; does **not** expand soft-verify H-table
# sprawl (H5/H6/H13/H14 remain paused). No forged soft scorecard flag.
COVERAGE_GUARANTEE_SCOPE_MARGINAL = "marginal_exchangeable"
_JP_CV_SCOPE_FAMILIES = ("jackknife_plus", "cv_plus")


def jp_cv_blob_requires_marginal_coverage_scope(blob: object) -> bool:
    """True iff nonempty family blob exposes coverage and/or coverage_floor keys."""
    if not isinstance(blob, dict) or not blob:
        return False
    return "coverage" in blob or "coverage_floor" in blob


def coverage_guarantee_scope_is_marginal(blob: object) -> bool:
    """True iff blob declares coverage_guarantee_scope=marginal_exchangeable."""
    return (
        isinstance(blob, dict)
        and blob.get("coverage_guarantee_scope") == COVERAGE_GUARANTEE_SCOPE_MARGINAL
    )


def coverage_guarantee_scope_consistency_errors(families: object) -> list[str]:
    """Return soft-verify errors for Jackknife+/CV+ marginal scope honesty (Day Wave 42).

    Contract (research diagnostic only):
    - Non-dict families / empty blob / neither ``coverage`` nor ``coverage_floor`` → skip.
    - Nonempty jp/cv blob with coverage keys but missing ``coverage_guarantee_scope``
      → ``coverage_guarantee_scope_missing:<fam>``.
    - Key present but value != ``marginal_exchangeable``
      → ``coverage_guarantee_scope_invalid:<fam>``.
    - NaN coverage/floor still requires scope (key presence only).
    """
    if not isinstance(families, dict):
        return []
    errors: list[str] = []
    for fam in _JP_CV_SCOPE_FAMILIES:
        blob = families.get(fam)
        if not jp_cv_blob_requires_marginal_coverage_scope(blob):
            continue
        assert isinstance(blob, dict)
        if "coverage_guarantee_scope" not in blob:
            errors.append(f"coverage_guarantee_scope_missing:{fam}")
        elif blob.get("coverage_guarantee_scope") != COVERAGE_GUARANTEE_SCOPE_MARGINAL:
            errors.append(f"coverage_guarantee_scope_invalid:{fam}")
    return errors


def mean_microprice_weight_balance_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: mean_microprice_weight_balance ∈ [0, 1] when finite.

    Applies to northset and candle_order_book family receipts. NaN / absent → skip.
    Research diagnostic only; never live Sharpe / promotion.
    """
    if not isinstance(blob, dict):
        return []
    val = blob.get("mean_microprice_weight_balance")
    if val is None:
        return []
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        # Bools/strings are not probabilities; never coerce them into range.
        return ["mean_microprice_weight_balance_non_numeric"]
    x = float(val)
    if x != x:  # NaN
        return []
    if not (0.0 <= x <= 1.0):
        return ["mean_microprice_weight_balance_out_of_unit_interval"]
    return []


def join_coverage_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: join_coverage ∈ (0, 1] when finite, or [min_join, 1] when floor present.

    Applies to northset and candle_order_book fuse receipts. Uses
    ``book_join_coverage_floor`` when finite (northset fail-closed companion);
    otherwise requires open-unit interval (0, 1]. NaN / absent → skip.
    Research diagnostic only; never live Sharpe / promotion.
    """
    if not isinstance(blob, dict):
        return []
    val = blob.get("join_coverage")
    if val is None:
        return []
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        # Bools/strings must not coerce into a passing coverage.
        return ["join_coverage_non_numeric"]
    x = float(val)
    if x != x:  # NaN
        return []
    if abs(x) == float("inf"):
        return ["join_coverage_non_finite_fail_closed"]

    floor: float | None = None
    raw_floor = blob.get("book_join_coverage_floor")
    if raw_floor is None:
        raw_floor = blob.get("min_join_coverage")
    try:
        f = float(raw_floor)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        f = float("nan")
    if f == f and abs(f) != float("inf") and 0.0 <= f <= 1.0:
        floor = f

    if floor is not None:
        if not (floor <= x <= 1.0):
            return ["join_coverage_outside_floor_to_one_fail_closed"]
        return []
    if not (0.0 < x <= 1.0):
        return ["join_coverage_outside_open_unit_interval_fail_closed"]
    return []


def mean_book_age_seconds_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: mean_book_age_seconds ≥ 0 when finite.

    Fuse/join age diagnostic on northset + candle_order_book receipts. NaN
    (no book ages) skipped. Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    val = blob.get("mean_book_age_seconds")
    try:
        x = float(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return []
    if x != x:  # NaN
        return []
    if x < 0.0 or abs(x) == float("inf"):
        return ["mean_book_age_seconds_negative_or_non_finite"]
    return []


def book_age_seconds_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: mean/max book_age_seconds ≥ 0; max ≥ mean when both finite.

    Applies to northset and candle_order_book fuse receipts that stamp
    ``mean_book_age_seconds`` / ``max_book_age_seconds``. NaN / absent → skip.
    ±inf → fail-closed. Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    errs: list[str] = []

    def _finite(key: str) -> float | None:
        try:
            x = float(blob.get(key))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
        if x != x:
            return None
        if abs(x) == float("inf"):
            errs.append(f"{key}_non_finite_fail_closed")
            return None
        return x

    mean = _finite("mean_book_age_seconds")
    mx = _finite("max_book_age_seconds")
    if mean is not None and mean < 0.0:
        errs.append("mean_book_age_seconds_negative")
    if mx is not None and mx < 0.0:
        errs.append("max_book_age_seconds_negative")
    if mean is not None and mx is not None and mx + 1e-12 < mean:
        errs.append("max_book_age_seconds_lt_mean")
    return errs


def structure_finite_rate_honesty_errors(blob: object) -> list[str]:
    """Soft-verify structure finite-rate keys ∈ [0, 1] when finite.

    Checks:
    - aggregate ``structure_finite_rate`` (northset + candle receipts)
    - candle ``finite_rate_*`` companions (microprice_minus_mid / size concentration)

    Never equate northset aggregate with candle companions. NaN/absent skip;
    ±inf fail-closed. Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    keys = (
        "structure_finite_rate",
        "finite_rate_microprice_minus_mid",
        "finite_rate_bid_size_concentration_top",
        "finite_rate_ask_size_concentration_top",
    )
    errs: list[str] = []
    for key in keys:
        if key not in blob:
            continue
        try:
            x = float(blob.get(key))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append(f"{key}_non_numeric")
            continue
        if x != x:
            continue
        if abs(x) == float("inf") or not (0.0 <= x <= 1.0):
            errs.append(f"{key}_out_of_unit_interval")
    return errs


def northset_top_level_claim_honesty_errors(blob: object) -> list[str]:
    """Soft-verify northset top-level research_only + claim markers.

    When either key is present: research_only must be True and claim must equal
    ``research_diagnostic_only``. Coupling: ``research_only is True`` ⇒ claim
    key present and correct (fail-closed if claim missing). Absent both → skip.
    Research diagnostic only; never live Sharpe. Off kyle nest.
    """
    if not isinstance(blob, dict):
        return []
    if "research_only" not in blob and "claim" not in blob:
        return []
    errs: list[str] = []
    if "research_only" in blob and blob.get("research_only") is not True:
        errs.append("northset_research_only_missing_or_false")
    if blob.get("research_only") is True:
        if "claim" not in blob:
            errs.append("northset_claim_missing_while_research_only_true")
        elif blob.get("claim") != "research_diagnostic_only":
            errs.append("northset_claim_not_research_diagnostic_only")
    elif "claim" in blob:
        # claim ⇒ research_only: a present claim without the honesty flag is not
        # acceptable evidence (the coupling must hold in both directions).
        if "research_only" not in blob:
            errs.append("northset_research_only_missing_or_false")
        if blob.get("claim") != "research_diagnostic_only":
            errs.append("northset_claim_not_research_diagnostic_only")
    return errs


def northset_shape_and_session_l2_floors_honesty_errors(blob: object) -> list[str]:
    """Soft-verify optional shape floors + session_l2_identity_floor ∈ [0, 1].

    When a floor key is present and not None/NaN, it must be finite and in [0, 1].
    Keys: depth_shape / concentration_top / queue_priority / side_notional /
    tob_size_share *_finite_floor, session_l2_identity_floor, and
    ``book_join_coverage_floor`` (stamped join companion).
    session_l2_identity_gate when present must be "enforced" or "skipped".
    Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    errs: list[str] = []
    floor_keys = (
        "depth_shape_finite_floor",
        "concentration_top_finite_floor",
        "queue_priority_finite_floor",
        "side_notional_finite_floor",
        "tob_size_share_finite_floor",
        "session_l2_identity_floor",
        "book_join_coverage_floor",
    )
    for key in floor_keys:
        if key not in blob:
            continue
        val = blob.get(key)
        if val is None:
            continue
        try:
            x = float(val)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append(f"{key}_non_numeric")
            continue
        if x != x:
            continue
        if abs(x) == float("inf") or not (0.0 <= x <= 1.0):
            errs.append(f"{key}_out_of_unit_interval")
    if "session_l2_identity_gate" in blob:
        gate = blob.get("session_l2_identity_gate")
        if gate not in ("enforced", "skipped"):
            errs.append("session_l2_identity_gate_invalid")
    return errs


def size_concentration_top_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: mean_bid/ask_size_concentration_top ∈ (0, 1] when finite.

    Candle_order_book structure receipts. NaN/absent skip; ±inf fail-closed.
    Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    errs: list[str] = []
    for key in ("mean_bid_size_concentration_top", "mean_ask_size_concentration_top"):
        if key not in blob:
            continue
        try:
            x = float(blob.get(key))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append(f"{key}_non_numeric")
            continue
        if x != x:
            continue
        if abs(x) == float("inf"):
            errs.append(f"{key}_non_finite_fail_closed")
            continue
        if not (0.0 < x <= 1.0):
            errs.append(f"{key}_out_of_open_unit_interval")
    return errs


def mean_microprice_minus_mid_honesty_errors(blob: object) -> list[str]:
    """Soft-verify microprice−mid means finite when present (signed OK).

    Keys: mean_microprice_minus_mid (price units) and mean_microprice_minus_mid_bps.
    Never equate the two. NaN/absent skip; ±inf fail-closed.
    Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    errs: list[str] = []
    for key in ("mean_microprice_minus_mid", "mean_microprice_minus_mid_bps"):
        if key not in blob:
            continue
        try:
            x = float(blob.get(key))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append(f"{key}_non_numeric")
            continue
        if x != x:
            continue
        if abs(x) == float("inf"):
            errs.append(f"{key}_non_finite_fail_closed")
    return errs


def candle_microprice_minus_mid_finite_pack_honesty_errors(blob: object) -> list[str]:
    """Pack soft-verify: candle/northset microprice−mid means finite when stamped.

    Delegates to :func:`mean_microprice_minus_mid_honesty_errors` (price + bps).
    Research diagnostic only; never live Sharpe. Off kyle invent.
    """
    return mean_microprice_minus_mid_honesty_errors(blob)


def northset_log_size_slope_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: mean_bid/ask_log_size_slope finite when present (not ±inf).

    Slopes may be negative (deeper levels thinner). NaN skipped.
    Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    errs: list[str] = []
    for key in ("mean_bid_log_size_slope", "mean_ask_log_size_slope"):
        if key not in blob or blob.get(key) is None:
            continue
        val = blob.get(key)
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            errs.append(f"{key}_non_numeric")
            continue
        x = float(val)
        if x != x:
            continue
        if abs(x) == float("inf"):
            errs.append(f"{key}_non_finite")
    return errs


def northset_qlike_means_honesty_errors(blob: object) -> list[str]:
    """Soft-verify northset QLIKE keys ≥ 0 when finite.

    Parkinson / Garman-Klass / Rogers-Satchell / Yang-Zhang vs close-to-close
    QLIKE are losses — negative is dishonest. NaN skipped.
    Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    errs: list[str] = []
    for key in (
        "parkinson_qlike_vs_cc",
        "garman_klass_qlike_vs_cc",
        "rogers_satchell_qlike_vs_cc",
        "yang_zhang_qlike_vs_cc",
        "overnight_plus_oc_qlike_vs_cc",
        "session_rv_qlike_vs_cc",
    ):
        if key not in blob or blob.get(key) is None:
            continue
        val = blob.get(key)
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            errs.append(f"{key}_non_numeric")
            continue
        x = float(val)
        if x != x:
            continue
        if abs(x) == float("inf"):
            errs.append(f"{key}_non_finite")
            continue
        if x < 0.0:
            errs.append(f"{key}_negative")
    return errs


def northset_range_spread_honesty_errors(blob: object) -> list[str]:
    """Soft-verify range/spread estimator means when finite.

    - ``corwin_schultz_spread`` / ``abdi_ranaldo_spread`` ∈ [0, 1] (relative spreads)
    - ``roll_spread`` / ``mean_true_range`` / ``yang_zhang_variance`` ≥ 0
      (Roll can exceed 1 depending on scale)
    NaN skipped. Research diagnostic only; never live Sharpe. Off kyle_*.
    """
    if not isinstance(blob, dict):
        return []
    errs: list[str] = []
    unit_keys = ("corwin_schultz_spread", "abdi_ranaldo_spread")
    nonneg_keys = ("roll_spread", "mean_true_range", "yang_zhang_variance")
    for key in unit_keys + nonneg_keys:
        if key not in blob:
            continue
        raw = blob.get(key)
        if raw is None:
            continue  # JSON null = unavailable (NaN), never non-numeric
        try:
            x = float(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append(f"{key}_non_numeric")
            continue
        if x != x:
            continue
        if abs(x) == float("inf"):
            errs.append(f"{key}_non_finite")
            continue
        if key in unit_keys:
            if not (0.0 <= x <= 1.0):
                errs.append(f"{key}_out_of_unit_interval")
        elif x < 0.0:
            errs.append(f"{key}_negative")
    return errs


def amihud_mean_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: amihud_mean ≥ 0 when finite.

    Amihud illiquidity is |ret| / dollar volume — negative is dishonest.
    NaN skipped. Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    try:
        x = float(blob.get("amihud_mean"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return []
    if x != x:
        return []
    if abs(x) == float("inf"):
        return ["amihud_mean_non_finite"]
    if x < 0.0:
        return ["amihud_mean_negative"]
    return []


def northset_queue_sweep_ofi_honesty_errors(blob: object) -> list[str]:
    """Soft-verify queue_imbalance_mean, sweep_any_rate, ofi_mean_ic when present.

    - queue_imbalance_mean ∈ [-1, 1] when finite
    - sweep_any_rate ∈ [0, 1] when finite
    - ofi_mean_ic finite when present (not ±inf); NaN skipped
    Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    errs: list[str] = []

    def _finite(key: str) -> float | None:
        try:
            x = float(blob.get(key))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
        if x != x:
            return None
        if abs(x) == float("inf"):
            errs.append(f"{key}_non_finite")
            return None
        return x

    qi = _finite("queue_imbalance_mean")
    if qi is not None and not (-1.0 <= qi <= 1.0):
        errs.append("queue_imbalance_mean_out_of_unit_interval")

    sar = _finite("sweep_any_rate")
    if sar is not None and not (0.0 <= sar <= 1.0):
        errs.append("sweep_any_rate_out_of_unit_interval")

    # ofi_mean_ic: finite-when-present only (IC can be negative)
    _ = _finite("ofi_mean_ic")
    return errs


def depth_shape_finite_rate_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: depth_shape_finite_rate ∈ [0, 1] when finite.

    Candle_order_book / northset fuse rate of finite DEPTH_SHAPE fields.
    NaN (thin/missing shape cols) skipped. Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    val = blob.get("depth_shape_finite_rate")
    try:
        x = float(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return []
    if x != x:  # NaN
        return []
    if not (0.0 <= x <= 1.0):
        return ["depth_shape_finite_rate_out_of_unit_interval"]
    return []


def mean_tob_notional_share_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: mean_tob_notional_share ∈ (0, 1] when finite.

    NaN/absent skip; ±inf fail-closed. ≠ mean_tob_size_share.
    Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    if "mean_tob_notional_share" not in blob:
        return []
    try:
        x = float(blob.get("mean_tob_notional_share"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return ["mean_tob_notional_share_non_numeric"]
    if x != x:
        return []
    if abs(x) == float("inf") or not (0.0 < x <= 1.0):
        return ["mean_tob_notional_share_out_of_open_unit_interval"]
    return []


def candle_spread_bps_nonneg_honesty_errors(blob: object) -> list[str]:
    """Soft-verify ``mean_spread_bps`` ≥ 0 when finite (candle or northset stamp).

    Spread in bps cannot be negative. Complements half-spread identity helper
    (which only checks 2× half). NaN/absent skip. Research diagnostic only.
    """
    if not isinstance(blob, dict):
        return []
    if "mean_spread_bps" not in blob:
        return []
    try:
        x = float(blob.get("mean_spread_bps"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return ["mean_spread_bps_non_numeric"]
    if x != x:
        return []
    if abs(x) == float("inf") or x < 0.0:
        return ["mean_spread_bps_negative_or_non_finite"]
    return []


def candle_log_slopes_finite_honesty_errors(blob: object) -> list[str]:
    """Soft-verify candle bid/ask log size+price slopes finite when stamped.

    When scored/stamped: means must not be ±inf (NaN skip). Signed OK.
    Research diagnostic only; off kyle invent.
    """
    if not isinstance(blob, dict):
        return []
    if blob.get("family") not in (None, "candle_order_book"):
        return []
    errs: list[str] = []
    for key in (
        "mean_bid_log_size_slope",
        "mean_ask_log_size_slope",
        "mean_bid_log_price_slope",
        "mean_ask_log_price_slope",
    ):
        if key not in blob:
            continue
        try:
            x = float(blob.get(key))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append(f"{key}_non_numeric")
            continue
        if x != x:
            continue
        if abs(x) == float("inf"):
            errs.append(f"{key}_non_finite")
    return errs


def mean_depth_imbalance_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: mean_depth_imbalance ∈ [-1, 1] when finite.

    NaN/absent skip; ±inf fail-closed. ≠ imbalance_top / notional_imbalance.
    Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    if "mean_depth_imbalance" not in blob:
        return []
    try:
        x = float(blob.get("mean_depth_imbalance"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return ["mean_depth_imbalance_non_numeric"]
    if x != x:
        return []
    if abs(x) == float("inf") or not (-1.0 <= x <= 1.0):
        return ["mean_depth_imbalance_out_of_unit_interval"]
    return []


def mean_depth_imbalance_abs_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: mean_depth_imbalance_abs ∈ [0, 1] when finite.

    Absolute depth imbalance — ≠ signed mean_depth_imbalance. NaN/absent skip.
    Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    if "mean_depth_imbalance_abs" not in blob:
        return []
    try:
        x = float(blob.get("mean_depth_imbalance_abs"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return ["mean_depth_imbalance_abs_non_numeric"]
    if x != x:
        return []
    if abs(x) == float("inf") or not (0.0 <= x <= 1.0):
        return ["mean_depth_imbalance_abs_out_of_unit_interval"]
    return []


def mean_imbalance_top_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: mean_imbalance_top ∈ [-1, 1] when finite.

    Top-of-book size imbalance — ≠ mean_depth_imbalance / mean_notional_imbalance.
    touch_size_imbalance is an alias; do not require a second receipt key.
    NaN/absent skip. Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    if "mean_imbalance_top" not in blob:
        return []
    try:
        x = float(blob.get("mean_imbalance_top"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return ["mean_imbalance_top_non_numeric"]
    if x != x:
        return []
    if abs(x) == float("inf") or not (-1.0 <= x <= 1.0):
        return ["mean_imbalance_top_out_of_unit_interval"]
    return []


def northset_include_kyle_ofi_nest_presence_honesty_errors(blob: object) -> list[str]:
    """Soft-verify ``include_kyle_ofi`` ↔ nested ``kyle_ofi`` presence.

    Stamp contract (``bench_northset``):
    - ``include_kyle_ofi is True`` ⇒ ``kyle_ofi`` is a dict with ``research_only is True``
    - ``include_kyle_ofi is False`` ⇒ ``kyle_ofi`` key absent

    Does not invent nest internals / does not overwrite kyle_ofi.py — presence only.
    Skip when flag absent. Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    if blob.get("family") not in (None, "northset"):
        return []
    if "include_kyle_ofi" not in blob:
        return []
    flag = blob.get("include_kyle_ofi")
    if type(flag) is not bool:
        return []  # bool-flags helper owns type

    has_nest = "kyle_ofi" in blob
    nest = blob.get("kyle_ofi") if has_nest else None

    if flag is True:
        if not isinstance(nest, dict):
            return ["kyle_ofi_nest_missing_while_include_kyle_ofi_true"]
        if nest.get("research_only") is not True:
            return ["kyle_ofi_nest_research_only_not_true_while_include_kyle_ofi_true"]
        return []
    # flag False
    if has_nest:
        return ["kyle_ofi_nest_present_while_include_kyle_ofi_false"]
    return []


def northset_sweep_control_sample_adequate_honesty_errors(blob: object) -> list[str]:
    """Soft-verify ``sweep_*_control_sample_adequate True`` ⇒ companions honest.

    For each of ``sweep_reject`` / ``sweep_follow``:
    - adequate True ⇒ ``*_control_n_dates`` ≥ 1 when present, ``*_control_diff_p``
      ∈ [0, 1] when finite, ``*_control_diff_t`` finite when present (not ±inf)
    - adequate False / absent → skip numeric companions (bool helper owns type)
    Does not invent kyle nest / METRICS_*. Research diagnostic only.
    """
    if not isinstance(blob, dict):
        return []
    errs: list[str] = []
    for prefix in ("sweep_reject", "sweep_follow"):
        flag_key = f"{prefix}_control_sample_adequate"
        if flag_key not in blob:
            continue
        if blob.get(flag_key) is not True:
            continue
        n_key = f"{prefix}_control_n_dates"
        p_key = f"{prefix}_control_diff_p"
        t_key = f"{prefix}_control_diff_t"
        if n_key in blob:
            try:
                n = float(blob.get(n_key))  # type: ignore[arg-type]
            except (TypeError, ValueError):
                errs.append(f"{n_key}_non_numeric_while_sample_adequate")
            else:
                if n != n or abs(n) == float("inf") or n < 1.0:
                    errs.append(f"{n_key}_lt_one_while_sample_adequate")
        if p_key in blob:
            try:
                p = float(blob.get(p_key))  # type: ignore[arg-type]
            except (TypeError, ValueError):
                errs.append(f"{p_key}_non_numeric_while_sample_adequate")
            else:
                if p != p or abs(p) == float("inf") or not (0.0 <= p <= 1.0):
                    errs.append(f"{p_key}_invalid_while_sample_adequate")
        if t_key in blob:
            try:
                tv = float(blob.get(t_key))  # type: ignore[arg-type]
            except (TypeError, ValueError):
                errs.append(f"{t_key}_non_numeric_while_sample_adequate")
            else:
                if tv != tv:
                    pass  # NaN skip — inadequate companions may leave NaN even if flag wrong
                elif abs(tv) == float("inf"):
                    errs.append(f"{t_key}_non_finite_while_sample_adequate")
    return errs


def northset_shape_columns_ensured_book_panel_path_honesty_errors(blob: object) -> list[str]:
    """Soft-verify ``shape_columns_ensured`` ↔ ``book_panel_path`` stamp pair.

    Contract (``bench_northset`` / DATA_CONTRACTS): synth ensure runs when
    ``book_panel_path`` is absent — ``shape_columns_ensured is True`` ⇔ path
    None/empty; ``False`` ⇔ nonempty path string.

    When only one side present → skip. Research diagnostic only; never live
    Sharpe. Off nest invent / kyle_ofi overwrite (nest has its own path helper).
    """
    if not isinstance(blob, dict):
        return []
    if blob.get("family") not in (None, "northset"):
        return []
    if "shape_columns_ensured" not in blob:
        return []
    flag = blob.get("shape_columns_ensured")
    if type(flag) is not bool:
        return []  # bool-flags helper owns type

    has_path_key = "book_panel_path" in blob
    path = blob.get("book_panel_path") if has_path_key else None
    path_nonempty = isinstance(path, str) and bool(path.strip())

    if flag is True:
        if has_path_key and path is not None and not isinstance(path, str):
            # Malformed path must not read as the benign "absent" case.
            return ["book_panel_path_non_string_while_shape_columns_ensured"]
        if has_path_key and path is not None and path_nonempty:
            return ["book_panel_path_set_while_shape_columns_ensured"]
        return []
    # flag False
    if has_path_key and not path_nonempty:
        return ["book_panel_path_missing_while_shape_columns_not_ensured"]
    if not has_path_key:
        return ["book_panel_path_missing_while_shape_columns_not_ensured"]
    return []


def northset_shape_columns_ensured_rates_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: ``shape_columns_ensured is True`` ⇒ shape rates present + finite.

    When synth ensure ran, these three rates must be stamped and finite ∈ [0, 1]:
    ``depth_shape_finite_rate``, ``concentration_top_finite_rate``,
    ``queue_priority_finite_rate``. ``False`` / absent / non-bool → skip (bool
    helper owns type). Research diagnostic only; never live Sharpe.
    Off kyle_ofi / METRICS_*.
    """
    if not isinstance(blob, dict):
        return []
    flag = blob.get("shape_columns_ensured")
    if flag is not True:
        return []
    errs: list[str] = []
    for key in (
        "depth_shape_finite_rate",
        "concentration_top_finite_rate",
        "queue_priority_finite_rate",
    ):
        if key not in blob:
            errs.append(f"{key}_missing_while_shape_columns_ensured")
            continue
        try:
            x = float(blob.get(key))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append(f"{key}_non_numeric_while_shape_columns_ensured")
            continue
        if x != x or abs(x) == float("inf"):
            errs.append(f"{key}_non_finite_while_shape_columns_ensured")
        elif not (0.0 <= x <= 1.0):
            errs.append(f"{key}_out_of_unit_interval_while_shape_columns_ensured")
    return errs


def northset_metrics_required_finite_ok_rates_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: ``metrics_required_finite_ok is True`` ⇒ companion rates ok.

    When the receipt claims REQUIRED LOB metrics were finite on a sample row,
    the stamped structure finite-rate companions must be present and finite
    ∈ [0, 1]. Does **not** invent per-key ``METRICS_REQUIRED_*`` receipt stamps
    (those stay inside book_metrics asserts). Off kyle_ofi. Research only.
    """
    if not isinstance(blob, dict):
        return []
    if blob.get("metrics_required_finite_ok") is not True:
        return []
    errs: list[str] = []
    for key in (
        "depth_shape_finite_rate",
        "concentration_top_finite_rate",
        "queue_priority_finite_rate",
        "side_notional_finite_rate",
        "tob_size_share_finite_rate",
        "structure_finite_rate",
    ):
        if key not in blob:
            errs.append(f"{key}_missing_while_metrics_required_finite_ok")
            continue
        val = blob.get(key)
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            errs.append(f"{key}_non_numeric_while_metrics_required_finite_ok")
            continue
        x = float(val)
        if x != x or abs(x) == float("inf"):
            errs.append(f"{key}_non_finite_while_metrics_required_finite_ok")
        elif not (0.0 <= x <= 1.0):
            errs.append(f"{key}_out_of_unit_interval_while_metrics_required_finite_ok")
    return errs


def northset_metrics_required_keys_finite_when_present_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: METRICS_REQUIRED keys on the receipt, if present, are finite.

    Does not invent keys. Any overlapping ``METRICS_REQUIRED_FINITE_KEYS`` stamp
    must parse finite (not NaN/±inf). Off kyle_ofi / METRICS_* invent. Research only.
    """
    if not isinstance(blob, dict):
        return []
    try:
        from quant_fund.microstructure.book_metrics import METRICS_REQUIRED_FINITE_KEYS
    except Exception:
        return []
    errs: list[str] = []
    for key in METRICS_REQUIRED_FINITE_KEYS:
        if key not in blob:
            continue
        try:
            x = float(blob.get(key))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append(f"{key}_non_numeric_metrics_required")
            continue
        if x != x or abs(x) == float("inf"):
            errs.append(f"{key}_non_finite_metrics_required")
    return errs


def northset_depth_notional_spread_over_mid_honesty_errors(blob: object) -> list[str]:
    """Soft-verify depth / side-notional / spread_over_mid receipt means.

    - mean_bid_depth / mean_ask_depth ≥ 0 when finite
    - mean_side_notional_proxy_bid / ask ≥ 0 when finite
    - mean_top_of_book_notional_proxy ≥ 0 when finite
    - mean_spread_over_mid ≥ 0 when finite (≠ mean_spread_bps scale)
    NaN/absent skip; ±inf fail. Research diagnostic only; never live Sharpe.
    Does not touch kyle_* / book_metrics METRICS_* keys.
    """
    if not isinstance(blob, dict):
        return []
    errs: list[str] = []
    for key in (
        "mean_bid_depth",
        "mean_ask_depth",
        "mean_side_notional_proxy_bid",
        "mean_side_notional_proxy_ask",
        "mean_top_of_book_notional_proxy",
        "mean_spread_over_mid",
    ):
        if key not in blob:
            continue
        try:
            x = float(blob.get(key))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append(f"{key}_non_numeric")
            continue
        if x != x:
            continue
        if abs(x) == float("inf"):
            errs.append(f"{key}_non_finite")
            continue
        if x < 0.0:
            errs.append(f"{key}_negative")
    return errs


def northset_price_slope_tick_top_levels_honesty_errors(blob: object) -> list[str]:
    """Soft-verify price-slope / tick-spacing / top-size / n_levels means.

    - mean_bid/ask_log_price_slope finite when present (signed OK; ≠ size slopes)
    - mean_bid/ask_mean_log_tick_spacing finite when present (may be negative:
      log of sub-unit tick gaps)
    - mean_top_bid/ask_size ≥ 0 when finite
    - mean_n_bid/ask_levels ≥ 0 when finite
    NaN/absent skip. Research diagnostic only; never live Sharpe.
    Does not touch kyle_* / book_metrics METRICS_* keys.
    """
    if not isinstance(blob, dict):
        return []
    errs: list[str] = []
    for key in ("mean_bid_log_price_slope", "mean_ask_log_price_slope"):
        if key not in blob:
            continue
        try:
            x = float(blob.get(key))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append(f"{key}_non_numeric")
            continue
        if x != x:
            continue
        if abs(x) == float("inf"):
            errs.append(f"{key}_non_finite")
    # Log tick spacing is finite-when-present only: gaps below 1.0 (e.g. 0.01
    # ticks) give legitimately negative logs — sign is not an honesty failure.
    for key in ("mean_bid_mean_log_tick_spacing", "mean_ask_mean_log_tick_spacing"):
        if key not in blob:
            continue
        raw = blob.get(key)
        if raw is None:
            continue
        try:
            x = float(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append(f"{key}_non_numeric")
            continue
        if x != x:
            continue
        if abs(x) == float("inf"):
            errs.append(f"{key}_non_finite")
    for key in (
        "mean_top_bid_size",
        "mean_top_ask_size",
        "mean_n_bid_levels",
        "mean_n_ask_levels",
    ):
        if key not in blob:
            continue
        raw = blob.get(key)
        if raw is None:
            continue
        try:
            x = float(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append(f"{key}_non_numeric")
            continue
        if x != x:
            continue
        if abs(x) == float("inf"):
            errs.append(f"{key}_non_finite")
            continue
        if x < 0.0:
            errs.append(f"{key}_negative")
    return errs


def northset_fwd_ret_after_sweep_honesty_errors(blob: object) -> list[str]:
    """Soft-verify mean_fwd_ret_after_* sweep companions finite when present.

    Signed conditional forward returns — may be negative; only ±inf / non-numeric
    fail. Keys: high/low reclaim and high/low follow. NaN/absent skip.
    Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    errs: list[str] = []
    for key in (
        "mean_fwd_ret_after_high_reclaim",
        "mean_fwd_ret_after_low_reclaim",
        "mean_fwd_ret_after_high_follow",
        "mean_fwd_ret_after_low_follow",
    ):
        if key not in blob:
            continue
        try:
            x = float(blob.get(key))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append(f"{key}_non_numeric")
            continue
        if x != x:
            continue
        if abs(x) == float("inf"):
            errs.append(f"{key}_non_finite")
    return errs


def northset_overnight_rv_semi_honesty_errors(blob: object) -> list[str]:
    """Soft-verify stamped overnight / bipower / semi companions.

    - overnight_share ∈ [0, 1] when finite
    - session_mean_rv / session_mean_bv ≥ 0 when finite
    - semi_up / semi_down ≥ 0 when finite
    NaN/absent skip. Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    errs: list[str] = []
    if "overnight_share" in blob:
        try:
            x = float(blob.get("overnight_share"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append("overnight_share_non_numeric")
        else:
            if x == x and abs(x) != float("inf") and not (0.0 <= x <= 1.0):
                errs.append("overnight_share_out_of_unit_interval")
            elif x == x and abs(x) == float("inf"):
                errs.append("overnight_share_non_finite")
    for key in ("session_mean_rv", "session_mean_bv", "semi_up", "semi_down"):
        if key not in blob:
            continue
        try:
            x = float(blob.get(key))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append(f"{key}_non_numeric")
            continue
        if x != x:
            continue
        if abs(x) == float("inf"):
            errs.append(f"{key}_non_finite")
            continue
        if x < 0.0:
            errs.append(f"{key}_negative")
    return errs


def mean_notional_imbalance_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: mean_notional_imbalance ∈ [-1, 1] when finite.

    NaN/absent skip; ±inf fail-closed. ≠ imbalance_top. Research diagnostic only.
    """
    if not isinstance(blob, dict):
        return []
    if "mean_notional_imbalance" not in blob:
        return []
    try:
        x = float(blob.get("mean_notional_imbalance"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return ["mean_notional_imbalance_non_numeric"]
    if x != x:
        return []
    if abs(x) == float("inf") or not (-1.0 <= x <= 1.0):
        return ["mean_notional_imbalance_out_of_unit_interval"]
    return []


def mean_queue_priority_honesty_errors(blob: object) -> list[str]:
    """Soft-verify queue priority means ∈ [0, 1] when finite.

    Keys: mean_queue_priority_proxy, mean_ask_queue_priority_proxy, and legacy
    mean_queue_priority alias. NaN/absent skip; ±inf fail-closed.
    Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    errs: list[str] = []
    keys = (
        "mean_queue_priority_proxy",
        "mean_ask_queue_priority_proxy",
        "mean_queue_priority",
    )
    for key in keys:
        if key not in blob:
            continue
        try:
            x = float(blob.get(key))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append(f"{key}_non_numeric")
            continue
        if x != x:
            continue
        if abs(x) == float("inf") or not (0.0 <= x <= 1.0):
            errs.append(f"{key}_out_of_unit_interval")
    return errs


def candle_log_tick_spacing_finite_honesty_errors(blob: object) -> list[str]:
    """Soft-verify candle ``mean_*_mean_log_tick_spacing`` finite when stamped.

    Log tick spacing may be negative (log of a small relative tick). Do **not**
    require ≥0 (northset price-slope helper's ≥0 contract is northset-scale).
    NaN skip; ±inf fail-closed. Research diagnostic only; off kyle invent.
    """
    if not isinstance(blob, dict):
        return []
    if blob.get("family") not in (None, "candle_order_book"):
        return []
    errs: list[str] = []
    for key in (
        "mean_bid_mean_log_tick_spacing",
        "mean_ask_mean_log_tick_spacing",
    ):
        if key not in blob:
            continue
        try:
            x = float(blob.get(key))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append(f"{key}_non_numeric")
            continue
        if x != x:
            continue
        if abs(x) == float("inf"):
            errs.append(f"{key}_non_finite")
    return errs


def mean_tob_size_share_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: mean_tob_size_share ∈ (0, 1] when finite.

    NaN/absent skip; ±inf fail-closed. Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    if "mean_tob_size_share" not in blob:
        return []
    val = blob.get("mean_tob_size_share")
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        # Bools/strings must not coerce into a passing share.
        return ["mean_tob_size_share_non_numeric"]
    x = float(val)
    if x != x:
        return []
    if abs(x) == float("inf") or not (0.0 < x <= 1.0):
        return ["mean_tob_size_share_out_of_open_unit_interval"]
    return []


def mean_candle_dir_x_imbalance_honesty_errors(blob: object) -> list[str]:
    """Soft-verify ``mean_candle_dir_x_imbalance`` ∈ [-1, 1] when finite.

    FEATURE_COLS interaction of ternary direction and imbalance — mean must lie
    in signed unit. NaN/absent skip; ±inf fail-closed. Research diagnostic only.
    """
    if not isinstance(blob, dict):
        return []
    if "mean_candle_dir_x_imbalance" not in blob:
        return []
    try:
        x = float(blob.get("mean_candle_dir_x_imbalance"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return ["mean_candle_dir_x_imbalance_non_numeric"]
    if x != x:
        return []
    if abs(x) == float("inf") or not (-1.0 <= x <= 1.0):
        return ["mean_candle_dir_x_imbalance_out_of_signed_unit"]
    return []


def mean_close_mid_abs_rel_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: mean_close_mid_abs_rel ≥ 0 when finite.

    Candle close−mid absolute relative diagnostic (≠ mean_effective_spread /
    quoted book spread). NaN/absent skip; ±inf fail-closed.
    Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    if "mean_close_mid_abs_rel" not in blob:
        return []
    try:
        x = float(blob.get("mean_close_mid_abs_rel"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return ["mean_close_mid_abs_rel_non_numeric"]
    if x != x:
        return []
    if abs(x) == float("inf"):
        return ["mean_close_mid_abs_rel_non_finite"]
    if x < 0.0:
        return ["mean_close_mid_abs_rel_negative"]
    return []


def northset_book_shape_finite_rates_honesty_errors(blob: object) -> list[str]:
    """Soft-verify northset book-shape finite_rate companions ∈ [0, 1].

    Keys: concentration_top_finite_rate, queue_priority_finite_rate,
    side_notional_finite_rate, tob_size_share_finite_rate.
    (depth_shape_finite_rate / gap_finite_rate have their own helpers.)
    NaN/absent skipped; ±inf / out-of-range fail. Research diagnostic only;
    never live Sharpe. Does not touch kyle_* / book_metrics METRICS_* keys.
    """
    if not isinstance(blob, dict):
        return []
    errs: list[str] = []
    for key in (
        "concentration_top_finite_rate",
        "queue_priority_finite_rate",
        "side_notional_finite_rate",
        "tob_size_share_finite_rate",
    ):
        if key not in blob:
            continue
        try:
            x = float(blob.get(key))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append(f"{key}_non_numeric")
            continue
        if x != x:
            continue
        if abs(x) == float("inf"):
            errs.append(f"{key}_non_finite")
            continue
        if not (0.0 <= x <= 1.0):
            errs.append(f"{key}_out_of_unit_interval")
    return errs


def session_volume_conservation_vs_reconstructs_never_equate_honesty_errors(
    blob: object,
) -> list[str]:
    """Never equate H24 volume conservation with H23 session reconstructs.

    Sibling rates on the northset receipt — volume conservation ≠ OHLC envelope
    reconstructs (DATA_CONTRACTS). When **both** keys are present:

    - Catalog bind lock: reconstructs → H23, conservation → H24 (distinct hyp ids)
    - Numeric equality on clean synth is allowed (never fail on value equality)
    - Distinct key identity is structural (``session_reconstructs_daily_rate`` ≠
      ``session_volume_conservation_rate``)

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_recon = "session_reconstructs_daily_rate"
    k_vol = "session_volume_conservation_rate"
    if k_recon not in blob or k_vol not in blob:
        return []

    errs: list[str] = []
    if k_recon == k_vol:
        errs.append("session_reconstructs_volume_conservation_keys_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    recon_hyp = by_key.get(k_recon)
    vol_hyp = by_key.get(k_vol)
    if recon_hyp != H23_HYPOTHESIS_ID:
        errs.append("session_reconstructs_daily_rate_not_bound_h23")
    if vol_hyp != H24_HYPOTHESIS_ID:
        errs.append("session_volume_conservation_rate_not_bound_h24")
    if recon_hyp is not None and vol_hyp is not None and recon_hyp == vol_hyp:
        errs.append("session_reconstructs_volume_conservation_hypothesis_collapsed")
    return errs


def session_volume_conservation_rate_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: session_volume_conservation_rate ∈ [0, 1] when finite.

    Session volume conservation sibling of reconstructs/chain rates.
    NaN/absent skip. Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    val = blob.get("session_volume_conservation_rate")
    try:
        x = float(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return []
    if x != x:
        return []
    if abs(x) == float("inf") or not (0.0 <= x <= 1.0):
        return ["session_volume_conservation_rate_out_of_unit_interval"]
    return []


def session_ohlc_vs_reconstructs_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate ``session_ohlc_identity_rate`` with ``session_reconstructs_daily_rate``.

    DATA_CONTRACTS: session-candle OHLC identity ≠ session envelope reconstructing
    the daily bar. When both keys are present:

    - Distinct key identity
    - H23 binds to reconstructs only (``H23_HYPOTHESIS_ID``); session OHLC is not H23
    - Numeric equality on clean synth is allowed

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_ohlc = "session_ohlc_identity_rate"
    k_recon = "session_reconstructs_daily_rate"
    if k_ohlc not in blob or k_recon not in blob:
        return []

    errs: list[str] = []
    if k_ohlc == k_recon:
        errs.append("session_ohlc_reconstructs_keys_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_recon) != H23_HYPOTHESIS_ID:
        errs.append("session_reconstructs_daily_rate_not_bound_h23")
    if by_key.get(k_ohlc) == H23_HYPOTHESIS_ID:
        errs.append("session_ohlc_identity_rate_incorrectly_bound_h23")
    return errs


def session_ohlc_vs_volume_conservation_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate ``session_ohlc_identity_rate`` with ``session_volume_conservation_rate``.

    DATA_CONTRACTS: session-candle OHLC identity ≠ session volume conservation
    (H24). When both keys are present:

    - Distinct key identity
    - H24 binds to volume conservation only (``H24_HYPOTHESIS_ID``); session OHLC
      is not H23/H24
    - Numeric equality on clean synth is allowed

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_ohlc = "session_ohlc_identity_rate"
    k_vol = "session_volume_conservation_rate"
    if k_ohlc not in blob or k_vol not in blob:
        return []

    errs: list[str] = []
    if k_ohlc == k_vol:
        errs.append("session_ohlc_volume_conservation_keys_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_vol) != H24_HYPOTHESIS_ID:
        errs.append("session_volume_conservation_rate_not_bound_h24")
    if by_key.get(k_ohlc) == H24_HYPOTHESIS_ID:
        errs.append("session_ohlc_identity_rate_incorrectly_bound_h24")
    if by_key.get(k_ohlc) == H23_HYPOTHESIS_ID:
        errs.append("session_ohlc_identity_rate_incorrectly_bound_h23")
    return errs


def session_ohlc_vs_session_chain_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate ``session_ohlc_identity_rate`` with ``session_chain_rate``.

    DATA_CONTRACTS: session-candle OHLC identity ≠ H29 session chain rate.
    When both keys are present:

    - Distinct key identity
    - H29 binds to chain only (``H29_HYPOTHESIS_ID``); session OHLC is not H23/H29
    - Numeric equality on clean synth is allowed

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_ohlc = "session_ohlc_identity_rate"
    k_chain = "session_chain_rate"
    if k_ohlc not in blob or k_chain not in blob:
        return []

    errs: list[str] = []
    if k_ohlc == k_chain:
        errs.append("session_ohlc_session_chain_keys_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_chain) != H29_HYPOTHESIS_ID:
        errs.append("session_chain_rate_not_bound_h29")
    if by_key.get(k_ohlc) == H29_HYPOTHESIS_ID:
        errs.append("session_ohlc_identity_rate_incorrectly_bound_h29")
    if by_key.get(k_ohlc) == H23_HYPOTHESIS_ID:
        errs.append("session_ohlc_identity_rate_incorrectly_bound_h23")
    return errs


def session_reconstructs_daily_rate_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: session_reconstructs_daily_rate ∈ [0, 1] when finite.

    Session→daily reconstruction rate — ≠ session_ohlc_identity_rate.
    NaN/absent skip. Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    val = blob.get("session_reconstructs_daily_rate")
    try:
        x = float(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return []
    if x != x:
        return []
    if abs(x) == float("inf") or not (0.0 <= x <= 1.0):
        return ["session_reconstructs_daily_rate_out_of_unit_interval"]
    return []


def book_uncrossed_rate_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: book_uncrossed_rate ∈ [0, 1] when finite.

    Northset book integrity rate (H21 companion). NaN/absent skip.
    Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    val = blob.get("book_uncrossed_rate")
    try:
        x = float(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return []
    if x != x:
        return []
    if abs(x) == float("inf") or not (0.0 <= x <= 1.0):
        return ["book_uncrossed_rate_out_of_unit_interval"]
    return []


def book_uncrossed_vs_imbalance_p_ic_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate H21 ``book_uncrossed_rate`` with H22 ``imbalance_top_p_ic``.

    Triad siblings: uncrossed book bound ≠ imbalance discovery IC p-value.
    When both keys are present:

    - Distinct key identity
    - Distinct hyp ids (H21 ≠ H22)
    - Gate helpers: H21 on book_uncrossed only; H22 on imbalance_top_p_ic only
    - Numeric equality is allowed (different units; rare)

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_book = "book_uncrossed_rate"
    k_imb = "imbalance_top_p_ic"
    if k_book not in blob or k_imb not in blob:
        return []

    errs: list[str] = []
    if k_book == k_imb:
        errs.append("book_uncrossed_imbalance_p_ic_keys_collapsed")
    if H21_HYPOTHESIS_ID == H22_HYPOTHESIS_ID:
        errs.append("h21_h22_hypothesis_ids_collapsed")

    elig = {"book_hypothesis_eligible": True}
    if not northset_has_finite_book_uncrossed_rate({k_book: 1.0, **elig}):
        errs.append("book_uncrossed_h21_gate_helper_broken")
    if northset_has_finite_book_uncrossed_rate({k_imb: 1.0, **elig}):
        errs.append("imbalance_p_ic_incorrectly_triggers_h21_gate")
    if not northset_has_finite_imbalance_p_ic({k_imb: 0.05, **elig}):
        errs.append("imbalance_p_ic_h22_gate_helper_broken")
    if northset_has_finite_imbalance_p_ic({k_book: 1.0, **elig}):
        errs.append("book_uncrossed_incorrectly_triggers_h22_gate")
    return errs


def ohlc_identity_vs_imbalance_p_ic_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate H20 ``ohlc_identity_rate`` with H22 ``imbalance_top_p_ic``.

    Triad diagonal: OHLC envelope bound ≠ imbalance discovery IC p-value.
    Completes H20↔H21↔H22 never-equate mesh (edges already landed). When both
    keys are present:

    - Distinct key identity
    - Distinct hyp ids (H20 ≠ H22)
    - Gate helpers: H20 on ohlc only; H22 on imbalance_top_p_ic only
    - Numeric equality is allowed (different units; rare)

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_ohlc = "ohlc_identity_rate"
    k_imb = "imbalance_top_p_ic"
    if k_ohlc not in blob or k_imb not in blob:
        return []

    errs: list[str] = []
    if k_ohlc == k_imb:
        errs.append("ohlc_identity_imbalance_p_ic_keys_collapsed")
    if H20_HYPOTHESIS_ID == H22_HYPOTHESIS_ID:
        errs.append("h20_h22_hypothesis_ids_collapsed")

    if not northset_has_finite_ohlc_identity_rate({k_ohlc: 1.0}):
        errs.append("ohlc_identity_h20_gate_helper_broken")
    if northset_has_finite_ohlc_identity_rate({k_imb: 0.05}):
        errs.append("imbalance_p_ic_incorrectly_triggers_h20_gate")
    elig = {"book_hypothesis_eligible": True}
    if not northset_has_finite_imbalance_p_ic({k_imb: 0.05, **elig}):
        errs.append("imbalance_p_ic_h22_gate_helper_broken")
    if northset_has_finite_imbalance_p_ic({k_ohlc: 1.0, **elig}):
        errs.append("ohlc_identity_incorrectly_triggers_h22_gate")
    return errs


def session_bulk_vpin_vs_siblings_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate ``session_bulk_vpin`` with ``vpin_mean`` / ``session_book_vpin_mean``.

    DATA_CONTRACTS VPIN triad: candle signed-volume bulk ≠ daily Cont proxy mean
    (H32 path) ≠ session-L2 ``|ofi_sum|/ofi_abs_sum`` mean (H43 path). Bulk has
    **no** H-id. When ``session_bulk_vpin`` and at least one sibling are present:

    - Distinct key identity vs each present sibling
    - Distinct hyp ids H32 ≠ H43
    - Bulk must not bind to H32/H43 metric keys (``vpin_p_ic`` / ``session_book_vpin_p_ic``)
    - H43 gate helper triggers on session_book_vpin_p_ic only, not bulk
    - Numeric equality on synth is allowed

    Skip when bulk absent or both siblings absent. Research diagnostic only;
    never live Sharpe. Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_bulk = "session_bulk_vpin"
    k_daily = "vpin_mean"
    k_sess = "session_book_vpin_mean"
    if k_bulk not in blob:
        return []
    if k_daily not in blob and k_sess not in blob:
        return []

    errs: list[str] = []
    if k_daily in blob and k_bulk == k_daily:
        errs.append("session_bulk_vpin_vpin_mean_keys_collapsed")
    if k_sess in blob and k_bulk == k_sess:
        errs.append("session_bulk_vpin_session_book_vpin_mean_keys_collapsed")
    if k_daily in blob and k_sess in blob and k_daily == k_sess:
        errs.append("vpin_mean_session_book_vpin_mean_keys_collapsed")
    if H32_HYPOTHESIS_ID == H43_HYPOTHESIS_ID:
        errs.append("h32_h43_hypothesis_ids_collapsed")

    # Catalog bind lock: H32/H43 attach to *_p_ic, never to session_bulk_vpin
    h32_metric = "vpin_p_ic"
    h43_metric = "session_book_vpin_p_ic"
    if k_bulk in {h32_metric, h43_metric}:
        errs.append("session_bulk_vpin_incorrectly_bound_as_h32_or_h43_metric")
    if h32_metric == k_bulk or h43_metric == k_bulk:
        errs.append("session_bulk_vpin_metric_key_collapsed_into_h_gate")

    if northset_has_finite_session_book_vpin_p_ic({k_bulk: 0.5}):
        errs.append("session_bulk_vpin_incorrectly_triggers_h43_gate")
    if not northset_has_finite_session_book_vpin_p_ic(
        {h43_metric: 0.05, "session_book_hypothesis_eligible": True}
    ):
        # eligible default True in helper — still require p_ic key
        errs.append("session_book_vpin_p_ic_h43_gate_helper_broken")
    return errs


def ohlc_identity_vs_book_uncrossed_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate H20 ``ohlc_identity_rate`` with H21 ``book_uncrossed_rate``.

    Triad siblings (DATA_CONTRACTS H20/H21/H22): OHLC envelope ≠ uncrossed book.
    When both keys are present:

    - Distinct key identity
    - Distinct hyp ids (H20 ≠ H21)
    - H20 finite gate helper triggers on ohlc only; H21 gate on book_uncrossed only
    - Numeric equality on clean synth is allowed

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_ohlc = "ohlc_identity_rate"
    k_book = "book_uncrossed_rate"
    if k_ohlc not in blob or k_book not in blob:
        return []

    errs: list[str] = []
    if k_ohlc == k_book:
        errs.append("ohlc_identity_book_uncrossed_keys_collapsed")
    if H20_HYPOTHESIS_ID == H21_HYPOTHESIS_ID:
        errs.append("h20_h21_hypothesis_ids_collapsed")

    if not northset_has_finite_ohlc_identity_rate({k_ohlc: 1.0}):
        errs.append("ohlc_identity_h20_gate_helper_broken")
    if northset_has_finite_ohlc_identity_rate({k_book: 1.0}):
        errs.append("book_uncrossed_incorrectly_triggers_h20_gate")
    if not northset_has_finite_book_uncrossed_rate({k_book: 1.0, "book_hypothesis_eligible": True}):
        errs.append("book_uncrossed_h21_gate_helper_broken")
    if northset_has_finite_book_uncrossed_rate({k_ohlc: 1.0, "book_hypothesis_eligible": True}):
        errs.append("ohlc_identity_incorrectly_triggers_h21_gate")
    return errs


def ohlc_identity_vs_gap_finite_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate ``ohlc_identity_rate`` with ``gap_finite_rate``.

    DATA_CONTRACTS: OHLC envelope identity ≠ overnight gap finiteness. When both
    keys are present:

    - Distinct key identity
    - H20 finite gate binds to ``ohlc_identity_rate`` only (gap has **no** H20 claim)
    - Numeric equality on clean synth is allowed

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_ohlc = "ohlc_identity_rate"
    k_gap = "gap_finite_rate"
    if k_ohlc not in blob or k_gap not in blob:
        return []

    errs: list[str] = []
    if k_ohlc == k_gap:
        errs.append("ohlc_identity_gap_finite_keys_collapsed")

    # H20 watches daily ohlc only — gap must not share H20 id with session binds check
    if H20_HYPOTHESIS_ID in {H21_HYPOTHESIS_ID, H22_HYPOTHESIS_ID}:
        errs.append("h20_hypothesis_collapsed_into_h21_h22")

    # Positive: northset_has_finite_ohlc_identity_rate uses ohlc key, not gap
    if not northset_has_finite_ohlc_identity_rate({k_ohlc: 1.0}):
        errs.append("ohlc_identity_h20_gate_helper_broken")
    if northset_has_finite_ohlc_identity_rate({k_gap: 1.0}):
        errs.append("gap_finite_rate_incorrectly_triggers_h20_gate")
    return errs


def gap_finite_rate_vs_book_uncrossed_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate ``gap_finite_rate`` with H21 ``book_uncrossed_rate``.

    DATA_CONTRACTS: overnight gap finiteness ≠ uncrossed book. When both keys
    are present:

    - Distinct key identity
    - Distinct hyp/gate: H21 binds to ``book_uncrossed_rate`` only (gap has
      **no** H21 claim); H21 id stays distinct from H20/H22
    - Numeric equality on clean synth is allowed

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off H20/H21/H22 triad polish (already landed). Off nest invent / kyle_ofi.
    """
    if not isinstance(blob, dict):
        return []
    k_gap = "gap_finite_rate"
    k_book = "book_uncrossed_rate"
    if k_gap not in blob or k_book not in blob:
        return []

    errs: list[str] = []
    if k_gap == k_book:
        errs.append("gap_finite_book_uncrossed_keys_collapsed")
    if H21_HYPOTHESIS_ID in {H20_HYPOTHESIS_ID, H22_HYPOTHESIS_ID}:
        errs.append("h21_hypothesis_collapsed_into_h20_h22")

    elig = {"book_hypothesis_eligible": True}
    if not northset_has_finite_book_uncrossed_rate({k_book: 1.0, **elig}):
        errs.append("book_uncrossed_h21_gate_helper_broken")
    if northset_has_finite_book_uncrossed_rate({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h21_gate")
    return errs


def imbalance_top_p_ic_vs_gap_finite_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate H22 ``imbalance_top_p_ic`` with ``gap_finite_rate``.

    DATA_CONTRACTS: imbalance discovery IC p-value ≠ overnight gap finiteness.
    When both keys are present:

    - Distinct key identity
    - Distinct hyp/gate: H22 binds imbalance_top_p_ic only; gap has **no** H22
      claim and must not trigger H20/H21 gates
    - Numeric equality on clean synth is allowed (different units; rare)

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_imb = "imbalance_top_p_ic"
    k_gap = "gap_finite_rate"
    if k_imb not in blob or k_gap not in blob:
        return []

    errs: list[str] = []
    if k_imb == k_gap:
        errs.append("imbalance_p_ic_gap_finite_keys_collapsed")
    if H22_HYPOTHESIS_ID in {H20_HYPOTHESIS_ID, H21_HYPOTHESIS_ID}:
        errs.append("h22_hypothesis_collapsed_into_h20_or_h21")

    elig = {"book_hypothesis_eligible": True}
    if not northset_has_finite_imbalance_p_ic({k_imb: 0.05, **elig}):
        errs.append("imbalance_p_ic_h22_gate_helper_broken")
    if northset_has_finite_imbalance_p_ic({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h22_gate")

    if northset_has_finite_ohlc_identity_rate({k_gap: 1.0}):
        errs.append("gap_finite_rate_incorrectly_triggers_h20_gate")
    if northset_has_finite_ohlc_identity_rate({k_imb: 0.05}):
        errs.append("imbalance_p_ic_incorrectly_triggers_h20_gate")

    if northset_has_finite_book_uncrossed_rate({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h21_gate")
    if northset_has_finite_book_uncrossed_rate({k_imb: 0.05, **elig}):
        errs.append("imbalance_p_ic_incorrectly_triggers_h21_gate")
    return errs


def imbalance_top_p_ic_vs_session_reconstructs_never_equate_honesty_errors(
    blob: object,
) -> list[str]:
    """Never equate H22 ``imbalance_top_p_ic`` with H23 ``session_reconstructs_daily_rate``.

    DATA_CONTRACTS: imbalance discovery IC p-value ≠ session envelope reconstructing
    the daily bar. When both keys are present:

    - Distinct key identity
    - Distinct hyp ids (H22 ≠ H23)
    - H23 binds reconstructs only; imbalance must not bind H23
    - H22 gate on imbalance_top_p_ic only; reconstructs must not trigger H22
    - Numeric equality on clean synth is allowed

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_imb = "imbalance_top_p_ic"
    k_recon = "session_reconstructs_daily_rate"
    if k_imb not in blob or k_recon not in blob:
        return []

    errs: list[str] = []
    if k_imb == k_recon:
        errs.append("imbalance_p_ic_session_reconstructs_keys_collapsed")
    if H22_HYPOTHESIS_ID == H23_HYPOTHESIS_ID:
        errs.append("h22_h23_hypothesis_ids_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_recon) != H23_HYPOTHESIS_ID:
        errs.append("session_reconstructs_daily_rate_not_bound_h23")
    if by_key.get(k_imb) == H23_HYPOTHESIS_ID:
        errs.append("imbalance_top_p_ic_incorrectly_bound_h23")

    elig = {"book_hypothesis_eligible": True}
    if not northset_has_finite_imbalance_p_ic({k_imb: 0.05, **elig}):
        errs.append("imbalance_p_ic_h22_gate_helper_broken")
    if northset_has_finite_imbalance_p_ic({k_recon: 1.0, **elig}):
        errs.append("session_reconstructs_incorrectly_triggers_h22_gate")
    return errs


def microprice_p_ic_vs_gap_finite_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate H25 ``microprice_p_ic`` with ``gap_finite_rate``.

    DATA_CONTRACTS: microprice discovery IC p ≠ overnight gap finiteness.
    When both keys are present:

    - Distinct key identity
    - H25 binds ``microprice_p_ic`` only (gap has **no** H25 claim)
    - No dedicated microprice finite-gate helper — ``_finite_scalar`` on the
      ``microprice_p_ic`` key only (a gap-only blob must not count as finite microprice)
    - Gap must not incorrectly trigger H20 / H21 / H22 gates
    - Numeric equality on clean synth is allowed (different units; rare)

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_mp = "microprice_p_ic"
    k_gap = "gap_finite_rate"
    if k_mp not in blob or k_gap not in blob:
        return []

    errs: list[str] = []
    if k_mp == k_gap:
        errs.append("microprice_p_ic_gap_finite_keys_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_mp) != H25_HYPOTHESIS_ID:
        errs.append("microprice_p_ic_not_bound_h25")
    if by_key.get(k_gap) == H25_HYPOTHESIS_ID:
        errs.append("gap_finite_rate_incorrectly_bound_h25")
    if H25_HYPOTHESIS_ID in {H20_HYPOTHESIS_ID, H21_HYPOTHESIS_ID, H22_HYPOTHESIS_ID}:
        errs.append("h25_hypothesis_collapsed_into_h20_h21_h22")

    if not _finite_scalar({k_mp: 0.05}.get(k_mp)):
        errs.append("microprice_p_ic_finite_probe_broken")
    if _finite_scalar({k_gap: 1.0}.get(k_mp)):
        errs.append("gap_finite_rate_incorrectly_counts_as_finite_microprice")

    elig = {"book_hypothesis_eligible": True}
    if northset_has_finite_imbalance_p_ic({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h22_gate")
    if northset_has_finite_ohlc_identity_rate({k_gap: 1.0}):
        errs.append("gap_finite_rate_incorrectly_triggers_h20_gate")
    if northset_has_finite_book_uncrossed_rate({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h21_gate")
    return errs


def microprice_p_ic_vs_session_chain_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate H25 ``microprice_p_ic`` with H29 ``session_chain_rate``.

    DATA_CONTRACTS: microprice discovery IC p ≠ session reconstruction chain.
    Mirrors :func:`imbalance_top_p_ic_vs_session_chain_never_equate_honesty_errors`
    with H25 bind + key-scoped ``_finite_scalar`` (no dedicated microprice gate).
    When both keys are present:

    - Distinct key identity
    - Distinct hyp ids (H25 ≠ H29)
    - H25 binds ``microprice_p_ic`` only; H29 binds chain only
    - Chain must not count as finite microprice via key-scoped probe
    - Numeric equality on clean synth is allowed (different units)

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite. Off Commander #246+ identity batch.
    """
    if not isinstance(blob, dict):
        return []
    k_mp = "microprice_p_ic"
    k_chain = "session_chain_rate"
    if k_mp not in blob or k_chain not in blob:
        return []

    errs: list[str] = []
    if k_mp == k_chain:
        errs.append("microprice_p_ic_session_chain_keys_collapsed")
    if H25_HYPOTHESIS_ID == H29_HYPOTHESIS_ID:
        errs.append("h25_h29_hypothesis_ids_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_mp) != H25_HYPOTHESIS_ID:
        errs.append("microprice_p_ic_not_bound_h25")
    if by_key.get(k_chain) != H29_HYPOTHESIS_ID:
        errs.append("session_chain_rate_not_bound_h29")
    if by_key.get(k_mp) == H29_HYPOTHESIS_ID:
        errs.append("microprice_p_ic_incorrectly_bound_h29")
    if by_key.get(k_chain) == H25_HYPOTHESIS_ID:
        errs.append("session_chain_rate_incorrectly_bound_h25")

    if not _finite_scalar({k_mp: 0.05}.get(k_mp)):
        errs.append("microprice_p_ic_finite_probe_broken")
    if _finite_scalar({k_chain: 1.0}.get(k_mp)):
        errs.append("session_chain_incorrectly_counts_as_finite_microprice")
    return errs


def clv_p_ic_vs_gap_finite_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate H30 ``clv_p_ic`` with ``gap_finite_rate``.

    DATA_CONTRACTS: CLV discovery IC p ≠ overnight gap finiteness.
    Mirrors :func:`microprice_p_ic_vs_gap_finite_never_equate_honesty_errors`.
    When both keys are present:

    - Distinct key identity
    - H30 binds ``clv_p_ic`` only (gap has **no** H30 claim)
    - No dedicated clv finite-gate helper — ``_finite_scalar`` on the
      ``clv_p_ic`` key only (a gap-only blob must not count as finite clv)
    - Gap must not incorrectly trigger H20 / H21 / H22 gates
    - Numeric equality on clean synth is allowed (different units; rare)

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite. Not Commander #236+; not Sergeant
    ofi↔gap; not Lt wick↔gap.
    """
    if not isinstance(blob, dict):
        return []
    k_clv = "clv_p_ic"
    k_gap = "gap_finite_rate"
    if k_clv not in blob or k_gap not in blob:
        return []

    errs: list[str] = []
    if k_clv == k_gap:
        errs.append("clv_p_ic_gap_finite_keys_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_clv) != H30_HYPOTHESIS_ID:
        errs.append("clv_p_ic_not_bound_h30")
    if by_key.get(k_gap) == H30_HYPOTHESIS_ID:
        errs.append("gap_finite_rate_incorrectly_bound_h30")
    if H30_HYPOTHESIS_ID in {H20_HYPOTHESIS_ID, H21_HYPOTHESIS_ID, H22_HYPOTHESIS_ID}:
        errs.append("h30_hypothesis_collapsed_into_h20_h21_h22")

    if not _finite_scalar({k_clv: 0.05}.get(k_clv)):
        errs.append("clv_p_ic_finite_probe_broken")
    if _finite_scalar({k_gap: 1.0}.get(k_clv)):
        errs.append("gap_finite_rate_incorrectly_counts_as_finite_clv")

    elig = {"book_hypothesis_eligible": True}
    if northset_has_finite_imbalance_p_ic({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h22_gate")
    if northset_has_finite_ohlc_identity_rate({k_gap: 1.0}):
        errs.append("gap_finite_rate_incorrectly_triggers_h20_gate")
    if northset_has_finite_book_uncrossed_rate({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h21_gate")
    return errs


def dm_split_vs_park_p_vs_gap_finite_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate H31 ``dm_split_vs_park_p`` with ``gap_finite_rate``.

    DATA_CONTRACTS: overnight-split vs Parkinson DM p ≠ overnight gap finiteness.
    Mirrors :func:`clv_p_ic_vs_gap_finite_never_equate_honesty_errors`.
    When both keys are present:

    - Distinct key identity
    - H31 binds ``dm_split_vs_park_p`` only (gap has **no** H31 claim)
    - No dedicated dm_split finite-gate helper — ``_finite_scalar`` on the
      ``dm_split_vs_park_p`` key only (a gap-only blob must not count as finite dm)
    - Gap must not incorrectly trigger H20 / H21 / H22 gates
    - Numeric equality on clean synth is allowed (different units; rare)

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite. Not Sergeant H28; not Lt H33 sweep.
    """
    if not isinstance(blob, dict):
        return []
    k_dm = "dm_split_vs_park_p"
    k_gap = "gap_finite_rate"
    if k_dm not in blob or k_gap not in blob:
        return []

    errs: list[str] = []
    if k_dm == k_gap:
        errs.append("dm_split_vs_park_p_gap_finite_keys_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_dm) != H31_HYPOTHESIS_ID:
        errs.append("dm_split_vs_park_p_not_bound_h31")
    if by_key.get(k_gap) == H31_HYPOTHESIS_ID:
        errs.append("gap_finite_rate_incorrectly_bound_h31")
    if H31_HYPOTHESIS_ID in {H20_HYPOTHESIS_ID, H21_HYPOTHESIS_ID, H22_HYPOTHESIS_ID}:
        errs.append("h31_hypothesis_collapsed_into_h20_h21_h22")

    if not _finite_scalar({k_dm: 0.05}.get(k_dm)):
        errs.append("dm_split_vs_park_p_finite_probe_broken")
    if _finite_scalar({k_gap: 1.0}.get(k_dm)):
        errs.append("gap_finite_rate_incorrectly_counts_as_finite_dm_split")

    elig = {"book_hypothesis_eligible": True}
    if northset_has_finite_imbalance_p_ic({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h22_gate")
    if northset_has_finite_ohlc_identity_rate({k_gap: 1.0}):
        errs.append("gap_finite_rate_incorrectly_triggers_h20_gate")
    if northset_has_finite_book_uncrossed_rate({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h21_gate")
    return errs


def sweep_follow_event_p_vs_gap_finite_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate H36 ``sweep_follow_event_p`` with ``gap_finite_rate``.

    DATA_CONTRACTS: sweep-follow event p ≠ overnight gap finiteness.
    Mirrors :func:`clv_p_ic_vs_gap_finite_never_equate_honesty_errors`.
    When both keys are present:

    - Distinct key identity
    - H36 binds ``sweep_follow_event_p`` only (gap has **no** H36 claim)
    - No dedicated sweep_follow_event finite-gate helper — ``_finite_scalar`` on
      the ``sweep_follow_event_p`` key only (a gap-only blob must not count as
      finite sweep_follow_event_p)
    - Gap must not incorrectly trigger H20 / H21 / H22 gates
    - Numeric equality on clean synth is allowed (different units; rare)

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite. Not Sergeant H35; not Lt H34.
    """
    if not isinstance(blob, dict):
        return []
    k_ev = "sweep_follow_event_p"
    k_gap = "gap_finite_rate"
    if k_ev not in blob or k_gap not in blob:
        return []

    errs: list[str] = []
    if k_ev == k_gap:
        errs.append("sweep_follow_event_p_gap_finite_keys_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_ev) != H36_HYPOTHESIS_ID:
        errs.append("sweep_follow_event_p_not_bound_h36")
    if by_key.get(k_gap) == H36_HYPOTHESIS_ID:
        errs.append("gap_finite_rate_incorrectly_bound_h36")
    if H36_HYPOTHESIS_ID in {H20_HYPOTHESIS_ID, H21_HYPOTHESIS_ID, H22_HYPOTHESIS_ID}:
        errs.append("h36_hypothesis_collapsed_into_h20_h21_h22")

    if not _finite_scalar({k_ev: 0.05}.get(k_ev)):
        errs.append("sweep_follow_event_p_finite_probe_broken")
    if _finite_scalar({k_gap: 1.0}.get(k_ev)):
        errs.append("gap_finite_rate_incorrectly_counts_as_finite_sweep_follow_event_p")

    elig = {"book_hypothesis_eligible": True}
    if northset_has_finite_imbalance_p_ic({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h22_gate")
    if northset_has_finite_ohlc_identity_rate({k_gap: 1.0}):
        errs.append("gap_finite_rate_incorrectly_triggers_h20_gate")
    if northset_has_finite_book_uncrossed_rate({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h21_gate")
    return errs


def sweep_reject_cost_adjusted_mean_bps_vs_gap_finite_never_equate_honesty_errors(
    blob: object,
) -> list[str]:
    """Never equate H39 ``sweep_reject_cost_adjusted_mean_bps`` with ``gap_finite_rate``.

    DATA_CONTRACTS: sweep-reject cost-adjusted mean bps ≠ overnight gap finiteness.
    Mirrors :func:`sweep_follow_event_p_vs_gap_finite_never_equate_honesty_errors`.
    When both keys are present:

    - Distinct key identity
    - H39 binds ``sweep_reject_cost_adjusted_mean_bps`` only (gap has **no** H39 claim)
    - No dedicated reject-cost finite-gate helper — ``_finite_scalar`` on the
      ``sweep_reject_cost_adjusted_mean_bps`` key only (a gap-only blob must not
      count as finite reject-cost bps)
    - Gap must not incorrectly trigger H20 / H21 / H22 gates
    - Numeric equality on clean synth is allowed (different units; rare)

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite. Not Sergeant H38.
    """
    if not isinstance(blob, dict):
        return []
    k_cost = "sweep_reject_cost_adjusted_mean_bps"
    k_gap = "gap_finite_rate"
    if k_cost not in blob or k_gap not in blob:
        return []

    errs: list[str] = []
    if k_cost == k_gap:
        errs.append("sweep_reject_cost_adjusted_mean_bps_gap_finite_keys_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_cost) != H39_HYPOTHESIS_ID:
        errs.append("sweep_reject_cost_adjusted_mean_bps_not_bound_h39")
    if by_key.get(k_gap) == H39_HYPOTHESIS_ID:
        errs.append("gap_finite_rate_incorrectly_bound_h39")
    if H39_HYPOTHESIS_ID in {H20_HYPOTHESIS_ID, H21_HYPOTHESIS_ID, H22_HYPOTHESIS_ID}:
        errs.append("h39_hypothesis_collapsed_into_h20_h21_h22")

    if not _finite_scalar({k_cost: 1.0}.get(k_cost)):
        errs.append("sweep_reject_cost_adjusted_mean_bps_finite_probe_broken")
    if _finite_scalar({k_gap: 1.0}.get(k_cost)):
        errs.append(
            "gap_finite_rate_incorrectly_counts_as_finite_sweep_reject_cost_adjusted_mean_bps"
        )

    elig = {"book_hypothesis_eligible": True}
    if northset_has_finite_imbalance_p_ic({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h22_gate")
    if northset_has_finite_ohlc_identity_rate({k_gap: 1.0}):
        errs.append("gap_finite_rate_incorrectly_triggers_h20_gate")
    if northset_has_finite_book_uncrossed_rate({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h21_gate")
    return errs


def sweep_reject_control_diff_p_vs_gap_finite_never_equate_honesty_errors(
    blob: object,
) -> list[str]:
    """Never equate H44 ``sweep_reject_control_diff_p`` with ``gap_finite_rate``.

    DATA_CONTRACTS: sweep-reject control-diff p ≠ overnight gap finiteness.
    Mirrors :func:`sweep_reject_cost_adjusted_mean_bps_vs_gap_finite_never_equate_honesty_errors`.
    When both keys are present:

    - Distinct key identity
    - H44 binds ``sweep_reject_control_diff_p`` only (gap has **no** H44 claim)
    - No dedicated reject-control finite-gate helper — ``_finite_scalar`` on the
      ``sweep_reject_control_diff_p`` key only (a gap-only blob must not count as
      finite reject-control p)
    - Gap must not incorrectly trigger H20 / H21 / H22 gates
    - Numeric equality on clean synth is allowed (different units; rare)

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite. Not Sergeant H42.
    """
    if not isinstance(blob, dict):
        return []
    k_ctrl = "sweep_reject_control_diff_p"
    k_gap = "gap_finite_rate"
    if k_ctrl not in blob or k_gap not in blob:
        return []

    errs: list[str] = []
    if k_ctrl == k_gap:
        errs.append("sweep_reject_control_diff_p_gap_finite_keys_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_ctrl) != H44_HYPOTHESIS_ID:
        errs.append("sweep_reject_control_diff_p_not_bound_h44")
    if by_key.get(k_gap) == H44_HYPOTHESIS_ID:
        errs.append("gap_finite_rate_incorrectly_bound_h44")
    if H44_HYPOTHESIS_ID in {H20_HYPOTHESIS_ID, H21_HYPOTHESIS_ID, H22_HYPOTHESIS_ID}:
        errs.append("h44_hypothesis_collapsed_into_h20_h21_h22")

    if not _finite_scalar({k_ctrl: 0.05}.get(k_ctrl)):
        errs.append("sweep_reject_control_diff_p_finite_probe_broken")
    if _finite_scalar({k_gap: 1.0}.get(k_ctrl)):
        errs.append("gap_finite_rate_incorrectly_counts_as_finite_sweep_reject_control_diff_p")

    elig = {"book_hypothesis_eligible": True}
    if northset_has_finite_imbalance_p_ic({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h22_gate")
    if northset_has_finite_ohlc_identity_rate({k_gap: 1.0}):
        errs.append("gap_finite_rate_incorrectly_triggers_h20_gate")
    if northset_has_finite_book_uncrossed_rate({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h21_gate")
    return errs


def ofi_p_ic_vs_gap_finite_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate H27 ``ofi_p_ic`` with ``gap_finite_rate``.

    DATA_CONTRACTS: OFI discovery IC p ≠ overnight gap finiteness.
    When both keys are present:

    - Distinct key identity
    - H27 binds ``ofi_p_ic`` only (gap has **no** H27 claim)
    - No dedicated ofi finite-gate helper — ``_finite_scalar`` on the
      ``ofi_p_ic`` key only (a gap-only blob must not count as finite ofi)
    - Gap must not incorrectly trigger H20 / H21 / H22 gates
    - Numeric equality on clean synth is allowed (different units; rare)

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite. Not Commander #211+ props.
    """
    if not isinstance(blob, dict):
        return []
    k_ofi = "ofi_p_ic"
    k_gap = "gap_finite_rate"
    if k_ofi not in blob or k_gap not in blob:
        return []

    errs: list[str] = []
    if k_ofi == k_gap:
        errs.append("ofi_p_ic_gap_finite_keys_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_ofi) != H27_HYPOTHESIS_ID:
        errs.append("ofi_p_ic_not_bound_h27")
    if by_key.get(k_gap) == H27_HYPOTHESIS_ID:
        errs.append("gap_finite_rate_incorrectly_bound_h27")
    if H27_HYPOTHESIS_ID in {H20_HYPOTHESIS_ID, H21_HYPOTHESIS_ID, H22_HYPOTHESIS_ID}:
        errs.append("h27_hypothesis_collapsed_into_h20_h21_h22")

    if not _finite_scalar({k_ofi: 0.05}.get(k_ofi)):
        errs.append("ofi_p_ic_finite_probe_broken")
    if _finite_scalar({k_gap: 1.0}.get(k_ofi)):
        errs.append("gap_finite_rate_incorrectly_counts_as_finite_ofi")

    elig = {"book_hypothesis_eligible": True}
    if northset_has_finite_imbalance_p_ic({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h22_gate")
    if northset_has_finite_ohlc_identity_rate({k_gap: 1.0}):
        errs.append("gap_finite_rate_incorrectly_triggers_h20_gate")
    if northset_has_finite_book_uncrossed_rate({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h21_gate")
    return errs


def dm_gk_vs_park_p_vs_gap_finite_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate H28 ``dm_gk_vs_park_p`` with ``gap_finite_rate``.

    DATA_CONTRACTS: GK vs Park DM p-value ≠ overnight gap finiteness.
    When both keys are present:

    - Distinct key identity
    - H28 binds ``dm_gk_vs_park_p`` only (gap has **no** H28 claim)
    - No dedicated DM finite-gate helper — ``_finite_scalar`` on the
      ``dm_gk_vs_park_p`` key only (a gap-only blob must not count as finite DM p)
    - Gap must not incorrectly trigger H20 / H21 / H22 gates
    - Numeric equality on clean synth is allowed (different units; rare)

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off CoS dm_split H31; off Lt sweep H33; off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_dm = "dm_gk_vs_park_p"
    k_gap = "gap_finite_rate"
    if k_dm not in blob or k_gap not in blob:
        return []

    errs: list[str] = []
    if k_dm == k_gap:
        errs.append("dm_gk_vs_park_p_gap_finite_keys_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_dm) != H28_HYPOTHESIS_ID:
        errs.append("dm_gk_vs_park_p_not_bound_h28")
    if by_key.get(k_gap) == H28_HYPOTHESIS_ID:
        errs.append("gap_finite_rate_incorrectly_bound_h28")
    if H28_HYPOTHESIS_ID in {H20_HYPOTHESIS_ID, H21_HYPOTHESIS_ID, H22_HYPOTHESIS_ID}:
        errs.append("h28_hypothesis_collapsed_into_h20_h21_h22")

    if not _finite_scalar({k_dm: 0.05}.get(k_dm)):
        errs.append("dm_gk_vs_park_p_finite_probe_broken")
    if _finite_scalar({k_gap: 1.0}.get(k_dm)):
        errs.append("gap_finite_rate_incorrectly_counts_as_finite_dm_gk")

    elig = {"book_hypothesis_eligible": True}
    if northset_has_finite_imbalance_p_ic({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h22_gate")
    if northset_has_finite_ohlc_identity_rate({k_gap: 1.0}):
        errs.append("gap_finite_rate_incorrectly_triggers_h20_gate")
    if northset_has_finite_book_uncrossed_rate({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h21_gate")
    return errs


def sweep_reject_event_p_vs_gap_finite_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate H35 ``sweep_reject_event_p`` with ``gap_finite_rate``.

    DATA_CONTRACTS: sweep reject-event IC p ≠ overnight gap finiteness.
    When both keys are present:

    - Distinct key identity
    - H35 binds ``sweep_reject_event_p`` only (gap has **no** H35 claim)
    - No dedicated sweep-p finite-gate helper — ``_finite_scalar`` on the
      ``sweep_reject_event_p`` key only (a gap-only blob must not count as finite reject p)
    - Gap must not incorrectly trigger H20 / H21 / H22 gates
    - Numeric equality on clean synth is allowed (different units; rare)

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off CoS H36; off Lt H34; off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_sw = "sweep_reject_event_p"
    k_gap = "gap_finite_rate"
    if k_sw not in blob or k_gap not in blob:
        return []

    errs: list[str] = []
    if k_sw == k_gap:
        errs.append("sweep_reject_event_p_gap_finite_keys_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_sw) != H35_HYPOTHESIS_ID:
        errs.append("sweep_reject_event_p_not_bound_h35")
    if by_key.get(k_gap) == H35_HYPOTHESIS_ID:
        errs.append("gap_finite_rate_incorrectly_bound_h35")
    if H35_HYPOTHESIS_ID in {H20_HYPOTHESIS_ID, H21_HYPOTHESIS_ID, H22_HYPOTHESIS_ID}:
        errs.append("h35_hypothesis_collapsed_into_h20_h21_h22")

    if not _finite_scalar({k_sw: 0.05}.get(k_sw)):
        errs.append("sweep_reject_event_p_finite_probe_broken")
    if _finite_scalar({k_gap: 1.0}.get(k_sw)):
        errs.append("gap_finite_rate_incorrectly_counts_as_finite_sweep_reject_event_p")

    elig = {"book_hypothesis_eligible": True}
    if northset_has_finite_imbalance_p_ic({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h22_gate")
    if northset_has_finite_ohlc_identity_rate({k_gap: 1.0}):
        errs.append("gap_finite_rate_incorrectly_triggers_h20_gate")
    if northset_has_finite_book_uncrossed_rate({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h21_gate")
    return errs


def sweep_follow_placebo_p_vs_gap_finite_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate H38 ``sweep_follow_placebo_p`` with ``gap_finite_rate``.

    DATA_CONTRACTS: sweep follow-placebo IC p ≠ overnight gap finiteness.
    When both keys are present:

    - Distinct key identity
    - H38 binds ``sweep_follow_placebo_p`` only (gap has **no** H38 claim)
    - No dedicated sweep-p finite-gate helper — ``_finite_scalar`` on the
      ``sweep_follow_placebo_p`` key only (a gap-only blob must not count as finite placebo p)
    - Gap must not incorrectly trigger H20 / H21 / H22 gates
    - Numeric equality on clean synth is allowed (different units; rare)

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off CoS H39 cost; off Lt H37; off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_sw = "sweep_follow_placebo_p"
    k_gap = "gap_finite_rate"
    if k_sw not in blob or k_gap not in blob:
        return []

    errs: list[str] = []
    if k_sw == k_gap:
        errs.append("sweep_follow_placebo_p_gap_finite_keys_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_sw) != H38_HYPOTHESIS_ID:
        errs.append("sweep_follow_placebo_p_not_bound_h38")
    if by_key.get(k_gap) == H38_HYPOTHESIS_ID:
        errs.append("gap_finite_rate_incorrectly_bound_h38")
    if H38_HYPOTHESIS_ID in {H20_HYPOTHESIS_ID, H21_HYPOTHESIS_ID, H22_HYPOTHESIS_ID}:
        errs.append("h38_hypothesis_collapsed_into_h20_h21_h22")

    if not _finite_scalar({k_sw: 0.05}.get(k_sw)):
        errs.append("sweep_follow_placebo_p_finite_probe_broken")
    if _finite_scalar({k_gap: 1.0}.get(k_sw)):
        errs.append("gap_finite_rate_incorrectly_counts_as_finite_sweep_follow_placebo_p")

    elig = {"book_hypothesis_eligible": True}
    if northset_has_finite_imbalance_p_ic({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h22_gate")
    if northset_has_finite_ohlc_identity_rate({k_gap: 1.0}):
        errs.append("gap_finite_rate_incorrectly_triggers_h20_gate")
    if northset_has_finite_book_uncrossed_rate({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h21_gate")
    return errs


def sweep_follow_fold_positive_fraction_vs_gap_finite_never_equate_honesty_errors(
    blob: object,
) -> list[str]:
    """Never equate H42 ``sweep_follow_fold_positive_fraction`` with ``gap_finite_rate``.

    DATA_CONTRACTS: sweep follow fold-positive fraction ≠ overnight gap finiteness.
    When both keys are present:

    - Distinct key identity
    - H42 binds ``sweep_follow_fold_positive_fraction`` only (gap has **no** H42 claim)
    - No dedicated fold-fraction finite-gate helper — ``_finite_scalar`` on the
      fold key only (a gap-only blob must not count as finite fold fraction)
    - Gap must not incorrectly trigger H20 / H21 / H22 gates
    - Numeric equality on clean synth is allowed (different units; rare)

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off CoS H44; off Lt H41; off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_sw = "sweep_follow_fold_positive_fraction"
    k_gap = "gap_finite_rate"
    if k_sw not in blob or k_gap not in blob:
        return []

    errs: list[str] = []
    if k_sw == k_gap:
        errs.append("sweep_follow_fold_positive_fraction_gap_finite_keys_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_sw) != H42_HYPOTHESIS_ID:
        errs.append("sweep_follow_fold_positive_fraction_not_bound_h42")
    if by_key.get(k_gap) == H42_HYPOTHESIS_ID:
        errs.append("gap_finite_rate_incorrectly_bound_h42")
    if H42_HYPOTHESIS_ID in {H20_HYPOTHESIS_ID, H21_HYPOTHESIS_ID, H22_HYPOTHESIS_ID}:
        errs.append("h42_hypothesis_collapsed_into_h20_h21_h22")

    if not _finite_scalar({k_sw: 0.05}.get(k_sw)):
        errs.append("sweep_follow_fold_positive_fraction_finite_probe_broken")
    if _finite_scalar({k_gap: 1.0}.get(k_sw)):
        errs.append(
            "gap_finite_rate_incorrectly_counts_as_finite_sweep_follow_fold_positive_fraction"
        )

    elig = {"book_hypothesis_eligible": True}
    if northset_has_finite_imbalance_p_ic({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h22_gate")
    if northset_has_finite_ohlc_identity_rate({k_gap: 1.0}):
        errs.append("gap_finite_rate_incorrectly_triggers_h20_gate")
    if northset_has_finite_book_uncrossed_rate({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h21_gate")
    return errs


def wick_skew_p_ic_vs_gap_finite_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate H26 ``wick_skew_p_ic`` with ``gap_finite_rate``.

    DATA_CONTRACTS: wick-skew discovery IC p ≠ overnight gap finiteness.
    When both keys are present:

    - Distinct key identity
    - H26 binds ``wick_skew_p_ic`` only (gap has **no** H26 claim)
    - Finite probe on ``wick_skew_p_ic`` only (gap-only blob must not count)
    - Gap must not incorrectly trigger H20 / H21 / H22 gates
    - Numeric equality on clean synth is allowed (different units; rare)

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_wick = "wick_skew_p_ic"
    k_gap = "gap_finite_rate"
    if k_wick not in blob or k_gap not in blob:
        return []

    errs: list[str] = []
    if k_wick == k_gap:
        errs.append("wick_skew_p_ic_gap_finite_keys_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_wick) != H26_HYPOTHESIS_ID:
        errs.append("wick_skew_p_ic_not_bound_h26")
    if by_key.get(k_gap) == H26_HYPOTHESIS_ID:
        errs.append("gap_finite_rate_incorrectly_bound_h26")
    if H26_HYPOTHESIS_ID in {H20_HYPOTHESIS_ID, H21_HYPOTHESIS_ID, H22_HYPOTHESIS_ID}:
        errs.append("h26_hypothesis_collapsed_into_h20_h21_h22")

    if not _finite_scalar({k_wick: 0.05}.get(k_wick)):
        errs.append("wick_skew_p_ic_finite_probe_broken")
    if _finite_scalar({k_gap: 1.0}.get(k_wick)):
        errs.append("gap_finite_rate_incorrectly_counts_as_finite_wick_skew")

    elig = {"book_hypothesis_eligible": True}
    if northset_has_finite_imbalance_p_ic({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h22_gate")
    if northset_has_finite_ohlc_identity_rate({k_gap: 1.0}):
        errs.append("gap_finite_rate_incorrectly_triggers_h20_gate")
    if northset_has_finite_book_uncrossed_rate({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h21_gate")
    return errs


def vpin_p_ic_vs_gap_finite_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate H32 ``vpin_p_ic`` with ``gap_finite_rate``.

    DATA_CONTRACTS: VPIN discovery IC p ≠ overnight gap finiteness.
    When both keys are present:

    - Distinct key identity
    - H32 binds ``vpin_p_ic`` only (gap has **no** H32 claim)
    - Finite probe on ``vpin_p_ic`` only (gap-only blob must not count)
    - Gap must not incorrectly trigger H20 / H21 / H22 gates
    - Numeric equality on clean synth is allowed (different units; rare)

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_vpin = "vpin_p_ic"
    k_gap = "gap_finite_rate"
    if k_vpin not in blob or k_gap not in blob:
        return []

    errs: list[str] = []
    if k_vpin == k_gap:
        errs.append("vpin_p_ic_gap_finite_keys_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_vpin) != H32_HYPOTHESIS_ID:
        errs.append("vpin_p_ic_not_bound_h32")
    if by_key.get(k_gap) == H32_HYPOTHESIS_ID:
        errs.append("gap_finite_rate_incorrectly_bound_h32")
    if H32_HYPOTHESIS_ID in {H20_HYPOTHESIS_ID, H21_HYPOTHESIS_ID, H22_HYPOTHESIS_ID}:
        errs.append("h32_hypothesis_collapsed_into_h20_h21_h22")

    if not _finite_scalar({k_vpin: 0.05}.get(k_vpin)):
        errs.append("vpin_p_ic_finite_probe_broken")
    if _finite_scalar({k_gap: 1.0}.get(k_vpin)):
        errs.append("gap_finite_rate_incorrectly_counts_as_finite_vpin")

    elig = {"book_hypothesis_eligible": True}
    if northset_has_finite_imbalance_p_ic({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h22_gate")
    if northset_has_finite_ohlc_identity_rate({k_gap: 1.0}):
        errs.append("gap_finite_rate_incorrectly_triggers_h20_gate")
    if northset_has_finite_book_uncrossed_rate({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h21_gate")
    return errs


def sweep_reject_signed_p_ic_vs_gap_finite_never_equate_honesty_errors(
    blob: object,
) -> list[str]:
    """Never equate H33 ``sweep_reject_signed_p_ic`` with ``gap_finite_rate``.

    DATA_CONTRACTS: sweep-reject discovery IC p ≠ overnight gap finiteness.
    When both keys are present:

    - Distinct key identity
    - H33 binds ``sweep_reject_signed_p_ic`` only (gap has **no** H33 claim)
    - Finite probe on sweep-reject key only (gap-only blob must not count)
    - Gap must not incorrectly trigger H20 / H21 / H22 gates
    - Numeric equality on clean synth is allowed (different units; rare)

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_sw = "sweep_reject_signed_p_ic"
    k_gap = "gap_finite_rate"
    if k_sw not in blob or k_gap not in blob:
        return []

    errs: list[str] = []
    if k_sw == k_gap:
        errs.append("sweep_reject_signed_p_ic_gap_finite_keys_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_sw) != H33_HYPOTHESIS_ID:
        errs.append("sweep_reject_signed_p_ic_not_bound_h33")
    if by_key.get(k_gap) == H33_HYPOTHESIS_ID:
        errs.append("gap_finite_rate_incorrectly_bound_h33")
    if H33_HYPOTHESIS_ID in {H20_HYPOTHESIS_ID, H21_HYPOTHESIS_ID, H22_HYPOTHESIS_ID}:
        errs.append("h33_hypothesis_collapsed_into_h20_h21_h22")

    if not _finite_scalar({k_sw: 0.05}.get(k_sw)):
        errs.append("sweep_reject_signed_p_ic_finite_probe_broken")
    if _finite_scalar({k_gap: 1.0}.get(k_sw)):
        errs.append("gap_finite_rate_incorrectly_counts_as_finite_sweep_reject")

    elig = {"book_hypothesis_eligible": True}
    if northset_has_finite_imbalance_p_ic({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h22_gate")
    if northset_has_finite_ohlc_identity_rate({k_gap: 1.0}):
        errs.append("gap_finite_rate_incorrectly_triggers_h20_gate")
    if northset_has_finite_book_uncrossed_rate({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h21_gate")
    return errs


def sweep_follow_signed_p_ic_vs_gap_finite_never_equate_honesty_errors(
    blob: object,
) -> list[str]:
    """Never equate H34 ``sweep_follow_signed_p_ic`` with ``gap_finite_rate``.

    DATA_CONTRACTS: sweep-follow discovery IC p ≠ overnight gap finiteness.
    When both keys are present:

    - Distinct key identity
    - H34 binds ``sweep_follow_signed_p_ic`` only (gap has **no** H34 claim)
    - Finite probe on sweep-follow key only (gap-only blob must not count)
    - Gap must not incorrectly trigger H20 / H21 / H22 gates
    - Numeric equality on clean synth is allowed (different units; rare)

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_sw = "sweep_follow_signed_p_ic"
    k_gap = "gap_finite_rate"
    if k_sw not in blob or k_gap not in blob:
        return []

    errs: list[str] = []
    if k_sw == k_gap:
        errs.append("sweep_follow_signed_p_ic_gap_finite_keys_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_sw) != H34_HYPOTHESIS_ID:
        errs.append("sweep_follow_signed_p_ic_not_bound_h34")
    if by_key.get(k_gap) == H34_HYPOTHESIS_ID:
        errs.append("gap_finite_rate_incorrectly_bound_h34")
    if H34_HYPOTHESIS_ID in {H20_HYPOTHESIS_ID, H21_HYPOTHESIS_ID, H22_HYPOTHESIS_ID}:
        errs.append("h34_hypothesis_collapsed_into_h20_h21_h22")

    if not _finite_scalar({k_sw: 0.05}.get(k_sw)):
        errs.append("sweep_follow_signed_p_ic_finite_probe_broken")
    if _finite_scalar({k_gap: 1.0}.get(k_sw)):
        errs.append("gap_finite_rate_incorrectly_counts_as_finite_sweep_follow")

    elig = {"book_hypothesis_eligible": True}
    if northset_has_finite_imbalance_p_ic({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h22_gate")
    if northset_has_finite_ohlc_identity_rate({k_gap: 1.0}):
        errs.append("gap_finite_rate_incorrectly_triggers_h20_gate")
    if northset_has_finite_book_uncrossed_rate({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h21_gate")
    return errs


def sweep_reject_control_diff_p_vs_sweep_follow_control_diff_p_never_equate_honesty_errors(
    blob: object,
) -> list[str]:
    """Never equate H44 ``sweep_reject_control_diff_p`` with H45 follow twin.

    Sibling sweep control-diff p-values — reject path ≠ follow path.
    When both keys are present:

    - Distinct key identity
    - Distinct hyp ids (H44 ≠ H45)
    - Catalog bind lock: reject → H44; follow → H45
    - Numeric equality on clean synth is allowed

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_rej = "sweep_reject_control_diff_p"
    k_fol = "sweep_follow_control_diff_p"
    if k_rej not in blob or k_fol not in blob:
        return []

    errs: list[str] = []
    if k_rej == k_fol:
        errs.append("sweep_reject_follow_control_diff_p_keys_collapsed")
    if H44_HYPOTHESIS_ID == H45_HYPOTHESIS_ID:
        errs.append("h44_h45_hypothesis_ids_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_rej) != H44_HYPOTHESIS_ID:
        errs.append("sweep_reject_control_diff_p_not_bound_h44")
    if by_key.get(k_fol) != H45_HYPOTHESIS_ID:
        errs.append("sweep_follow_control_diff_p_not_bound_h45")
    if by_key.get(k_rej) == H45_HYPOTHESIS_ID:
        errs.append("sweep_reject_control_diff_p_incorrectly_bound_h45")
    if by_key.get(k_fol) == H44_HYPOTHESIS_ID:
        errs.append("sweep_follow_control_diff_p_incorrectly_bound_h44")
    return errs


def sweep_reject_fold_positive_fraction_vs_sweep_follow_fold_positive_fraction_never_equate_honesty_errors(
    blob: object,
) -> list[str]:
    """Never equate H41 ``sweep_reject_fold_positive_fraction`` with H42 follow twin.

    Sibling sweep fold-positive fractions — reject path ≠ follow path.
    When both keys are present:

    - Distinct key identity
    - Distinct hyp ids (H41 ≠ H42)
    - Catalog bind lock: reject → H41; follow → H42
    - Numeric equality on clean synth is allowed

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_rej = "sweep_reject_fold_positive_fraction"
    k_fol = "sweep_follow_fold_positive_fraction"
    if k_rej not in blob or k_fol not in blob:
        return []

    errs: list[str] = []
    if k_rej == k_fol:
        errs.append("sweep_reject_follow_fold_positive_fraction_keys_collapsed")
    if H41_HYPOTHESIS_ID == H42_HYPOTHESIS_ID:
        errs.append("h41_h42_hypothesis_ids_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_rej) != H41_HYPOTHESIS_ID:
        errs.append("sweep_reject_fold_positive_fraction_not_bound_h41")
    if by_key.get(k_fol) != H42_HYPOTHESIS_ID:
        errs.append("sweep_follow_fold_positive_fraction_not_bound_h42")
    if by_key.get(k_rej) == H42_HYPOTHESIS_ID:
        errs.append("sweep_reject_fold_positive_fraction_incorrectly_bound_h42")
    if by_key.get(k_fol) == H41_HYPOTHESIS_ID:
        errs.append("sweep_follow_fold_positive_fraction_incorrectly_bound_h41")
    return errs


def sweep_reject_event_p_vs_sweep_follow_event_p_never_equate_honesty_errors(
    blob: object,
) -> list[str]:
    """Never equate H35 ``sweep_reject_event_p`` with H36 ``sweep_follow_event_p``.

    DATA_CONTRACTS: sweep-reject event p ≠ sweep-follow event p. When both keys
    are present:

    - Distinct key identity
    - Distinct hyp ids (H35 ≠ H36)
    - H35 binds reject event only; H36 binds follow event only
    - Numeric equality on clean synth is allowed

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_rej = "sweep_reject_event_p"
    k_fol = "sweep_follow_event_p"
    if k_rej not in blob or k_fol not in blob:
        return []

    errs: list[str] = []
    if k_rej == k_fol:
        errs.append("sweep_reject_follow_event_p_keys_collapsed")
    if H35_HYPOTHESIS_ID == H36_HYPOTHESIS_ID:
        errs.append("h35_h36_hypothesis_ids_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_rej) != H35_HYPOTHESIS_ID:
        errs.append("sweep_reject_event_p_not_bound_h35")
    if by_key.get(k_fol) != H36_HYPOTHESIS_ID:
        errs.append("sweep_follow_event_p_not_bound_h36")
    if by_key.get(k_rej) == H36_HYPOTHESIS_ID:
        errs.append("sweep_reject_event_p_incorrectly_bound_h36")
    if by_key.get(k_fol) == H35_HYPOTHESIS_ID:
        errs.append("sweep_follow_event_p_incorrectly_bound_h35")
    return errs


def sweep_reject_signed_p_ic_vs_sweep_follow_signed_p_ic_never_equate_honesty_errors(
    blob: object,
) -> list[str]:
    """Never equate H33 ``sweep_reject_signed_p_ic`` with H34 ``sweep_follow_signed_p_ic``.

    Sibling sweep signed-IC p-values — reject path ≠ follow path.
    When both keys are present:

    - Distinct key identity
    - Distinct hyp ids (H33 ≠ H34)
    - Catalog bind lock: reject → H33; follow → H34
    - Key-scoped ``_finite_scalar`` probes (no dedicated gate helpers)
    - Numeric equality on clean synth is allowed (different event sets)

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off CoS H44; off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_rej = "sweep_reject_signed_p_ic"
    k_fol = "sweep_follow_signed_p_ic"
    if k_rej not in blob or k_fol not in blob:
        return []

    errs: list[str] = []
    if k_rej == k_fol:
        errs.append("sweep_reject_follow_signed_p_ic_keys_collapsed")
    if H33_HYPOTHESIS_ID == H34_HYPOTHESIS_ID:
        errs.append("h33_h34_hypothesis_ids_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_rej) != H33_HYPOTHESIS_ID:
        errs.append("sweep_reject_signed_p_ic_not_bound_h33")
    if by_key.get(k_fol) != H34_HYPOTHESIS_ID:
        errs.append("sweep_follow_signed_p_ic_not_bound_h34")
    if by_key.get(k_rej) == H34_HYPOTHESIS_ID:
        errs.append("sweep_reject_signed_p_ic_incorrectly_bound_h34")
    if by_key.get(k_fol) == H33_HYPOTHESIS_ID:
        errs.append("sweep_follow_signed_p_ic_incorrectly_bound_h33")

    if not _finite_scalar({k_rej: 0.05}.get(k_rej)):
        errs.append("sweep_reject_signed_p_ic_finite_probe_broken")
    if _finite_scalar({k_fol: 0.05}.get(k_rej)):
        errs.append("sweep_follow_signed_p_ic_incorrectly_counts_as_finite_reject")
    if not _finite_scalar({k_fol: 0.05}.get(k_fol)):
        errs.append("sweep_follow_signed_p_ic_finite_probe_broken")
    if _finite_scalar({k_rej: 0.05}.get(k_fol)):
        errs.append("sweep_reject_signed_p_ic_incorrectly_counts_as_finite_follow")
    return errs


def sweep_reject_cost_adjusted_mean_bps_vs_sweep_follow_cost_adjusted_mean_bps_never_equate_honesty_errors(
    blob: object,
) -> list[str]:
    """Never equate H39 ``sweep_reject_cost_adjusted_mean_bps`` with H40 ``sweep_follow_cost_adjusted_mean_bps``.

    Sibling sweep cost-adjusted mean bps — reject path ≠ follow path.
    Mirrors :func:`sweep_reject_signed_p_ic_vs_sweep_follow_signed_p_ic_never_equate_honesty_errors`
    (H33≠H34). When both keys are present:

    - Distinct key identity
    - Distinct hyp ids (H39 ≠ H40)
    - Catalog bind lock: reject → H39; follow → H40
    - Key-scoped ``_finite_scalar`` probes (no dedicated gate helpers)
    - Numeric equality on clean synth is allowed (different event sets)

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off Sergeant H37≠H38; off Lt H35≠H36; off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_rej = "sweep_reject_cost_adjusted_mean_bps"
    k_fol = "sweep_follow_cost_adjusted_mean_bps"
    if k_rej not in blob or k_fol not in blob:
        return []

    errs: list[str] = []
    if k_rej == k_fol:
        errs.append("sweep_reject_follow_cost_adjusted_mean_bps_keys_collapsed")
    if H39_HYPOTHESIS_ID == H40_HYPOTHESIS_ID:
        errs.append("h39_h40_hypothesis_ids_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_rej) != H39_HYPOTHESIS_ID:
        errs.append("sweep_reject_cost_adjusted_mean_bps_not_bound_h39")
    if by_key.get(k_fol) != H40_HYPOTHESIS_ID:
        errs.append("sweep_follow_cost_adjusted_mean_bps_not_bound_h40")
    if by_key.get(k_rej) == H40_HYPOTHESIS_ID:
        errs.append("sweep_reject_cost_adjusted_mean_bps_incorrectly_bound_h40")
    if by_key.get(k_fol) == H39_HYPOTHESIS_ID:
        errs.append("sweep_follow_cost_adjusted_mean_bps_incorrectly_bound_h39")

    if not _finite_scalar({k_rej: 1.0}.get(k_rej)):
        errs.append("sweep_reject_cost_adjusted_mean_bps_finite_probe_broken")
    if _finite_scalar({k_fol: 1.0}.get(k_rej)):
        errs.append("sweep_follow_cost_adjusted_mean_bps_incorrectly_counts_as_finite_reject")
    if not _finite_scalar({k_fol: 1.0}.get(k_fol)):
        errs.append("sweep_follow_cost_adjusted_mean_bps_finite_probe_broken")
    if _finite_scalar({k_rej: 1.0}.get(k_fol)):
        errs.append("sweep_reject_cost_adjusted_mean_bps_incorrectly_counts_as_finite_follow")
    return errs


def sweep_reject_placebo_p_vs_sweep_follow_placebo_p_never_equate_honesty_errors(
    blob: object,
) -> list[str]:
    """Never equate H37 ``sweep_reject_placebo_p`` with H38 ``sweep_follow_placebo_p``.

    Sibling sweep placebo p-values — reject placebo ≠ follow placebo.
    When both keys are present:

    - Distinct key identity
    - Distinct hyp ids (H37 ≠ H38)
    - Catalog bind lock: reject placebo → H37; follow placebo → H38
    - Key-scoped ``_finite_scalar`` probes (no dedicated gate helpers)
    - Numeric equality on clean synth is allowed (different event sets)

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off CoS H39≠H40; off Lt H35≠H36; off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_rej = "sweep_reject_placebo_p"
    k_fol = "sweep_follow_placebo_p"
    if k_rej not in blob or k_fol not in blob:
        return []

    errs: list[str] = []
    if k_rej == k_fol:
        errs.append("sweep_reject_follow_placebo_p_keys_collapsed")
    if H37_HYPOTHESIS_ID == H38_HYPOTHESIS_ID:
        errs.append("h37_h38_hypothesis_ids_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_rej) != H37_HYPOTHESIS_ID:
        errs.append("sweep_reject_placebo_p_not_bound_h37")
    if by_key.get(k_fol) != H38_HYPOTHESIS_ID:
        errs.append("sweep_follow_placebo_p_not_bound_h38")
    if by_key.get(k_rej) == H38_HYPOTHESIS_ID:
        errs.append("sweep_reject_placebo_p_incorrectly_bound_h38")
    if by_key.get(k_fol) == H37_HYPOTHESIS_ID:
        errs.append("sweep_follow_placebo_p_incorrectly_bound_h37")

    if not _finite_scalar({k_rej: 0.05}.get(k_rej)):
        errs.append("sweep_reject_placebo_p_finite_probe_broken")
    if _finite_scalar({k_fol: 0.05}.get(k_rej)):
        errs.append("sweep_follow_placebo_p_incorrectly_counts_as_finite_reject_placebo")
    if not _finite_scalar({k_fol: 0.05}.get(k_fol)):
        errs.append("sweep_follow_placebo_p_finite_probe_broken")
    if _finite_scalar({k_rej: 0.05}.get(k_fol)):
        errs.append("sweep_reject_placebo_p_incorrectly_counts_as_finite_follow_placebo")
    return errs


def sweep_reject_placebo_p_vs_gap_finite_never_equate_honesty_errors(
    blob: object,
) -> list[str]:
    """Never equate H37 ``sweep_reject_placebo_p`` with ``gap_finite_rate``.

    DATA_CONTRACTS: sweep-reject placebo p ≠ overnight gap finiteness.
    When both keys are present:

    - Distinct key identity
    - H37 binds ``sweep_reject_placebo_p`` only (gap has **no** H37 claim)
    - Finite probe on placebo key only (gap-only blob must not count)
    - Gap must not incorrectly trigger H20 / H21 / H22 gates
    - Numeric equality on clean synth is allowed (different units; rare)

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_sw = "sweep_reject_placebo_p"
    k_gap = "gap_finite_rate"
    if k_sw not in blob or k_gap not in blob:
        return []

    errs: list[str] = []
    if k_sw == k_gap:
        errs.append("sweep_reject_placebo_p_gap_finite_keys_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_sw) != H37_HYPOTHESIS_ID:
        errs.append("sweep_reject_placebo_p_not_bound_h37")
    if by_key.get(k_gap) == H37_HYPOTHESIS_ID:
        errs.append("gap_finite_rate_incorrectly_bound_h37")
    if H37_HYPOTHESIS_ID in {H20_HYPOTHESIS_ID, H21_HYPOTHESIS_ID, H22_HYPOTHESIS_ID}:
        errs.append("h37_hypothesis_collapsed_into_h20_h21_h22")

    if not _finite_scalar({k_sw: 0.05}.get(k_sw)):
        errs.append("sweep_reject_placebo_p_finite_probe_broken")
    if _finite_scalar({k_gap: 1.0}.get(k_sw)):
        errs.append("gap_finite_rate_incorrectly_counts_as_finite_sweep_reject_placebo")

    elig = {"book_hypothesis_eligible": True}
    if northset_has_finite_imbalance_p_ic({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h22_gate")
    if northset_has_finite_ohlc_identity_rate({k_gap: 1.0}):
        errs.append("gap_finite_rate_incorrectly_triggers_h20_gate")
    if northset_has_finite_book_uncrossed_rate({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h21_gate")
    return errs


def sweep_follow_cost_adjusted_mean_bps_vs_gap_finite_never_equate_honesty_errors(
    blob: object,
) -> list[str]:
    """Never equate H40 ``sweep_follow_cost_adjusted_mean_bps`` with ``gap_finite_rate``.

    DATA_CONTRACTS: sweep-follow cost-adjusted mean bps ≠ overnight gap finiteness.
    When both keys are present:

    - Distinct key identity
    - H40 binds cost-adjusted follow mean only (gap has **no** H40 claim)
    - Finite probe on cost key only (gap-only blob must not count)
    - Gap must not incorrectly trigger H20 / H21 / H22 gates
    - Numeric equality on clean synth is allowed (different units; rare)

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_sw = "sweep_follow_cost_adjusted_mean_bps"
    k_gap = "gap_finite_rate"
    if k_sw not in blob or k_gap not in blob:
        return []

    errs: list[str] = []
    if k_sw == k_gap:
        errs.append("sweep_follow_cost_adjusted_mean_bps_gap_finite_keys_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_sw) != H40_HYPOTHESIS_ID:
        errs.append("sweep_follow_cost_adjusted_mean_bps_not_bound_h40")
    if by_key.get(k_gap) == H40_HYPOTHESIS_ID:
        errs.append("gap_finite_rate_incorrectly_bound_h40")
    if H40_HYPOTHESIS_ID in {H20_HYPOTHESIS_ID, H21_HYPOTHESIS_ID, H22_HYPOTHESIS_ID}:
        errs.append("h40_hypothesis_collapsed_into_h20_h21_h22")

    if not _finite_scalar({k_sw: 0.05}.get(k_sw)):
        errs.append("sweep_follow_cost_adjusted_mean_bps_finite_probe_broken")
    if _finite_scalar({k_gap: 1.0}.get(k_sw)):
        errs.append("gap_finite_rate_incorrectly_counts_as_finite_sweep_follow_cost")

    elig = {"book_hypothesis_eligible": True}
    if northset_has_finite_imbalance_p_ic({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h22_gate")
    if northset_has_finite_ohlc_identity_rate({k_gap: 1.0}):
        errs.append("gap_finite_rate_incorrectly_triggers_h20_gate")
    if northset_has_finite_book_uncrossed_rate({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h21_gate")
    return errs


def sweep_reject_fold_positive_fraction_vs_gap_finite_never_equate_honesty_errors(
    blob: object,
) -> list[str]:
    """Never equate H41 ``sweep_reject_fold_positive_fraction`` with ``gap_finite_rate``.

    DATA_CONTRACTS: sweep-reject fold-positive fraction ≠ overnight gap finiteness.
    When both keys are present:

    - Distinct key identity
    - H41 binds fold-positive reject fraction only (gap has **no** H41 claim)
    - Finite probe on fold key only (gap-only blob must not count)
    - Gap must not incorrectly trigger H20 / H21 / H22 gates
    - Numeric equality on clean synth is allowed (different units; rare)

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_sw = "sweep_reject_fold_positive_fraction"
    k_gap = "gap_finite_rate"
    if k_sw not in blob or k_gap not in blob:
        return []

    errs: list[str] = []
    if k_sw == k_gap:
        errs.append("sweep_reject_fold_positive_fraction_gap_finite_keys_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_sw) != H41_HYPOTHESIS_ID:
        errs.append("sweep_reject_fold_positive_fraction_not_bound_h41")
    if by_key.get(k_gap) == H41_HYPOTHESIS_ID:
        errs.append("gap_finite_rate_incorrectly_bound_h41")
    if H41_HYPOTHESIS_ID in {H20_HYPOTHESIS_ID, H21_HYPOTHESIS_ID, H22_HYPOTHESIS_ID}:
        errs.append("h41_hypothesis_collapsed_into_h20_h21_h22")

    if not _finite_scalar({k_sw: 0.05}.get(k_sw)):
        errs.append("sweep_reject_fold_positive_fraction_finite_probe_broken")
    if _finite_scalar({k_gap: 1.0}.get(k_sw)):
        errs.append("gap_finite_rate_incorrectly_counts_as_finite_sweep_reject_fold")

    elig = {"book_hypothesis_eligible": True}
    if northset_has_finite_imbalance_p_ic({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h22_gate")
    if northset_has_finite_ohlc_identity_rate({k_gap: 1.0}):
        errs.append("gap_finite_rate_incorrectly_triggers_h20_gate")
    if northset_has_finite_book_uncrossed_rate({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h21_gate")
    return errs


def sweep_follow_control_diff_p_vs_gap_finite_never_equate_honesty_errors(
    blob: object,
) -> list[str]:
    """Never equate H45 ``sweep_follow_control_diff_p`` with ``gap_finite_rate``.

    DATA_CONTRACTS: sweep-follow control-diff p ≠ overnight gap finiteness.
    When both keys are present:

    - Distinct key identity
    - H45 binds follow control-diff p only (gap has **no** H45 claim)
    - Finite probe on control key only (gap-only blob must not count)
    - Gap must not incorrectly trigger H20 / H21 / H22 gates
    - Numeric equality on clean synth is allowed (different units; rare)

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_sw = "sweep_follow_control_diff_p"
    k_gap = "gap_finite_rate"
    if k_sw not in blob or k_gap not in blob:
        return []

    errs: list[str] = []
    if k_sw == k_gap:
        errs.append("sweep_follow_control_diff_p_gap_finite_keys_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_sw) != H45_HYPOTHESIS_ID:
        errs.append("sweep_follow_control_diff_p_not_bound_h45")
    if by_key.get(k_gap) == H45_HYPOTHESIS_ID:
        errs.append("gap_finite_rate_incorrectly_bound_h45")
    if H45_HYPOTHESIS_ID in {H20_HYPOTHESIS_ID, H21_HYPOTHESIS_ID, H22_HYPOTHESIS_ID}:
        errs.append("h45_hypothesis_collapsed_into_h20_h21_h22")

    if not _finite_scalar({k_sw: 0.05}.get(k_sw)):
        errs.append("sweep_follow_control_diff_p_finite_probe_broken")
    if _finite_scalar({k_gap: 1.0}.get(k_sw)):
        errs.append("gap_finite_rate_incorrectly_counts_as_finite_sweep_follow_control")

    elig = {"book_hypothesis_eligible": True}
    if northset_has_finite_imbalance_p_ic({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h22_gate")
    if northset_has_finite_ohlc_identity_rate({k_gap: 1.0}):
        errs.append("gap_finite_rate_incorrectly_triggers_h20_gate")
    if northset_has_finite_book_uncrossed_rate({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h21_gate")
    return errs


def session_book_vpin_p_ic_vs_gap_finite_never_equate_honesty_errors(
    blob: object,
) -> list[str]:
    """Never equate H43 ``session_book_vpin_p_ic`` with ``gap_finite_rate``.

    DATA_CONTRACTS: session-book VPIN discovery IC p ≠ overnight gap finiteness.
    When both keys are present:

    - Distinct key identity
    - H43 gate helper binds session_book_vpin_p_ic only (gap must not trigger it)
    - Gap must not incorrectly trigger H20 / H21 / H22 gates
    - Numeric equality on clean synth is allowed (different units; rare)

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_vpin = "session_book_vpin_p_ic"
    k_gap = "gap_finite_rate"
    if k_vpin not in blob or k_gap not in blob:
        return []

    errs: list[str] = []
    if k_vpin == k_gap:
        errs.append("session_book_vpin_p_ic_gap_finite_keys_collapsed")

    if H43_HYPOTHESIS_ID in {H20_HYPOTHESIS_ID, H21_HYPOTHESIS_ID, H22_HYPOTHESIS_ID}:
        errs.append("h43_hypothesis_collapsed_into_h20_h21_h22")

    elig = {"session_book_hypothesis_eligible": True, "book_hypothesis_eligible": True}
    if not northset_has_finite_session_book_vpin_p_ic({k_vpin: 0.05, **elig}):
        errs.append("session_book_vpin_p_ic_h43_gate_helper_broken")
    if northset_has_finite_session_book_vpin_p_ic({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h43_gate")

    if northset_has_finite_imbalance_p_ic({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h22_gate")
    if northset_has_finite_ohlc_identity_rate({k_gap: 1.0}):
        errs.append("gap_finite_rate_incorrectly_triggers_h20_gate")
    if northset_has_finite_book_uncrossed_rate({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h21_gate")
    return errs


def gap_finite_rate_vs_session_reconstructs_never_equate_honesty_errors(
    blob: object,
) -> list[str]:
    """Never equate ``gap_finite_rate`` with H23 ``session_reconstructs_daily_rate``.

    DATA_CONTRACTS: overnight gap finiteness ≠ session→daily reconstruction.
    When both keys are present:

    - Distinct key identity
    - Hyp/gate: reconstructs binds ``H23_HYPOTHESIS_ID``; gap is **not** H23
      and must not trigger H21/H20 gates
    - Numeric equality on clean synth is allowed

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off H20/H21/H22 triad. Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_gap = "gap_finite_rate"
    k_recon = "session_reconstructs_daily_rate"
    if k_gap not in blob or k_recon not in blob:
        return []

    errs: list[str] = []
    if k_gap == k_recon:
        errs.append("gap_finite_session_reconstructs_keys_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_recon) != H23_HYPOTHESIS_ID:
        errs.append("session_reconstructs_daily_rate_not_bound_h23")
    if by_key.get(k_gap) == H23_HYPOTHESIS_ID:
        errs.append("gap_finite_rate_incorrectly_bound_h23")

    elig = {"book_hypothesis_eligible": True}
    if northset_has_finite_book_uncrossed_rate({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h21_gate")
    if northset_has_finite_book_uncrossed_rate({k_recon: 1.0, **elig}):
        errs.append("session_reconstructs_incorrectly_triggers_h21_gate")

    if northset_has_finite_ohlc_identity_rate({k_gap: 1.0}):
        errs.append("gap_finite_rate_incorrectly_triggers_h20_gate")
    if northset_has_finite_ohlc_identity_rate({k_recon: 1.0}):
        errs.append("session_reconstructs_incorrectly_triggers_h20_gate")

    if H23_HYPOTHESIS_ID in {H20_HYPOTHESIS_ID, H21_HYPOTHESIS_ID}:
        errs.append("h23_hypothesis_collapsed_into_h20_or_h21")
    return errs


def gap_finite_rate_vs_session_volume_conservation_never_equate_honesty_errors(
    blob: object,
) -> list[str]:
    """Never equate ``gap_finite_rate`` with H24 ``session_volume_conservation_rate``.

    DATA_CONTRACTS: overnight gap finiteness ≠ session volume conservation.
    When both keys are present:

    - Distinct key identity
    - Hyp/gate: conservation binds ``H24_HYPOTHESIS_ID``; gap is **not** H24
      and must not trigger H21/H20 gates
    - Numeric equality on clean synth is allowed

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off H20/H21/H22 triad. Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_gap = "gap_finite_rate"
    k_vol = "session_volume_conservation_rate"
    if k_gap not in blob or k_vol not in blob:
        return []

    errs: list[str] = []
    if k_gap == k_vol:
        errs.append("gap_finite_session_volume_conservation_keys_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_vol) != H24_HYPOTHESIS_ID:
        errs.append("session_volume_conservation_rate_not_bound_h24")
    if by_key.get(k_gap) == H24_HYPOTHESIS_ID:
        errs.append("gap_finite_rate_incorrectly_bound_h24")

    elig = {"book_hypothesis_eligible": True}
    if northset_has_finite_book_uncrossed_rate({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h21_gate")
    if northset_has_finite_book_uncrossed_rate({k_vol: 1.0, **elig}):
        errs.append("session_volume_conservation_incorrectly_triggers_h21_gate")

    if northset_has_finite_ohlc_identity_rate({k_gap: 1.0}):
        errs.append("gap_finite_rate_incorrectly_triggers_h20_gate")
    if northset_has_finite_ohlc_identity_rate({k_vol: 1.0}):
        errs.append("session_volume_conservation_incorrectly_triggers_h20_gate")

    if H24_HYPOTHESIS_ID in {H20_HYPOTHESIS_ID, H21_HYPOTHESIS_ID}:
        errs.append("h24_hypothesis_collapsed_into_h20_or_h21")
    return errs


def gap_finite_rate_vs_session_chain_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate ``gap_finite_rate`` with H29 ``session_chain_rate``.

    DATA_CONTRACTS: overnight gap finiteness ≠ session chain identity. When both
    keys are present:

    - Distinct key identity
    - Hyp/gate: chain binds ``H29_HYPOTHESIS_ID``; gap is **not** H29 and must
      not trigger H21/H20 gates
    - Numeric equality on clean synth is allowed

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off H20/H21/H22 triad. Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_gap = "gap_finite_rate"
    k_chain = "session_chain_rate"
    if k_gap not in blob or k_chain not in blob:
        return []

    errs: list[str] = []
    if k_gap == k_chain:
        errs.append("gap_finite_session_chain_keys_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_chain) != H29_HYPOTHESIS_ID:
        errs.append("session_chain_rate_not_bound_h29")
    if by_key.get(k_gap) == H29_HYPOTHESIS_ID:
        errs.append("gap_finite_rate_incorrectly_bound_h29")

    elig = {"book_hypothesis_eligible": True}
    if northset_has_finite_book_uncrossed_rate({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h21_gate")
    if northset_has_finite_book_uncrossed_rate({k_chain: 1.0, **elig}):
        errs.append("session_chain_incorrectly_triggers_h21_gate")

    if northset_has_finite_ohlc_identity_rate({k_gap: 1.0}):
        errs.append("gap_finite_rate_incorrectly_triggers_h20_gate")
    if northset_has_finite_ohlc_identity_rate({k_chain: 1.0}):
        errs.append("session_chain_incorrectly_triggers_h20_gate")

    if H29_HYPOTHESIS_ID in {H20_HYPOTHESIS_ID, H21_HYPOTHESIS_ID}:
        errs.append("h29_hypothesis_collapsed_into_h20_or_h21")
    return errs


def session_ohlc_vs_book_uncrossed_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate ``session_ohlc_identity_rate`` with H21 ``book_uncrossed_rate``.

    DATA_CONTRACTS: session-candle OHLC identity ≠ uncrossed book. When both
    keys are present:

    - Distinct key identity
    - Hyp/gate: session OHLC is **not** H23; H21 binds ``book_uncrossed_rate``
      only (session OHLC must not trigger H21); H20 gate is daily-only
    - Numeric equality on clean synth is allowed

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off H20/H21/H22 triad polish. Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_sess = "session_ohlc_identity_rate"
    k_book = "book_uncrossed_rate"
    if k_sess not in blob or k_book not in blob:
        return []

    errs: list[str] = []
    if k_sess == k_book:
        errs.append("session_ohlc_book_uncrossed_keys_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_sess) == H23_HYPOTHESIS_ID:
        errs.append("session_ohlc_identity_rate_incorrectly_bound_h23")
    if by_key.get(k_book) == H23_HYPOTHESIS_ID:
        errs.append("book_uncrossed_rate_incorrectly_bound_h23")

    elig = {"book_hypothesis_eligible": True}
    if not northset_has_finite_book_uncrossed_rate({k_book: 1.0, **elig}):
        errs.append("book_uncrossed_h21_gate_helper_broken")
    if northset_has_finite_book_uncrossed_rate({k_sess: 1.0, **elig}):
        errs.append("session_ohlc_incorrectly_triggers_h21_gate")

    if northset_has_finite_ohlc_identity_rate({k_sess: 1.0}):
        errs.append("session_ohlc_incorrectly_triggers_h20_gate")
    if northset_has_finite_ohlc_identity_rate({k_book: 1.0}):
        errs.append("book_uncrossed_incorrectly_triggers_h20_gate")

    if H21_HYPOTHESIS_ID in {H20_HYPOTHESIS_ID, H23_HYPOTHESIS_ID}:
        errs.append("h21_hypothesis_collapsed_into_h20_or_h23")
    return errs


def book_uncrossed_vs_session_chain_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate H21 ``book_uncrossed_rate`` with H29 ``session_chain_rate``.

    DATA_CONTRACTS: uncrossed book ≠ session reconstruction chain. When both
    keys are present:

    - Distinct key identity
    - Distinct hyp ids (H21 ≠ H29)
    - H21 finite gate on book_uncrossed only; H29 binds chain only
    - Numeric equality on clean synth is allowed

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite. Leaves reconstructs lane clear.
    """
    if not isinstance(blob, dict):
        return []
    k_book = "book_uncrossed_rate"
    k_chain = "session_chain_rate"
    if k_book not in blob or k_chain not in blob:
        return []

    errs: list[str] = []
    if k_book == k_chain:
        errs.append("book_uncrossed_session_chain_keys_collapsed")
    if H21_HYPOTHESIS_ID == H29_HYPOTHESIS_ID:
        errs.append("h21_h29_hypothesis_ids_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_chain) != H29_HYPOTHESIS_ID:
        errs.append("session_chain_rate_not_bound_h29")
    if by_key.get(k_book) == H29_HYPOTHESIS_ID:
        errs.append("book_uncrossed_rate_incorrectly_bound_h29")

    elig = {"book_hypothesis_eligible": True}
    if not northset_has_finite_book_uncrossed_rate({k_book: 1.0, **elig}):
        errs.append("book_uncrossed_h21_gate_helper_broken")
    if northset_has_finite_book_uncrossed_rate({k_chain: 1.0, **elig}):
        errs.append("session_chain_incorrectly_triggers_h21_gate")
    return errs


def imbalance_top_p_ic_vs_session_chain_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate H22 ``imbalance_top_p_ic`` with H29 ``session_chain_rate``.

    DATA_CONTRACTS: imbalance discovery IC p ≠ session reconstruction chain.
    When both keys are present:

    - Distinct key identity
    - Distinct hyp ids (H22 ≠ H29)
    - H22 finite gate on imbalance_top_p_ic only; H29 binds chain only
    - Numeric equality on clean synth is allowed (different units)

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_imb = "imbalance_top_p_ic"
    k_chain = "session_chain_rate"
    if k_imb not in blob or k_chain not in blob:
        return []

    errs: list[str] = []
    if k_imb == k_chain:
        errs.append("imbalance_top_p_ic_session_chain_keys_collapsed")
    if H22_HYPOTHESIS_ID == H29_HYPOTHESIS_ID:
        errs.append("h22_h29_hypothesis_ids_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_chain) != H29_HYPOTHESIS_ID:
        errs.append("session_chain_rate_not_bound_h29")
    if by_key.get(k_imb) == H29_HYPOTHESIS_ID:
        errs.append("imbalance_top_p_ic_incorrectly_bound_h29")

    elig = {"book_hypothesis_eligible": True}
    if not northset_has_finite_imbalance_p_ic({k_imb: 0.05, **elig}):
        errs.append("imbalance_p_ic_h22_gate_helper_broken")
    if northset_has_finite_imbalance_p_ic({k_chain: 1.0, **elig}):
        errs.append("session_chain_incorrectly_triggers_h22_gate")
    return errs


def imbalance_top_p_ic_vs_session_ohlc_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate H22 ``imbalance_top_p_ic`` with ``session_ohlc_identity_rate``.

    DATA_CONTRACTS: imbalance discovery IC p ≠ session-candle OHLC identity.
    When both keys are present:

    - Distinct key identity
    - H22 finite gate on imbalance_top_p_ic only (session OHLC must not trigger it)
    - session_ohlc must **not** bind H23 / H24 / H29
    - Numeric equality on clean synth is allowed (different units)

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_imb = "imbalance_top_p_ic"
    k_sess = "session_ohlc_identity_rate"
    if k_imb not in blob or k_sess not in blob:
        return []

    errs: list[str] = []
    if k_imb == k_sess:
        errs.append("imbalance_top_p_ic_session_ohlc_keys_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    sess_hyp = by_key.get(k_sess)
    if sess_hyp == H23_HYPOTHESIS_ID:
        errs.append("session_ohlc_identity_rate_incorrectly_bound_h23")
    if sess_hyp == H24_HYPOTHESIS_ID:
        errs.append("session_ohlc_identity_rate_incorrectly_bound_h24")
    if sess_hyp == H29_HYPOTHESIS_ID:
        errs.append("session_ohlc_identity_rate_incorrectly_bound_h29")
    if by_key.get(k_imb) in {H23_HYPOTHESIS_ID, H24_HYPOTHESIS_ID, H29_HYPOTHESIS_ID}:
        errs.append("imbalance_top_p_ic_incorrectly_bound_session_hyp")

    elig = {"book_hypothesis_eligible": True}
    if not northset_has_finite_imbalance_p_ic({k_imb: 0.05, **elig}):
        errs.append("imbalance_p_ic_h22_gate_helper_broken")
    if northset_has_finite_imbalance_p_ic({k_sess: 1.0, **elig}):
        errs.append("session_ohlc_incorrectly_triggers_h22_gate")
    return errs


def imbalance_top_p_ic_vs_session_volume_conservation_never_equate_honesty_errors(
    blob: object,
) -> list[str]:
    """Never equate H22 ``imbalance_top_p_ic`` with H24 ``session_volume_conservation_rate``.

    DATA_CONTRACTS: imbalance discovery IC p ≠ session volume conservation.
    Completes H22↔session-rate mesh with H22↔H20/H21/H29 and H24 siblings.
    When both keys are present:

    - Distinct key identity
    - Distinct hyp ids (H22 ≠ H24)
    - H22 finite gate on imbalance_top_p_ic only; H24 binds volume only
    - imbalance_top_p_ic must not bind H24; volume must not trigger H22 gate
    - Numeric equality on clean synth is allowed (different units)

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_imb = "imbalance_top_p_ic"
    k_vol = "session_volume_conservation_rate"
    if k_imb not in blob or k_vol not in blob:
        return []

    errs: list[str] = []
    if k_imb == k_vol:
        errs.append("imbalance_top_p_ic_session_volume_conservation_keys_collapsed")
    if H22_HYPOTHESIS_ID == H24_HYPOTHESIS_ID:
        errs.append("h22_h24_hypothesis_ids_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_vol) != H24_HYPOTHESIS_ID:
        errs.append("session_volume_conservation_rate_not_bound_h24")
    if by_key.get(k_imb) == H24_HYPOTHESIS_ID:
        errs.append("imbalance_top_p_ic_incorrectly_bound_h24")

    elig = {"book_hypothesis_eligible": True}
    if not northset_has_finite_imbalance_p_ic({k_imb: 0.05, **elig}):
        errs.append("imbalance_p_ic_h22_gate_helper_broken")
    if northset_has_finite_imbalance_p_ic({k_vol: 1.0, **elig}):
        errs.append("session_volume_conservation_incorrectly_triggers_h22_gate")
    return errs


def book_uncrossed_vs_session_reconstructs_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate H21 ``book_uncrossed_rate`` with H23 ``session_reconstructs_daily_rate``.

    DATA_CONTRACTS: uncrossed book ≠ session envelope reconstructing the daily bar.
    When both keys are present:

    - Distinct key identity
    - Distinct hyp ids (H21 ≠ H23)
    - H21 finite gate on book_uncrossed only; H23 binds reconstructs only
    - Numeric equality on clean synth is allowed

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_book = "book_uncrossed_rate"
    k_recon = "session_reconstructs_daily_rate"
    if k_book not in blob or k_recon not in blob:
        return []

    errs: list[str] = []
    if k_book == k_recon:
        errs.append("book_uncrossed_session_reconstructs_keys_collapsed")
    if H21_HYPOTHESIS_ID == H23_HYPOTHESIS_ID:
        errs.append("h21_h23_hypothesis_ids_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_recon) != H23_HYPOTHESIS_ID:
        errs.append("session_reconstructs_daily_rate_not_bound_h23")
    if by_key.get(k_book) == H23_HYPOTHESIS_ID:
        errs.append("book_uncrossed_rate_incorrectly_bound_h23")

    elig = {"book_hypothesis_eligible": True}
    if not northset_has_finite_book_uncrossed_rate({k_book: 1.0, **elig}):
        errs.append("book_uncrossed_h21_gate_helper_broken")
    if northset_has_finite_book_uncrossed_rate({k_recon: 1.0, **elig}):
        errs.append("session_reconstructs_incorrectly_triggers_h21_gate")
    return errs


def book_uncrossed_vs_volume_conservation_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate H21 ``book_uncrossed_rate`` with H24 ``session_volume_conservation_rate``.

    DATA_CONTRACTS: uncrossed book ≠ session volume conservation. When both
    keys are present:

    - Distinct key identity
    - Distinct hyp ids (H21 ≠ H24)
    - H21 finite gate on book_uncrossed only; H24 binds volume only
    - book_uncrossed must not bind H24; volume must not trigger H21 gate
    - Numeric equality on clean synth is allowed

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_book = "book_uncrossed_rate"
    k_vol = "session_volume_conservation_rate"
    if k_book not in blob or k_vol not in blob:
        return []

    errs: list[str] = []
    if k_book == k_vol:
        errs.append("book_uncrossed_volume_conservation_keys_collapsed")
    if H21_HYPOTHESIS_ID == H24_HYPOTHESIS_ID:
        errs.append("h21_h24_hypothesis_ids_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_vol) != H24_HYPOTHESIS_ID:
        errs.append("session_volume_conservation_rate_not_bound_h24")
    if by_key.get(k_book) == H24_HYPOTHESIS_ID:
        errs.append("book_uncrossed_rate_incorrectly_bound_h24")

    elig = {"book_hypothesis_eligible": True}
    if not northset_has_finite_book_uncrossed_rate({k_book: 1.0, **elig}):
        errs.append("book_uncrossed_h21_gate_helper_broken")
    if northset_has_finite_book_uncrossed_rate({k_vol: 1.0, **elig}):
        errs.append("session_volume_conservation_incorrectly_triggers_h21_gate")
    return errs


def session_ohlc_vs_gap_finite_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate ``session_ohlc_identity_rate`` with ``gap_finite_rate``.

    DATA_CONTRACTS: session-candle OHLC identity ≠ overnight gap finiteness.
    When both keys are present:

    - Distinct key identity
    - Hyp/gate: session OHLC is **not** H23; gap has **no** H21 claim;
      H20 gate binds daily ``ohlc_identity_rate`` only (neither key is H20)
    - Numeric equality on clean synth is allowed

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off H20/H21/H22 triad. Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_sess = "session_ohlc_identity_rate"
    k_gap = "gap_finite_rate"
    if k_sess not in blob or k_gap not in blob:
        return []

    errs: list[str] = []
    if k_sess == k_gap:
        errs.append("session_ohlc_gap_finite_keys_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_sess) == H23_HYPOTHESIS_ID:
        errs.append("session_ohlc_identity_rate_incorrectly_bound_h23")
    if by_key.get(k_gap) == H23_HYPOTHESIS_ID:
        errs.append("gap_finite_rate_incorrectly_bound_h23")

    # H21 is book_uncrossed — gap must not share that gate
    elig = {"book_hypothesis_eligible": True}
    if northset_has_finite_book_uncrossed_rate({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h21_gate")
    if northset_has_finite_book_uncrossed_rate({k_sess: 1.0, **elig}):
        errs.append("session_ohlc_incorrectly_triggers_h21_gate")

    # H20 is daily ohlc only — neither session nor gap is H20
    if northset_has_finite_ohlc_identity_rate({k_sess: 1.0}):
        errs.append("session_ohlc_incorrectly_triggers_h20_gate")
    if northset_has_finite_ohlc_identity_rate({k_gap: 1.0}):
        errs.append("gap_finite_rate_incorrectly_triggers_h20_gate")

    if H20_HYPOTHESIS_ID in {H23_HYPOTHESIS_ID, H21_HYPOTHESIS_ID}:
        errs.append("h20_hypothesis_collapsed_into_session_or_book_binds")
    return errs


def ohlc_identity_vs_session_ohlc_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate daily ``ohlc_identity_rate`` with ``session_ohlc_identity_rate``.

    Distinct clocks (daily bars vs session L2). When **both** keys are present:

    - Key identity: ``ohlc_identity_rate`` ≠ ``session_ohlc_identity_rate``
    - H20 finite gate binds to **daily** ``ohlc_identity_rate`` only
      (``H20_HYPOTHESIS_ID``); session OHLC is not an H20 metric
    - Numeric equality on clean synth is allowed

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_daily = "ohlc_identity_rate"
    k_sess = "session_ohlc_identity_rate"
    if k_daily not in blob or k_sess not in blob:
        return []

    errs: list[str] = []
    if k_daily == k_sess:
        errs.append("ohlc_identity_session_ohlc_keys_collapsed")

    # Catalog lock: H20 gate helper watches daily key only
    if not callable(northset_has_finite_ohlc_identity_rate):
        errs.append("ohlc_identity_h20_gate_helper_missing")
    # Session key must not be the daily key string (structural)
    if k_sess == "ohlc_identity_rate":
        errs.append("session_ohlc_identity_rate_aliased_to_daily")

    # Ensure H20 id stays distinct from session H23/H24/H29 binds
    session_hyps = {
        H23_HYPOTHESIS_ID,
        H24_HYPOTHESIS_ID,
        H29_HYPOTHESIS_ID,
    }
    if H20_HYPOTHESIS_ID in session_hyps:
        errs.append("h20_hypothesis_collapsed_into_session_binds")
    return errs


def ohlc_identity_vs_session_reconstructs_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate daily ``ohlc_identity_rate`` with ``session_reconstructs_daily_rate``.

    DATA_CONTRACTS: daily OHLC envelope identity (H20) ≠ session envelope that
    reconstructs the daily bar (H23). When both keys are present:

    - Distinct key identity
    - Distinct hyp ids (H20 ≠ H23)
    - H20 finite gate on daily ohlc only; H23 binds reconstructs only
    - Numeric equality on clean synth is allowed

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_ohlc = "ohlc_identity_rate"
    k_recon = "session_reconstructs_daily_rate"
    if k_ohlc not in blob or k_recon not in blob:
        return []

    errs: list[str] = []
    if k_ohlc == k_recon:
        errs.append("ohlc_identity_session_reconstructs_keys_collapsed")
    if H20_HYPOTHESIS_ID == H23_HYPOTHESIS_ID:
        errs.append("h20_h23_hypothesis_ids_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_recon) != H23_HYPOTHESIS_ID:
        errs.append("session_reconstructs_daily_rate_not_bound_h23")
    if by_key.get(k_ohlc) == H23_HYPOTHESIS_ID:
        errs.append("ohlc_identity_rate_incorrectly_bound_h23")

    if not northset_has_finite_ohlc_identity_rate({k_ohlc: 1.0}):
        errs.append("ohlc_identity_h20_gate_helper_broken")
    if northset_has_finite_ohlc_identity_rate({k_recon: 1.0}):
        errs.append("session_reconstructs_incorrectly_triggers_h20_gate")
    return errs


def ohlc_identity_vs_session_chain_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate daily ``ohlc_identity_rate`` with H29 ``session_chain_rate``.

    DATA_CONTRACTS: daily OHLC envelope identity (H20) ≠ session chain identity
    (H29). When both keys are present:

    - Distinct key identity
    - Distinct hyp ids (H20 ≠ H29)
    - H20 finite gate on daily ohlc only; H29 binds chain only
    - Daily ohlc must not bind H23/H24/H29; chain must not trigger H20 gate
    - Numeric equality on clean synth is allowed

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_ohlc = "ohlc_identity_rate"
    k_chain = "session_chain_rate"
    if k_ohlc not in blob or k_chain not in blob:
        return []

    errs: list[str] = []
    if k_ohlc == k_chain:
        errs.append("ohlc_identity_session_chain_keys_collapsed")
    if H20_HYPOTHESIS_ID == H29_HYPOTHESIS_ID:
        errs.append("h20_h29_hypothesis_ids_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_chain) != H29_HYPOTHESIS_ID:
        errs.append("session_chain_rate_not_bound_h29")
    if by_key.get(k_ohlc) in {
        H23_HYPOTHESIS_ID,
        H24_HYPOTHESIS_ID,
        H29_HYPOTHESIS_ID,
    }:
        errs.append("ohlc_identity_rate_incorrectly_bound_session_hyp")

    if not northset_has_finite_ohlc_identity_rate({k_ohlc: 1.0}):
        errs.append("ohlc_identity_h20_gate_helper_broken")
    if northset_has_finite_ohlc_identity_rate({k_chain: 1.0}):
        errs.append("session_chain_incorrectly_triggers_h20_gate")
    return errs


def ohlc_identity_vs_volume_conservation_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate daily ``ohlc_identity_rate`` with H24 ``session_volume_conservation_rate``.

    DATA_CONTRACTS: daily OHLC envelope identity (H20) ≠ session volume conservation
    (H24). When both keys are present:

    - Distinct key identity
    - Distinct hyp ids (H20 ≠ H24)
    - H24 binds volume only; H20 finite gate on daily ohlc only
    - Daily ohlc must not bind H23/H24/H29; volume must not trigger H20 gate
    - Numeric equality on clean synth is allowed

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_ohlc = "ohlc_identity_rate"
    k_vol = "session_volume_conservation_rate"
    if k_ohlc not in blob or k_vol not in blob:
        return []

    errs: list[str] = []
    if k_ohlc == k_vol:
        errs.append("ohlc_identity_volume_conservation_keys_collapsed")
    if H20_HYPOTHESIS_ID == H24_HYPOTHESIS_ID:
        errs.append("h20_h24_hypothesis_ids_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_vol) != H24_HYPOTHESIS_ID:
        errs.append("session_volume_conservation_rate_not_bound_h24")
    if by_key.get(k_ohlc) in {
        H23_HYPOTHESIS_ID,
        H24_HYPOTHESIS_ID,
        H29_HYPOTHESIS_ID,
    }:
        errs.append("ohlc_identity_rate_incorrectly_bound_session_hyp")

    if not northset_has_finite_ohlc_identity_rate({k_ohlc: 1.0}):
        errs.append("ohlc_identity_h20_gate_helper_broken")
    if northset_has_finite_ohlc_identity_rate({k_vol: 1.0}):
        errs.append("session_volume_conservation_incorrectly_triggers_h20_gate")
    return errs


def ohlc_identity_rate_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: ohlc_identity_rate ∈ [0, 1] when finite.

    Daily bar OHLC identity rate (≠ session_ohlc_identity_rate). NaN skipped.
    Research diagnostic only; never live Sharpe. Does not touch kyle_* keys.
    """
    if not isinstance(blob, dict):
        return []
    try:
        x = float(blob.get("ohlc_identity_rate"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return []
    if x != x:
        return []
    if abs(x) == float("inf"):
        return ["ohlc_identity_rate_non_finite"]
    if not (0.0 <= x <= 1.0):
        return ["ohlc_identity_rate_out_of_unit_interval"]
    return []


def session_chain_vs_session_identity_siblings_never_equate_honesty_errors(
    blob: object,
) -> list[str]:
    """Never equate H29 session_chain_rate with H23 reconstructs / H24 conservation.

    Sibling session identity rates — chain ≠ reconstructs ≠ volume conservation.
    When ``session_chain_rate`` is present alongside either sibling:

    - Catalog bind lock: chain → H29; reconstructs → H23; conservation → H24
    - Distinct hyp ids (no collapse)
    - Numeric equality on clean synth is allowed

    Skip when chain key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_chain = "session_chain_rate"
    k_recon = "session_reconstructs_daily_rate"
    k_vol = "session_volume_conservation_rate"
    if k_chain not in blob:
        return []
    if k_recon not in blob and k_vol not in blob:
        return []

    errs: list[str] = []
    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    chain_hyp = by_key.get(k_chain)
    recon_hyp = by_key.get(k_recon)
    vol_hyp = by_key.get(k_vol)
    if chain_hyp != H29_HYPOTHESIS_ID:
        errs.append("session_chain_rate_not_bound_h29")
    if k_recon in blob and recon_hyp != H23_HYPOTHESIS_ID:
        errs.append("session_reconstructs_daily_rate_not_bound_h23")
    if k_vol in blob and vol_hyp != H24_HYPOTHESIS_ID:
        errs.append("session_volume_conservation_rate_not_bound_h24")
    if chain_hyp is not None and recon_hyp is not None and chain_hyp == recon_hyp:
        errs.append("session_chain_reconstructs_hypothesis_collapsed")
    if chain_hyp is not None and vol_hyp is not None and chain_hyp == vol_hyp:
        errs.append("session_chain_volume_conservation_hypothesis_collapsed")
    return errs


def session_chain_rate_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: session_chain_rate ∈ [0, 1] when finite.

    Northset session reconstruction chain rate. NaN/absent skip; ±inf fail-closed.
    Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    val = blob.get("session_chain_rate")
    try:
        x = float(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return []
    if x != x:
        return []
    if abs(x) == float("inf") or not (0.0 <= x <= 1.0):
        return ["session_chain_rate_out_of_unit_interval"]
    return []


def session_mean_jump_ratio_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: session_mean_jump_ratio ∈ [0, 1] when finite.

    BNS bipower jump share is clipped to [0,1] in estimators.session_bipower_jump;
    out-of-range or ±inf is dishonest. NaN skipped.
    Research diagnostic only; never live Sharpe. Does not touch kyle_* keys.
    """
    if not isinstance(blob, dict):
        return []
    try:
        x = float(blob.get("session_mean_jump_ratio"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return []
    if x != x:
        return []
    if abs(x) == float("inf"):
        return ["session_mean_jump_ratio_non_finite"]
    if not (0.0 <= x <= 1.0):
        return ["session_mean_jump_ratio_out_of_unit_interval"]
    return []


def sweep_follow_signed_mean_ic_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: sweep_follow_signed_mean_ic finite when present (signed OK).

    ±inf fail-closed; NaN/absent skip. ≠ sweep_reject_signed_mean_ic.
    Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    if "sweep_follow_signed_mean_ic" not in blob:
        return []
    try:
        x = float(blob.get("sweep_follow_signed_mean_ic"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return ["sweep_follow_signed_mean_ic_non_numeric"]
    if x != x:
        return []
    if abs(x) == float("inf"):
        return ["sweep_follow_signed_mean_ic_non_finite_fail_closed"]
    return []


def sweep_reject_signed_mean_ic_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: sweep_reject_signed_mean_ic finite when present (signed OK).

    ±inf fail-closed; NaN/absent skip. Companion to H33 path; not a live gate.
    Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    if "sweep_reject_signed_mean_ic" not in blob:
        return []
    try:
        x = float(blob.get("sweep_reject_signed_mean_ic"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return ["sweep_reject_signed_mean_ic_non_numeric"]
    if x != x:
        return []
    if abs(x) == float("inf"):
        return ["sweep_reject_signed_mean_ic_non_finite_fail_closed"]
    return []


def sweep_reject_event_mean_bps_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: sweep_reject_event_mean_bps finite when present (signed OK).

    ±inf fail-closed; NaN/absent skip. Never equate to sweep_follow_* keys.
    Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    if "sweep_reject_event_mean_bps" not in blob:
        return []
    try:
        x = float(blob.get("sweep_reject_event_mean_bps"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return ["sweep_reject_event_mean_bps_non_numeric"]
    if x != x:
        return []
    if abs(x) == float("inf"):
        return ["sweep_reject_event_mean_bps_non_finite_fail_closed"]
    return []


def sweep_follow_event_mean_bps_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: sweep_follow_event_mean_bps finite when present.

    Signed pre-cost event excess bps — may be negative; only ±inf is dishonest.
    NaN/absent skipped.

    Honesty: never equate to ``sweep_follow_cost_adjusted_mean_bps`` (post-cost)
    or to ``sweep_reject_event_mean_bps`` (different signal). Research diagnostic
    only; never live Sharpe. Does not touch kyle_* keys.
    """
    if not isinstance(blob, dict):
        return []
    if "sweep_follow_event_mean_bps" not in blob:
        return []
    try:
        x = float(blob.get("sweep_follow_event_mean_bps"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return ["sweep_follow_event_mean_bps_non_numeric"]
    if x != x:
        return []
    if abs(x) == float("inf"):
        return ["sweep_follow_event_mean_bps_non_finite"]
    return []


def sweep_follow_cost_adjusted_mean_bps_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: sweep_follow_cost_adjusted_mean_bps finite when present.

    Signed bps after cost haircut — may be negative; only ±inf / non-numeric is
    dishonest. NaN skipped.

    Honesty: never equate to ``sweep_follow_event_mean_bps`` (pre-cost excess).
    Equality is allowed when cost is zero but they are different statistics —
    do not soft-fail on coincidence. Research diagnostic only; never live Sharpe.
    Does not touch kyle_* keys.
    """
    if not isinstance(blob, dict):
        return []
    try:
        x = float(blob.get("sweep_follow_cost_adjusted_mean_bps"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return []
    if x != x:
        return []
    if abs(x) == float("inf"):
        return ["sweep_follow_cost_adjusted_mean_bps_non_finite"]
    return []


def sweep_reject_cost_adjusted_mean_bps_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: sweep_reject_cost_adjusted_mean_bps finite when present.

    Signed bps after cost haircut — may be negative; only ±inf / non-numeric is
    dishonest. NaN/absent skipped.

    Honesty: never equate to ``sweep_reject_event_mean_bps`` (pre-cost) or to
    ``sweep_follow_cost_adjusted_mean_bps`` (different signal). Research
    diagnostic only; never live Sharpe. Does not touch kyle_* keys.
    """
    if not isinstance(blob, dict):
        return []
    if "sweep_reject_cost_adjusted_mean_bps" not in blob:
        return []
    try:
        x = float(blob.get("sweep_reject_cost_adjusted_mean_bps"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return ["sweep_reject_cost_adjusted_mean_bps_non_numeric"]
    if x != x:
        return []
    if abs(x) == float("inf"):
        return ["sweep_reject_cost_adjusted_mean_bps_non_finite"]
    return []


def northset_n_bars_scored_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: n_bars / n_fused / n_scored when present — ints ≥0; chain order.

    Top-level northset receipt (≠ kyle nest n_fused/n_scored). When present:
    - each is a non-negative integer (NaN skip per key)
    - ``n_scored ≤ n_fused`` when both present
    - ``n_fused ≤ n_bars`` when both present
    - ``n_scored ≤ n_bars`` when both present (legacy pair)

    Never invent always-on ``min_names`` (unstamped). Research diagnostic only;
    never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    errs: list[str] = []

    def _nonneg_int(key: str) -> int | None:
        if key not in blob:
            return None
        val = blob.get(key)
        try:
            x = float(val)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append(f"{key}_non_numeric")
            return None
        if x != x:
            return None
        if abs(x) == float("inf") or x < 0 or abs(x - int(x)) > 1e-9:
            errs.append(f"{key}_not_nonneg_int")
            return None
        return int(x)

    nb = _nonneg_int("n_bars")
    nf = _nonneg_int("n_fused")
    ns = _nonneg_int("n_scored")
    if nb is not None and ns is not None and ns > nb:
        errs.append("n_scored_gt_n_bars")
    if nf is not None and ns is not None and ns > nf:
        errs.append("n_scored_gt_n_fused")
    if nb is not None and nf is not None and nf > nb:
        errs.append("n_fused_gt_n_bars")
    return errs


def ofi_p_ic_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: ofi_p_ic / ofi_t_ic finite when present (signed OK).

    ±inf fail-closed; NaN/absent skip. ≠ ofi_mean_ic alias path may coexist.
    Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    errs: list[str] = []
    for key in ("ofi_p_ic", "ofi_t_ic"):
        if key not in blob:
            continue
        try:
            x = float(blob.get(key))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append(f"{key}_non_numeric")
            continue
        if x != x:
            continue
        if abs(x) == float("inf"):
            errs.append(f"{key}_non_finite_fail_closed")
    return errs


def northset_vpin_sweep_fold_honesty_errors(blob: object) -> list[str]:
    """Soft-verify vpin_p/t_ic, n_sweep_low, sweep_*_fold_positive_fraction.

    - vpin_p_ic / vpin_t_ic finite when present (signed OK; ±inf fail-closed)
    - n_sweep_low ≥ 0 when present
    - sweep_reject_fold_positive_fraction / sweep_follow_fold_positive_fraction
      ∈ [0, 1] when finite
    NaN/absent skipped. Research diagnostic only; never live Sharpe.
    Does not touch kyle_* / book_metrics METRICS_* keys.
    """
    if not isinstance(blob, dict):
        return []
    errs: list[str] = []

    for key in ("vpin_p_ic", "vpin_t_ic"):
        if key not in blob:
            continue
        try:
            x = float(blob.get(key))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append(f"{key}_non_numeric")
            continue
        if x != x:
            continue
        if abs(x) == float("inf"):
            errs.append(f"{key}_non_finite_fail_closed")

    if "n_sweep_low" in blob:
        try:
            n = float(blob.get("n_sweep_low"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append("n_sweep_low_non_numeric")
        else:
            if n != n or abs(n) == float("inf"):
                errs.append("n_sweep_low_non_finite")
            elif n < 0.0:
                errs.append("n_sweep_low_negative")

    for key in (
        "sweep_reject_fold_positive_fraction",
        "sweep_follow_fold_positive_fraction",
    ):
        if key not in blob:
            continue
        try:
            x = float(blob.get(key))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append(f"{key}_non_numeric")
            continue
        if x != x:
            continue
        if abs(x) == float("inf"):
            errs.append(f"{key}_non_finite")
            continue
        if not (0.0 <= x <= 1.0):
            errs.append(f"{key}_out_of_unit_interval")
    return errs


def microprice_p_ic_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: microprice_p_ic / microprice_t_ic finite when present (signed OK).

    ±inf fail-closed; NaN/absent skip. Companion aliases of microprice_minus_mid_bps IC.
    Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    errs: list[str] = []
    for key in ("microprice_p_ic", "microprice_t_ic"):
        if key not in blob:
            continue
        try:
            x = float(blob.get(key))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append(f"{key}_non_numeric")
            continue
        if x != x:
            continue
        if abs(x) == float("inf"):
            errs.append(f"{key}_non_finite_fail_closed")
    return errs


def clv_p_ic_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: clv_p_ic / clv_t_ic finite when present (signed OK).

    ±inf fail-closed; NaN/absent skip. Companion aliases of close_location_value IC
    (≠ microprice_p_ic / microprice_t_ic). Research diagnostic only; never live Sharpe.
    Does not touch kyle_* / book_metrics METRICS_* keys.
    """
    if not isinstance(blob, dict):
        return []
    errs: list[str] = []
    for key in ("clv_p_ic", "clv_t_ic"):
        if key not in blob:
            continue
        try:
            x = float(blob.get(key))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append(f"{key}_non_numeric")
            continue
        if x != x:
            continue
        if abs(x) == float("inf"):
            errs.append(f"{key}_non_finite_fail_closed")
    return errs


def northset_queue_imbalance_ic_honesty_errors(blob: object) -> list[str]:
    """Soft-verify queue_imbalance_* IC companions when present.

    - ``queue_imbalance_mean_ic`` / ``queue_imbalance_mean_rank_ic`` / ``_t_ic`` finite
    - ``queue_imbalance_p_ic`` ∈ [0, 1] when finite
    - ``queue_imbalance_n_dates`` ≥ 0 when finite
    Never equate to ``queue_imbalance_mean`` (mean ≠ IC) or ofi_* IC.
    Research diagnostic only; never live Sharpe. Off kyle_ofi / METRICS_*.
    """
    if not isinstance(blob, dict):
        return []
    errs: list[str] = []
    for key in (
        "queue_imbalance_mean_ic",
        "queue_imbalance_mean_rank_ic",
        "queue_imbalance_t_ic",
    ):
        if key not in blob:
            continue
        try:
            x = float(blob.get(key))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append(f"{key}_non_numeric")
            continue
        if x != x:
            continue
        if abs(x) == float("inf"):
            errs.append(f"{key}_non_finite_fail_closed")
    if "queue_imbalance_p_ic" in blob:
        try:
            p = float(blob.get("queue_imbalance_p_ic"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append("queue_imbalance_p_ic_non_numeric")
        else:
            if p == p and abs(p) != float("inf") and not (0.0 <= p <= 1.0):
                errs.append("queue_imbalance_p_ic_out_of_unit_interval")
            elif p == p and abs(p) == float("inf"):
                errs.append("queue_imbalance_p_ic_non_finite_fail_closed")
    if "queue_imbalance_n_dates" in blob:
        try:
            n = float(blob.get("queue_imbalance_n_dates"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append("queue_imbalance_n_dates_non_numeric")
        else:
            if n == n and abs(n) != float("inf") and n < 0.0:
                errs.append("queue_imbalance_n_dates_negative")
            elif n == n and abs(n) == float("inf"):
                errs.append("queue_imbalance_n_dates_non_finite")
    return errs


def northset_vpin_ic_pack_honesty_errors(blob: object) -> list[str]:
    """Soft-verify vpin_* IC pack companions beyond finite-only p/t.

    - ``vpin_mean_ic`` / ``vpin_mean_rank_ic`` / ``vpin_t_ic`` finite
    - ``vpin_p_ic`` ∈ [0, 1] when finite (stricter than finite-only)
    - ``vpin_n_dates`` ≥ 0 when finite
    Never equate to ``vpin_mean``. Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    errs: list[str] = []
    for key in ("vpin_mean_ic", "vpin_mean_rank_ic", "vpin_t_ic"):
        if key not in blob:
            continue
        try:
            x = float(blob.get(key))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append(f"{key}_non_numeric")
            continue
        if x != x:
            continue
        if abs(x) == float("inf"):
            errs.append(f"{key}_non_finite_fail_closed")
    if "vpin_p_ic" in blob:
        try:
            p = float(blob.get("vpin_p_ic"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append("vpin_p_ic_non_numeric")
        else:
            if p == p and abs(p) != float("inf") and not (0.0 <= p <= 1.0):
                errs.append("vpin_p_ic_out_of_unit_interval")
            elif p == p and abs(p) == float("inf"):
                errs.append("vpin_p_ic_non_finite_fail_closed")
    if "vpin_n_dates" in blob:
        try:
            n = float(blob.get("vpin_n_dates"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append("vpin_n_dates_non_numeric")
        else:
            if n == n and abs(n) != float("inf") and n < 0.0:
                errs.append("vpin_n_dates_negative")
            elif n == n and abs(n) == float("inf"):
                errs.append("vpin_n_dates_non_finite")
    return errs


def candle_structure_ic_implies_mean_honesty_errors(blob: object) -> list[str]:
    """Soft-verify structure LOB IC⇒mean pairs on candle_order_book.

    Covers imbalance_top, queue_imbalance, tob_size_share, bid/ask size concentration.
    Never equate IC to mean. Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    if blob.get("family") != "candle_order_book":
        return []
    errs: list[str] = []
    specs = (
        ("ic_imbalance_top", "mean_imbalance_top", -1.0, 1.0),
        ("ic_queue_imbalance", "mean_queue_imbalance", -1.0, 1.0),
        ("ic_tob_size_share", "mean_tob_size_share", 0.0, 1.0),
        ("ic_bid_size_concentration_top", "mean_bid_size_concentration_top", 0.0, 1.0),
        ("ic_ask_size_concentration_top", "mean_ask_size_concentration_top", 0.0, 1.0),
    )
    for ic_key, mean_key, lo, hi in specs:
        if ic_key not in blob:
            continue
        if mean_key not in blob:
            errs.append(f"{mean_key}_missing_while_{ic_key}_scored")
            continue
        try:
            m = float(blob.get(mean_key))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append(f"{mean_key}_non_numeric")
            continue
        if m != m:
            errs.append(f"{mean_key}_nan_while_{ic_key}_scored")
        elif abs(m) == float("inf") or not (lo <= m <= hi):
            errs.append(f"{mean_key}_out_of_unit_interval")
    return errs


def northset_ofi_lag_ic_honesty_errors(blob: object) -> list[str]:
    """Soft-verify ofi_lag_* IC companions when present.

    - ``ofi_lag_mean_ic`` / ``ofi_lag_mean_rank_ic`` / ``ofi_lag_t_ic`` finite
    - ``ofi_lag_p_ic`` ∈ [0, 1] when finite
    - ``ofi_lag_n_dates`` ≥ 0 when finite
    Never equate to ``ofi_mean_ic`` / ``ofi_p_ic`` or ``ofi_lag1_corr``.
    Research diagnostic only; never live Sharpe. Off kyle_ofi / METRICS_*.
    """
    if not isinstance(blob, dict):
        return []
    errs: list[str] = []
    for key in ("ofi_lag_mean_ic", "ofi_lag_mean_rank_ic", "ofi_lag_t_ic"):
        if key not in blob:
            continue
        try:
            x = float(blob.get(key))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append(f"{key}_non_numeric")
            continue
        if x != x:
            continue
        if abs(x) == float("inf"):
            errs.append(f"{key}_non_finite_fail_closed")
    if "ofi_lag_p_ic" in blob:
        try:
            p = float(blob.get("ofi_lag_p_ic"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append("ofi_lag_p_ic_non_numeric")
        else:
            if p == p and abs(p) != float("inf") and not (0.0 <= p <= 1.0):
                errs.append("ofi_lag_p_ic_out_of_unit_interval")
            elif p == p and abs(p) == float("inf"):
                errs.append("ofi_lag_p_ic_non_finite_fail_closed")
    if "ofi_lag_n_dates" in blob:
        try:
            n = float(blob.get("ofi_lag_n_dates"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append("ofi_lag_n_dates_non_numeric")
        else:
            if n == n and abs(n) != float("inf") and n < 0.0:
                errs.append("ofi_lag_n_dates_negative")
            elif n == n and abs(n) == float("inf"):
                errs.append("ofi_lag_n_dates_non_finite")
    return errs


def candle_depth_imbalance_ic_implies_mean_honesty_errors(blob: object) -> list[str]:
    """Soft-verify depth imbalance IC⇒mean pairs on candle_order_book.

    - ``ic_imbalance_depth`` ⇒ ``mean_depth_imbalance`` ∈ [-1, 1]
    - ``ic_depth_imbalance_abs`` ⇒ ``mean_depth_imbalance_abs`` ∈ [0, 1]
    Never equate IC to mean; never equate abs to signed. Research diagnostic only.
    """
    if not isinstance(blob, dict):
        return []
    if blob.get("family") != "candle_order_book":
        return []
    errs: list[str] = []
    if "ic_imbalance_depth" in blob:
        if "mean_depth_imbalance" not in blob:
            errs.append("mean_depth_imbalance_missing_while_imbalance_depth_ic_scored")
        else:
            try:
                m = float(blob.get("mean_depth_imbalance"))  # type: ignore[arg-type]
            except (TypeError, ValueError):
                errs.append("mean_depth_imbalance_non_numeric")
            else:
                if m != m:
                    errs.append("mean_depth_imbalance_nan_while_imbalance_depth_ic_scored")
                elif abs(m) == float("inf") or not (-1.0 <= m <= 1.0):
                    errs.append("mean_depth_imbalance_out_of_unit_interval")
    if "ic_depth_imbalance_abs" in blob:
        if "mean_depth_imbalance_abs" not in blob:
            errs.append("mean_depth_imbalance_abs_missing_while_depth_imbalance_abs_ic_scored")
        else:
            try:
                m = float(blob.get("mean_depth_imbalance_abs"))  # type: ignore[arg-type]
            except (TypeError, ValueError):
                errs.append("mean_depth_imbalance_abs_non_numeric")
            else:
                if m != m:
                    errs.append("mean_depth_imbalance_abs_nan_while_depth_imbalance_abs_ic_scored")
                elif abs(m) == float("inf") or not (0.0 <= m <= 1.0):
                    errs.append("mean_depth_imbalance_abs_out_of_unit_interval")
    return errs


def ofi_mean_ic_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: ofi_mean_ic finite when present (signed OK).

    ±inf fail-closed; NaN/absent skip. ≠ imbalance_top_mean_ic.
    Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    if "ofi_mean_ic" not in blob:
        return []
    try:
        x = float(blob.get("ofi_mean_ic"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return ["ofi_mean_ic_non_numeric"]
    if x != x:
        return []
    if abs(x) == float("inf"):
        return ["ofi_mean_ic_non_finite_fail_closed"]
    return []


def _ic_pack_honesty_errors(
    blob: dict,
    *,
    mean_ic_key: str,
    rank_ic_key: str | None,
    t_key: str,
    p_key: str,
    n_key: str | None,
) -> list[str]:
    """Shared IC-pack soft-verify: mean/rank/t finite; p∈[0,1]; n≥0."""
    errs: list[str] = []
    for key in (mean_ic_key, rank_ic_key, t_key):
        if key is None or key not in blob:
            continue
        try:
            x = float(blob.get(key))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append(f"{key}_non_numeric")
            continue
        if x != x:
            continue
        if abs(x) == float("inf"):
            errs.append(f"{key}_non_finite_fail_closed")
    if p_key in blob:
        try:
            p = float(blob.get(p_key))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append(f"{p_key}_non_numeric")
        else:
            if p == p and abs(p) != float("inf") and not (0.0 <= p <= 1.0):
                errs.append(f"{p_key}_out_of_unit_interval")
            elif p == p and abs(p) == float("inf"):
                errs.append(f"{p_key}_non_finite_fail_closed")
    if n_key is not None and n_key in blob:
        try:
            n = float(blob.get(n_key))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append(f"{n_key}_non_numeric")
        else:
            if n == n and abs(n) != float("inf") and n < 0.0:
                errs.append(f"{n_key}_negative")
            elif n == n and abs(n) == float("inf"):
                errs.append(f"{n_key}_non_finite")
    return errs


def candle_order_book_claim_honesty_errors(blob: object) -> list[str]:
    """Soft-verify candle_order_book research_only + claim markers.

    When ``family == "candle_order_book"`` and either key present:
    ``research_only is True`` and ``claim == "research_diagnostic_only"``.
    Coupling: ``research_only is True`` ⇒ claim present and correct.
    Research diagnostic only; never live Sharpe. Off kyle_ofi / METRICS_*.
    """
    if not isinstance(blob, dict):
        return []
    if blob.get("family") != "candle_order_book":
        return []
    if "research_only" not in blob and "claim" not in blob:
        return []
    errs: list[str] = []
    if "research_only" in blob and blob.get("research_only") is not True:
        errs.append("candle_research_only_missing_or_false")
    if blob.get("research_only") is True:
        if "claim" not in blob:
            errs.append("candle_claim_missing_while_research_only_true")
        elif blob.get("claim") != "research_diagnostic_only":
            errs.append("candle_claim_not_research_diagnostic_only")
    elif "claim" in blob:
        # claim ⇒ research_only: a present claim without the honesty flag is not
        # acceptable evidence (the coupling must hold in both directions).
        if "research_only" not in blob:
            errs.append("candle_research_only_missing_or_false")
        if blob.get("claim") != "research_diagnostic_only":
            errs.append("candle_claim_not_research_diagnostic_only")
    return errs


def candle_all_ic_pearson_unit_honesty_errors(blob: object) -> list[str]:
    """Soft-verify every ``ic_*_pearson`` ∈ [-1, 1] when finite on candle receipts."""
    if not isinstance(blob, dict):
        return []
    if blob.get("family") != "candle_order_book":
        return []
    errs: list[str] = []
    for key, raw in blob.items():
        if not isinstance(key, str) or not key.startswith("ic_") or not key.endswith("_pearson"):
            continue
        if raw is None:
            continue
        try:
            x = float(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append(f"{key}_non_numeric")
            continue
        if x != x:
            continue
        if abs(x) == float("inf"):
            errs.append(f"{key}_non_finite_fail_closed")
        elif not (-1.0 <= x <= 1.0):
            errs.append(f"{key}_out_of_unit_interval")
    return errs


def candle_all_ic_p_unit_honesty_errors(blob: object) -> list[str]:
    """Soft-verify every candle ``ic_*_p`` ∈ [0, 1] when finite.

    Companion catch-all to :func:`candle_all_ic_pearson_unit_honesty_errors`.
    Skips non-p suffixes (``_pearson``, ``_n_dates``, bare ``ic_*``).
    """
    if not isinstance(blob, dict):
        return []
    if blob.get("family") != "candle_order_book":
        return []
    errs: list[str] = []
    for key, raw in blob.items():
        if not isinstance(key, str) or not key.startswith("ic_") or not key.endswith("_p"):
            continue
        if key.endswith("_pearson"):
            continue
        if raw is None:
            continue
        try:
            x = float(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append(f"{key}_non_numeric")
            continue
        if x != x:
            continue
        if abs(x) == float("inf") or not (0.0 <= x <= 1.0):
            errs.append(f"{key}_out_of_unit_interval")
    return errs


def candle_all_ic_t_finite_honesty_errors(blob: object) -> list[str]:
    """Soft-verify every candle ``ic_*_t`` is finite when present (not ±inf)."""
    if not isinstance(blob, dict):
        return []
    if blob.get("family") != "candle_order_book":
        return []
    errs: list[str] = []
    for key, raw in blob.items():
        if not isinstance(key, str) or not key.startswith("ic_") or not key.endswith("_t"):
            continue
        if raw is None:
            continue
        try:
            x = float(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append(f"{key}_non_numeric")
            continue
        if x != x:
            continue
        if abs(x) == float("inf"):
            errs.append(f"{key}_non_finite_fail_closed")
    return errs


def candle_all_ic_n_dates_nonneg_honesty_errors(blob: object) -> list[str]:
    """Soft-verify every candle ``ic_*_n_dates`` ≥ 0 when finite."""
    if not isinstance(blob, dict):
        return []
    if blob.get("family") != "candle_order_book":
        return []
    errs: list[str] = []
    for key, raw in blob.items():
        if not isinstance(key, str) or not key.startswith("ic_") or not key.endswith("_n_dates"):
            continue
        if raw is None:
            continue
        try:
            x = float(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append(f"{key}_non_numeric")
            continue
        if x != x:
            continue
        if abs(x) == float("inf") or x < 0.0:
            errs.append(f"{key}_negative_or_non_finite")
    return errs


def northset_all_rate_unit_honesty_errors(blob: object) -> list[str]:
    """Soft-verify every top-level ``*_rate`` ∈ [0, 1] when finite.

    Covers candle-pattern rates (doji/hammer/…) and sweep_*_rate companions
    alongside identity rates. Skip kyle_*/METRICS_*. Research diagnostic only.
    """
    if not isinstance(blob, dict):
        return []
    errs: list[str] = []
    for key, raw in blob.items():
        if not isinstance(key, str) or not key.endswith("_rate"):
            continue
        if key.startswith("kyle_") or "METRICS_" in key:
            continue
        if raw is None:
            continue
        try:
            x = float(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append(f"{key}_non_numeric")
            continue
        if x != x:
            continue
        if abs(x) == float("inf") or not (0.0 <= x <= 1.0):
            errs.append(f"{key}_out_of_unit_interval")
    return errs


def northset_sweep_evidence_blob_honesty_errors(blob: object) -> list[str]:
    """Soft-verify nested ``sweep_evidence`` research-diagnostic contract.

    When ``sweep_evidence`` dict present:
    - ``research_only is True`` when key present
    - ``horizons`` non-empty list of positive ints when present
    - each event_studies row: ``sample_adequate`` bool; if False then ``reject_fdr`` is not True
    - ``n_events`` / ``n_dates`` ≥ 0 when finite
    Never promotes inadequate samples. Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    ev = blob.get("sweep_evidence")
    if not isinstance(ev, dict):
        return []
    errs: list[str] = []
    if "research_only" in ev and ev.get("research_only") is not True:
        errs.append("sweep_evidence_research_only_missing_or_false")
    idx = ev.get("inference_index")
    if idx is not None and idx != "calendar_including_idle_zeros":
        errs.append("sweep_evidence_inference_index_invalid")
    if "horizons" in ev:
        hz = ev.get("horizons")
        if not isinstance(hz, list) or not hz:
            errs.append("sweep_evidence_horizons_invalid")
        else:
            for h in hz:
                try:
                    hf = float(h)
                except (TypeError, ValueError):
                    errs.append("sweep_evidence_horizons_invalid")
                    break
                # Reject non-finite / non-integral values before int() (inf raises).
                if hf != hf or abs(hf) == float("inf") or hf <= 0.0 or hf != int(hf):
                    errs.append("sweep_evidence_horizons_invalid")
                    break
    studies = ev.get("event_studies")
    if isinstance(studies, list):
        for i, row in enumerate(studies):
            if not isinstance(row, dict):
                errs.append(f"sweep_evidence_event_studies_{i}_not_dict")
                continue
            if "sample_adequate" in row and type(row.get("sample_adequate")) is not bool:
                errs.append(f"sweep_evidence_event_studies_{i}_sample_adequate_not_bool")
            if "reject_fdr" in row and type(row.get("reject_fdr")) is not bool:
                # Truthy non-bools (1, "true") must not bypass the inadequate-
                # sample promotion guard below.
                errs.append(f"sweep_evidence_event_studies_{i}_reject_fdr_not_bool")
            if row.get("sample_adequate") is False and row.get("reject_fdr") is True:
                errs.append(f"sweep_evidence_event_studies_{i}_reject_fdr_with_inadequate_sample")
            for nk in ("n_events", "n_dates"):
                if nk not in row:
                    continue
                try:
                    n = float(row.get(nk))  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    errs.append(f"sweep_evidence_event_studies_{i}_{nk}_non_numeric")
                    continue
                if n == n and abs(n) != float("inf") and n < 0.0:
                    errs.append(f"sweep_evidence_event_studies_{i}_{nk}_negative")
    return errs


def northset_dm_park_honesty_errors(blob: object) -> list[str]:
    """Soft-verify Diebold–Mariano vs Park companions on northset.

    For each of gk/rs/split vs park:
    - ``dm_*_vs_park_p`` ∈ [0, 1] when finite
    - ``dm_*_vs_park_stat`` finite when present (signed OK)
    - ``dm_*_vs_park_preferred`` ∈ allowed label set for that pair
    Research diagnostic only; never live Sharpe. Off kyle_ofi / METRICS_*.
    """
    if not isinstance(blob, dict):
        return []
    errs: list[str] = []
    allowed = {
        "gk": frozenset({"park", "gk"}),
        "rs": frozenset({"park", "rs"}),
        "split": frozenset({"park", "split"}),
    }
    for stem in ("gk", "rs", "split"):
        p_key = f"dm_{stem}_vs_park_p"
        s_key = f"dm_{stem}_vs_park_stat"
        pref_key = f"dm_{stem}_vs_park_preferred"
        if p_key in blob:
            try:
                p = float(blob.get(p_key))  # type: ignore[arg-type]
            except (TypeError, ValueError):
                errs.append(f"{p_key}_non_numeric")
            else:
                if p == p and abs(p) != float("inf") and not (0.0 <= p <= 1.0):
                    errs.append(f"{p_key}_out_of_unit_interval")
                elif p == p and abs(p) == float("inf"):
                    errs.append(f"{p_key}_non_finite_fail_closed")
        if s_key in blob:
            try:
                s = float(blob.get(s_key))  # type: ignore[arg-type]
            except (TypeError, ValueError):
                errs.append(f"{s_key}_non_numeric")
            else:
                if s == s and abs(s) == float("inf"):
                    errs.append(f"{s_key}_non_finite_fail_closed")
        if pref_key in blob:
            pref = blob.get(pref_key)
            # isinstance guard: unhashable JSON values (list/dict) would raise
            # TypeError on set membership instead of failing closed.
            if not isinstance(pref, str) or pref not in allowed[stem]:
                errs.append(f"{pref_key}_invalid")
    return errs


_NORTHSET_SWEEP_EVIDENCE_SCOPE_ALLOWED = frozenset(
    {"synthetic", "empirical_adjusted", "fixture_raw_unadjusted"}
)
_NORTHSET_IMPACT_ESTIMATOR_SCOPE_ALLOWED = frozenset({"per_security_equal_weight"})
_NORTHSET_YANG_ZHANG_QLIKE_SCOPE_ALLOWED = frozenset({"per_security_expanding_oos"})
_NORTHSET_VPIN_METHOD_ALLOWED = frozenset(
    {"count_window_bulk_ofi_proxy", "volume_bucket_bulk_ofi_proxy"}
)
_NORTHSET_CORWIN_SCHULTZ_PAIR_SCOPE_ALLOWED = frozenset({"prior_and_current_bar"})
_NORTHSET_OVERNIGHT_GAP_METHOD_ALLOWED = frozenset({"event_close_to_next_open"})
_NORTHSET_TWO_WAY_INFERENCE_INDEX_ALLOWED = frozenset({"event_rows_not_calendar_zeros"})


def northset_sweep_evidence_scope_honesty_errors(blob: object) -> list[str]:
    """Soft-verify sweep_evidence_scope / impact_estimator_scope when present.

    Stamp contract (``bench_northset`` / DATA_CONTRACTS):
    - ``sweep_evidence_scope`` ∈ {synthetic, empirical_adjusted, fixture_raw_unadjusted}
    - when ``data_source == "SYNTHETIC"`` and scope present → scope must be ``synthetic``
    - when ``price_basis == "split_adjusted"`` and ``data_source`` is not SYNTHETIC
      and scope present → scope must be ``empirical_adjusted``
    - when ``price_basis == "raw_fixture_opt_out"`` and ``data_source`` is not SYNTHETIC
      and scope present → scope must be ``fixture_raw_unadjusted``
    - ``impact_estimator_scope`` ∈ {per_security_equal_weight} when present (stamp)
    - ``include_kyle_ofi`` bool when present (legacy companion check)

    Research diagnostic only; never live Sharpe. Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    errs: list[str] = []
    scope = blob.get("sweep_evidence_scope") if "sweep_evidence_scope" in blob else None
    if scope is not None:
        # isinstance guard: unhashable JSON values would raise TypeError on
        # set membership instead of failing closed.
        if not isinstance(scope, str) or scope not in _NORTHSET_SWEEP_EVIDENCE_SCOPE_ALLOWED:
            errs.append("sweep_evidence_scope_invalid")
        else:
            data_source = blob.get("data_source")
            price_basis = blob.get("price_basis")
            if isinstance(data_source, str) and data_source == "SYNTHETIC":
                if scope != "synthetic":
                    errs.append("sweep_evidence_scope_not_synthetic_despite_SYNTHETIC_data_source")
            elif isinstance(price_basis, str):
                if price_basis == "split_adjusted" and scope != "empirical_adjusted":
                    errs.append(
                        "sweep_evidence_scope_not_empirical_adjusted_despite_split_adjusted"
                    )
                elif price_basis == "raw_fixture_opt_out" and scope != "fixture_raw_unadjusted":
                    errs.append("sweep_evidence_scope_not_fixture_raw_despite_raw_fixture_opt_out")
    if "impact_estimator_scope" in blob:
        ies = blob.get("impact_estimator_scope")
        # Stamp contract: bench_northset hardcodes per_security_equal_weight.
        # isinstance guard: unhashable JSON values would raise TypeError.
        if not isinstance(ies, str) or ies not in _NORTHSET_IMPACT_ESTIMATOR_SCOPE_ALLOWED:
            errs.append("impact_estimator_scope_invalid")
    if "include_kyle_ofi" in blob and type(blob.get("include_kyle_ofi")) is not bool:
        errs.append("include_kyle_ofi_not_bool")
    if blob.get("family") == "northset":
        yz_scope = blob.get("yang_zhang_qlike_scope")
        yz_qlike = blob.get("yang_zhang_qlike_vs_cc")
        yz_finite = False
        try:
            yz_f = float(yz_qlike)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            yz_f = float("nan")
        yz_finite = yz_f == yz_f and abs(yz_f) != float("inf")
        if (yz_finite or "yang_zhang_qlike_scope" in blob) and (
            not isinstance(yz_scope, str)
            or yz_scope not in _NORTHSET_YANG_ZHANG_QLIKE_SCOPE_ALLOWED
        ):
            errs.append("yang_zhang_qlike_scope_invalid")
        vpin_method = blob.get("vpin_method")
        vpin_val = blob.get("vpin_mean")
        vpin_finite = False
        try:
            vp = float(vpin_val)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            vp = float("nan")
        vpin_finite = vp == vp and abs(vp) != float("inf")
        if (vpin_finite or "vpin_method" in blob) and (
            not isinstance(vpin_method, str) or vpin_method not in _NORTHSET_VPIN_METHOD_ALLOWED
        ):
            errs.append("vpin_method_invalid")
        cs_scope = blob.get("corwin_schultz_pair_scope")
        cs_spread = blob.get("corwin_schultz_spread")
        cs_finite = False
        try:
            cs_f = float(cs_spread)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            cs_f = float("nan")
        cs_finite = cs_f == cs_f and abs(cs_f) != float("inf")
        if (cs_finite or "corwin_schultz_pair_scope" in blob) and (
            not isinstance(cs_scope, str)
            or cs_scope not in _NORTHSET_CORWIN_SCHULTZ_PAIR_SCOPE_ALLOWED
        ):
            errs.append("corwin_schultz_pair_scope_invalid")
        gap_method = blob.get("sweep_overnight_gap_method")
        gap_p = blob.get("sweep_follow_overnight_gap_p")
        gap_finite = False
        try:
            gap_f = float(gap_p)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            gap_f = float("nan")
        gap_finite = gap_f == gap_f and abs(gap_f) != float("inf")
        if (gap_finite or "sweep_overnight_gap_method" in blob) and (
            not isinstance(gap_method, str)
            or gap_method not in _NORTHSET_OVERNIGHT_GAP_METHOD_ALLOWED
        ):
            errs.append("sweep_overnight_gap_method_invalid")
        two_way_idx = blob.get("sweep_two_way_inference_index")
        two_way_p = blob.get("sweep_follow_two_way_cluster_p")
        two_way_finite = False
        try:
            tw_f = float(two_way_p)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            tw_f = float("nan")
        two_way_finite = tw_f == tw_f and abs(tw_f) != float("inf")
        if (two_way_finite or "sweep_two_way_inference_index" in blob) and (
            not isinstance(two_way_idx, str)
            or two_way_idx not in _NORTHSET_TWO_WAY_INFERENCE_INDEX_ALLOWED
        ):
            errs.append("sweep_two_way_inference_index_invalid")
    return errs


def candle_join_coverage_and_chain_honesty_errors(blob: object) -> list[str]:
    """Soft-verify candle join_coverage ∈ (0, 1], chain ints, and ratio identity.

    - join_coverage ∈ (0, 1] when finite
    - n_scored ≤ n_fused ≤ n_bars when pairs present
    - when join_coverage + n_fused + n_bars (n_bars>0) all finite:
      join_coverage ≈ n_fused / n_bars (fuse stamps lit(fused.height/n_candles))

    Complements sizing honesty. Research diagnostic only; never live Sharpe.
    Off kyle invent / METRICS_*.
    """
    if not isinstance(blob, dict):
        return []
    if blob.get("family") != "candle_order_book":
        return []
    errs: list[str] = []
    if "join_coverage" in blob:
        try:
            j = float(blob.get("join_coverage"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append("join_coverage_non_numeric")
        else:
            if j == j and abs(j) != float("inf") and not (0.0 < j <= 1.0):
                errs.append("join_coverage_out_of_open_unit_interval")
            elif j == j and abs(j) == float("inf"):
                errs.append("join_coverage_non_finite")

    # chain ints when present
    def _as_int(key: str) -> int | None:
        if key not in blob:
            return None
        try:
            x = float(blob.get(key))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append(f"{key}_non_numeric")
            return None
        if x != x or abs(x) == float("inf") or x < 0 or x != int(x):
            errs.append(f"{key}_not_nonneg_int")
            return None
        return int(x)

    n_bars = _as_int("n_bars")
    n_fused = _as_int("n_fused")
    n_scored = _as_int("n_scored")
    if n_fused is not None and n_bars is not None and n_fused > n_bars:
        errs.append("n_fused_gt_n_bars")
    if n_scored is not None and n_fused is not None and n_scored > n_fused:
        errs.append("n_scored_gt_n_fused")

    # Ratio identity: join_coverage is stamped as fused.height / n_candle_rows
    if "join_coverage" in blob and n_bars is not None and n_fused is not None and n_bars > 0:
        try:
            j = float(blob.get("join_coverage"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            pass  # non-numeric already flagged above
        else:
            if j == j and abs(j) != float("inf"):
                expected = float(n_fused) / float(n_bars)
                if not math.isclose(j, expected, rel_tol=1e-9, abs_tol=1e-12):
                    errs.append("join_coverage_not_n_fused_over_n_bars")
    return errs


def northset_receipt_dgp_data_source_honesty_errors(blob: object) -> list[str]:
    """Soft-verify always-on northset ``dgp`` / ``book_dgp`` ↔ ``data_source``.

    Stamped on ``bench_northset`` (≠ nest ``kyle_ofi`` twins — never equate).
    When present on the top-level receipt:

    - ``dgp`` and ``book_dgp`` both present → must match
    - ``data_source == "SYNTHETIC"`` ⇒ present dgp fields are ``synthetic_lob``
    - ``book_dgp``/``dgp`` ``synthetic_lob`` ⇒ ``data_source`` is ``SYNTHETIC`` when set
    - non-synthetic dgp + ``data_source`` set ⇒ ``data_source != "SYNTHETIC"``;
      if ``book_source`` also set, ``data_source == book_source``

    Skip when all three of dgp/book_dgp/data_source absent. Research diagnostic
    only; never live Sharpe. Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    if blob.get("family") not in (None, "northset"):
        return []

    malformed: list[str] = []

    def _str(key: str) -> str | None:
        if key not in blob or blob.get(key) is None:
            return None
        val = blob.get(key)
        if not isinstance(val, str):
            # Present-but-non-string provenance must not read as absent.
            malformed.append(f"{key}_non_str")
            return None
        s = val.strip()
        return s if s else None

    book_dgp = _str("book_dgp")
    dgp = _str("dgp")
    data_source = _str("data_source")
    book_source = _str("book_source")
    if book_dgp is None and dgp is None and data_source is None and not malformed:
        return []

    errs: list[str] = list(malformed)
    if book_dgp is not None and dgp is not None and book_dgp != dgp:
        errs.append("northset_dgp_book_dgp_mismatch")

    synth_dgp: bool | None = None
    if book_dgp is not None:
        synth_dgp = book_dgp == "synthetic_lob"
    elif dgp is not None:
        synth_dgp = dgp == "synthetic_lob"

    if synth_dgp is True and data_source is not None and data_source != "SYNTHETIC":
        errs.append("northset_synthetic_dgp_data_source_not_SYNTHETIC")
    if data_source == "SYNTHETIC":
        if book_dgp is not None and book_dgp != "synthetic_lob":
            errs.append("northset_SYNTHETIC_data_source_book_dgp_not_synthetic_lob")
        if dgp is not None and dgp != "synthetic_lob":
            errs.append("northset_SYNTHETIC_data_source_dgp_not_synthetic_lob")
        if book_dgp is None and dgp is None:
            errs.append("northset_SYNTHETIC_data_source_missing_dgp")
    if synth_dgp is False and data_source is not None:
        if data_source == "SYNTHETIC":
            errs.append("northset_nonsynthetic_dgp_data_source_SYNTHETIC")
        elif book_source is not None and data_source != book_source:
            errs.append("northset_nonsynthetic_data_source_ne_book_source")
    return errs


def northset_receipt_string_enum_honesty_errors(blob: object) -> list[str]:
    """Soft-verify northset string enum companions when present.

    - ``claim`` → research_diagnostic_only
    - ``research_only is True`` when present
    - ``session_l2_identity_gate`` → enforced|skipped
    - ``data_source`` / ``label``: SYNTHETIC source ⇔ SYN* label when both present
    Research diagnostic only; never live Sharpe. Off kyle nest.
    """
    if not isinstance(blob, dict):
        return []
    if blob.get("family") not in (None, "northset"):
        return []
    errs: list[str] = []
    if "claim" in blob and blob.get("claim") != "research_diagnostic_only":
        errs.append("northset_claim_not_research_diagnostic_only")
    if "research_only" in blob and blob.get("research_only") is not True:
        errs.append("northset_research_only_missing_or_false")
    if "session_l2_identity_gate" in blob:
        gate = blob.get("session_l2_identity_gate")
        if gate not in ("enforced", "skipped"):
            errs.append("session_l2_identity_gate_invalid")
    src = blob.get("data_source")
    lab = blob.get("label")
    if isinstance(src, str) and isinstance(lab, str):
        synth_src = src.upper() == "SYNTHETIC"
        synth_lab = lab.upper().startswith("SYN")
        if synth_src != synth_lab:
            errs.append("northset_data_source_label_synthetic_mismatch")
    return errs


def northset_all_t_ic_finite_honesty_errors(blob: object) -> list[str]:
    """Soft-verify every ``*_t_ic`` is finite when present (signed OK).

    Catch-all companion to ``*_p_ic`` unit-interval. Skip best_feature_* /
    kyle_* / METRICS_*. Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    errs: list[str] = []
    for key, raw in blob.items():
        if not isinstance(key, str) or not key.endswith("_t_ic"):
            continue
        if key.startswith(("best_feature", "kyle_")) or "METRICS_" in key:
            continue
        if raw is None:
            continue
        try:
            x = float(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append(f"{key}_non_numeric")
            continue
        if x != x:
            continue
        if abs(x) == float("inf"):
            errs.append(f"{key}_non_finite_fail_closed")
    return errs


def northset_all_mean_ic_finite_honesty_errors(blob: object) -> list[str]:
    """Soft-verify every ``*_mean_ic`` is finite when present (signed OK)."""
    if not isinstance(blob, dict):
        return []
    errs: list[str] = []
    for key, raw in blob.items():
        if not isinstance(key, str) or not key.endswith("_mean_ic"):
            continue
        if key.startswith(("best_feature", "kyle_")) or "METRICS_" in key:
            continue
        if raw is None:
            continue
        try:
            x = float(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append(f"{key}_non_numeric")
            continue
        if x != x:
            continue
        if abs(x) == float("inf"):
            errs.append(f"{key}_non_finite_fail_closed")
    return errs


def northset_all_mean_rank_ic_unit_honesty_errors(blob: object) -> list[str]:
    """Soft-verify every ``*_mean_rank_ic`` ∈ [-1, 1] when finite."""
    if not isinstance(blob, dict):
        return []
    errs: list[str] = []
    for key, raw in blob.items():
        if not isinstance(key, str) or not key.endswith("_mean_rank_ic"):
            continue
        if key.startswith(("best_feature", "kyle_")) or "METRICS_" in key:
            continue
        if raw is None:
            continue
        try:
            x = float(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append(f"{key}_non_numeric")
            continue
        if x != x:
            continue
        if abs(x) == float("inf"):
            errs.append(f"{key}_non_finite_fail_closed")
        elif not (-1.0 <= x <= 1.0):
            errs.append(f"{key}_out_of_unit_interval")
    return errs


def northset_all_finite_rate_unit_honesty_errors(blob: object) -> list[str]:
    """Soft-verify every ``*_finite_rate`` ∈ [0, 1] when finite.

    Complements dedicated shape/structure helpers. Skip kyle_*/METRICS_*.
    """
    if not isinstance(blob, dict):
        return []
    errs: list[str] = []
    for key, raw in blob.items():
        if not isinstance(key, str) or not key.endswith("_finite_rate"):
            continue
        if key.startswith("kyle_") or "METRICS_" in key:
            continue
        if raw is None:
            continue
        try:
            x = float(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append(f"{key}_non_numeric")
            continue
        if x != x:
            continue
        if abs(x) == float("inf") or not (0.0 <= x <= 1.0):
            errs.append(f"{key}_out_of_unit_interval")
    return errs


def northset_all_floor_unit_honesty_errors(blob: object) -> list[str]:
    """Soft-verify every ``*_floor`` ∈ [0, 1] when finite and not None.

    Complements dedicated floors honesty. Skip kyle_*/METRICS_*.
    """
    if not isinstance(blob, dict):
        return []
    errs: list[str] = []
    for key, raw in blob.items():
        if not isinstance(key, str) or not key.endswith("_floor"):
            continue
        if key.startswith("kyle_") or "METRICS_" in key:
            continue
        if raw is None:
            continue
        if raw is None:
            continue
        try:
            x = float(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append(f"{key}_non_numeric")
            continue
        if x != x:
            continue
        if abs(x) == float("inf") or not (0.0 <= x <= 1.0):
            errs.append(f"{key}_out_of_unit_interval")
    return errs


def northset_all_p_ic_unit_interval_honesty_errors(blob: object) -> list[str]:
    """Soft-verify every ``*_p_ic`` key ∈ [0, 1] when finite.

    Catch-all residual for IC packs not yet given a dedicated helper.
    Never invent missing keys. Research diagnostic only; never live Sharpe.
    Off kyle_ofi / METRICS_* / best_feature_* (Sergeant lane).
    """
    if not isinstance(blob, dict):
        return []
    errs: list[str] = []
    for key, raw in blob.items():
        if not isinstance(key, str) or not key.endswith("_p_ic"):
            continue
        if key.startswith("best_feature"):
            continue  # Sergeant best_feature lane
        if raw is None:
            continue
        try:
            p = float(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append(f"{key}_non_numeric")
            continue
        if p != p:
            continue
        if abs(p) == float("inf"):
            errs.append(f"{key}_non_finite_fail_closed")
        elif not (0.0 <= p <= 1.0):
            errs.append(f"{key}_out_of_unit_interval")
    return errs


def northset_all_n_dates_nonneg_honesty_errors(blob: object) -> list[str]:
    """Soft-verify every ``*_n_dates`` companion ≥ 0 when finite.

    Pairs with ``*_p_ic`` catch-all. Skip best_feature_*. Research diagnostic only.
    """
    if not isinstance(blob, dict):
        return []
    errs: list[str] = []
    for key, raw in blob.items():
        if not isinstance(key, str) or not key.endswith("_n_dates"):
            continue
        if key.startswith("best_feature"):
            continue
        try:
            n = float(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append(f"{key}_non_numeric")
            continue
        if n != n:
            continue
        if abs(n) == float("inf"):
            errs.append(f"{key}_non_finite")
        elif n < 0.0:
            errs.append(f"{key}_negative")
    return errs


def northset_session_close_ic_packs_honesty_errors(blob: object) -> list[str]:
    """Soft-verify session_close_* IC packs (mid/spread/imbalance/depths/micro).

    Distinct from session snaps / n_session counts. Research diagnostic only.
    """
    if not isinstance(blob, dict):
        return []
    errs: list[str] = []
    stems = (
        "session_close_mid",
        "session_close_spread_bps",
        "session_close_imbalance",
        "session_close_bid_depth",
        "session_close_ask_depth",
        "session_close_micro_bps",
        "session_spread_bps_mean",
        "session_imbalance_mean",
    )
    for stem in stems:
        errs.extend(
            _ic_pack_honesty_errors(
                blob,
                mean_ic_key=f"{stem}_mean_ic",
                rank_ic_key=f"{stem}_mean_rank_ic",
                t_key=f"{stem}_t_ic",
                p_key=f"{stem}_p_ic",
                n_key=f"{stem}_n_dates",
            )
        )
    return errs


def northset_sweep_signed_ic_packs_honesty_errors(blob: object) -> list[str]:
    """Soft-verify sweep_*_signed IC packs; never equate reject/follow/depth."""
    if not isinstance(blob, dict):
        return []
    errs: list[str] = []
    for stem in (
        "sweep_reject_signed",
        "sweep_follow_signed",
        "sweep_depth_signed",
    ):
        errs.extend(
            _ic_pack_honesty_errors(
                blob,
                mean_ic_key=f"{stem}_mean_ic",
                rank_ic_key=f"{stem}_mean_rank_ic",
                t_key=f"{stem}_t_ic",
                p_key=f"{stem}_p_ic",
                n_key=f"{stem}_n_dates",
            )
        )
    return errs


def northset_volume_over_range_ic_packs_honesty_errors(blob: object) -> list[str]:
    """Soft-verify volume_over_range(_abs)_* IC packs."""
    if not isinstance(blob, dict):
        return []
    errs: list[str] = []
    for stem in ("volume_over_range", "volume_over_range_abs"):
        errs.extend(
            _ic_pack_honesty_errors(
                blob,
                mean_ic_key=f"{stem}_mean_ic",
                rank_ic_key=f"{stem}_mean_rank_ic",
                t_key=f"{stem}_t_ic",
                p_key=f"{stem}_p_ic",
                n_key=f"{stem}_n_dates",
            )
        )
    return errs


def northset_amihud_ic_pack_honesty_errors(blob: object) -> list[str]:
    """Soft-verify amihud_* IC pack; never equate to amihud_mean."""
    if not isinstance(blob, dict):
        return []
    return _ic_pack_honesty_errors(
        blob,
        mean_ic_key="amihud_mean_ic",
        rank_ic_key="amihud_mean_rank_ic",
        t_key="amihud_t_ic",
        p_key="amihud_p_ic",
        n_key="amihud_n_dates",
    ) + _ic_pack_honesty_errors(
        blob,
        mean_ic_key="amihud_abs_mean_ic",
        rank_ic_key="amihud_abs_mean_rank_ic",
        t_key="amihud_abs_t_ic",
        p_key="amihud_abs_p_ic",
        n_key="amihud_abs_n_dates",
    )


def amihud_mean_ic_vs_amihud_abs_mean_ic_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate ``amihud_mean_ic`` with ``amihud_abs_mean_ic``.

    Sibling Amihud discovery ICs — signed Amihud path ≠ abs Amihud path.
    When both keys are present:

    - Distinct key identity
    - Companion packs stay distinct (``amihud_p_ic`` ≠ ``amihud_abs_p_ic``, etc.)
    - Key-scoped finite probes
    - Numeric equality on clean synth is allowed

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off CoS spread-IC; off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_mean = "amihud_mean_ic"
    k_abs = "amihud_abs_mean_ic"
    if k_mean not in blob or k_abs not in blob:
        return []

    errs: list[str] = []
    if k_mean == k_abs:
        errs.append("amihud_mean_ic_amihud_abs_mean_ic_keys_collapsed")

    companions = (
        ("amihud_p_ic", "amihud_abs_p_ic"),
        ("amihud_t_ic", "amihud_abs_t_ic"),
        ("amihud_n_dates", "amihud_abs_n_dates"),
    )
    for a, b in companions:
        if a == b:
            errs.append(f"amihud_companion_keys_collapsed_{a}")

    if not _finite_scalar({k_mean: 0.1}.get(k_mean)):
        errs.append("amihud_mean_ic_finite_probe_broken")
    if _finite_scalar({k_abs: 0.1}.get(k_mean)):
        errs.append("amihud_abs_mean_ic_incorrectly_counts_as_finite_amihud_mean_ic")
    if not _finite_scalar({k_abs: 0.1}.get(k_abs)):
        errs.append("amihud_abs_mean_ic_finite_probe_broken")
    if _finite_scalar({k_mean: 0.1}.get(k_abs)):
        errs.append("amihud_mean_ic_incorrectly_counts_as_finite_amihud_abs_mean_ic")
    return errs


def northset_imbalance_depth_ic_pack_honesty_errors(blob: object) -> list[str]:
    """Soft-verify imbalance_depth_* IC pack; never equate to mean_depth_imbalance."""
    if not isinstance(blob, dict):
        return []
    return _ic_pack_honesty_errors(
        blob,
        mean_ic_key="imbalance_depth_mean_ic",
        rank_ic_key="imbalance_depth_mean_rank_ic",
        t_key="imbalance_depth_t_ic",
        p_key="imbalance_depth_p_ic",
        n_key="imbalance_depth_n_dates",
    )


def northset_bid_log_size_slope_ic_pack_honesty_errors(blob: object) -> list[str]:
    """Soft-verify bid_log_size_slope_* IC pack."""
    if not isinstance(blob, dict):
        return []
    return _ic_pack_honesty_errors(
        blob,
        mean_ic_key="bid_log_size_slope_mean_ic",
        rank_ic_key="bid_log_size_slope_mean_rank_ic",
        t_key="bid_log_size_slope_t_ic",
        p_key="bid_log_size_slope_p_ic",
        n_key="bid_log_size_slope_n_dates",
    )


def northset_candle_body_ret_ic_pack_honesty_errors(blob: object) -> list[str]:
    """Soft-verify candle_body_ret_* IC pack on northset."""
    if not isinstance(blob, dict):
        return []
    return _ic_pack_honesty_errors(
        blob,
        mean_ic_key="candle_body_ret_mean_ic",
        rank_ic_key="candle_body_ret_mean_rank_ic",
        t_key="candle_body_ret_t_ic",
        p_key="candle_body_ret_p_ic",
        n_key="candle_body_ret_n_dates",
    )


def northset_wick_skew_ic_pack_honesty_errors(blob: object) -> list[str]:
    """Soft-verify wick_skew_* IC pack on northset (finite mean/rank/t; p∈[0,1]; n≥0)."""
    if not isinstance(blob, dict):
        return []
    return _ic_pack_honesty_errors(
        blob,
        mean_ic_key="wick_skew_mean_ic",
        rank_ic_key="wick_skew_mean_rank_ic",
        t_key="wick_skew_t_ic",
        p_key="wick_skew_p_ic",
        n_key="wick_skew_n_dates",
    )


def northset_session_book_vpin_ic_pack_honesty_errors(blob: object) -> list[str]:
    """Soft-verify session_book_vpin_* IC pack; ≠ daily vpin_* IC."""
    if not isinstance(blob, dict):
        return []
    return _ic_pack_honesty_errors(
        blob,
        mean_ic_key="session_book_vpin_mean_ic",
        rank_ic_key="session_book_vpin_mean_rank_ic",
        t_key="session_book_vpin_t_ic",
        p_key="session_book_vpin_p_ic",
        n_key="session_book_vpin_n_dates",
    )


def candle_feature_cols_ic_implies_mean_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: every scored FEATURE_COLS IC has a mean_* companion when family is candle.

    Uses ``mean_<col>`` (and ``mean_depth_imbalance`` for imbalance_depth).
    Bounds: concentration/tob/queue_priority/depth_abs/mwb/spread_over_mid ∈[0,1];
    imbalance_* ∈[-1,1]; others finite-only. Never equate IC to mean.
    """
    if not isinstance(blob, dict):
        return []
    if blob.get("family") != "candle_order_book":
        return []
    try:
        from quant_fund.microstructure.bench import FEATURE_COLS
    except Exception:
        return []
    unit01 = {
        "depth_imbalance_abs",
        "spread_over_mid",
        "bid_size_concentration_top",
        "ask_size_concentration_top",
        "queue_priority_proxy",
        "ask_queue_priority_proxy",
        "tob_size_share",
        "microprice_weight_balance",
        "candle_body_frac",
        "candle_range_frac",
    }
    unit_signed = {
        "imbalance_top",
        "imbalance_depth",
        "queue_imbalance",
        "notional_imbalance",
        "candle_direction",
        "candle_dir_x_imbalance",
    }
    errs: list[str] = []
    for col in FEATURE_COLS:
        ic_key = f"ic_{col}"
        if ic_key not in blob:
            continue
        mean_key = "mean_depth_imbalance" if col == "imbalance_depth" else f"mean_{col}"
        if mean_key not in blob:
            errs.append(f"{mean_key}_missing_while_{ic_key}_scored")
            continue
        try:
            m = float(blob.get(mean_key))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append(f"{mean_key}_non_numeric")
            continue
        if m != m:
            errs.append(f"{mean_key}_nan_while_{ic_key}_scored")
            continue
        if abs(m) == float("inf"):
            errs.append(f"{mean_key}_non_finite")
            continue
        if col in unit01 and not (0.0 <= m <= 1.0) or col in unit_signed and not (-1.0 <= m <= 1.0):
            errs.append(f"{mean_key}_out_of_unit_interval")
    return errs


def northset_imbalance_top_ic_pack_honesty_errors(blob: object) -> list[str]:
    """Soft-verify imbalance_top_* IC pack (p∈[0,1], n_dates≥0, ICs finite).

    Never equate to ``mean_imbalance_top``. Research diagnostic only.
    """
    if not isinstance(blob, dict):
        return []
    return _ic_pack_honesty_errors(
        blob,
        mean_ic_key="imbalance_top_mean_ic",
        rank_ic_key="imbalance_top_mean_rank_ic",
        t_key="imbalance_top_t_ic",
        p_key="imbalance_top_p_ic",
        n_key="imbalance_top_n_dates",
    )


def northset_product_stamp_honesty_errors(blob: object) -> list[str]:
    """Soft-verify stamped ``product`` on northset receipts.

    ``bench_northset`` stamps ``product: "Northset"``. When ``product`` is present
    and family is ``northset`` (or family absent): value must be the nonempty string
    ``Northset``. Skip other families / absent key. Research diagnostic only; never
    live Sharpe. Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    if "product" not in blob:
        return []
    fam = blob.get("family")
    if fam is not None and fam != "northset":
        return []
    val = blob.get("product")
    if not isinstance(val, str) or not val:
        return ["northset_product_not_nonempty_str"]
    if val != "Northset":
        return ["northset_product_unexpected_token"]
    return []


def northset_impact_proxy_warning_honesty_errors(blob: object) -> list[str]:
    """Soft-verify stamped ``impact_proxy_warning`` enum on northset receipts.

    DATA_CONTRACTS / benches: book-size / TOB proxies are not signed trade prints.
    When present on ``family == northset`` (or family absent with the key stamped):
    value must be the nonempty string
    ``depth_or_ofi_proxy_not_signed_trade_flow``. Skip other families / absent key.
    Research diagnostic only; never live Sharpe. Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    if "impact_proxy_warning" not in blob:
        return []
    fam = blob.get("family")
    if fam is not None and fam != "northset":
        return []
    val = blob.get("impact_proxy_warning")
    expected = "depth_or_ofi_proxy_not_signed_trade_flow"
    if not isinstance(val, str) or not val:
        return ["impact_proxy_warning_not_nonempty_str"]
    if val != expected:
        return ["impact_proxy_warning_unexpected_token"]
    return []


def close_location_value_clv_alias_identity_honesty_errors(blob: object) -> list[str]:
    """Soft-verify ``close_location_value_*`` IC aliases match ``clv_*`` when both stamped.

    DATA_CONTRACTS: ``clv_p_ic`` / ``clv_t_ic`` are receipt aliases of
    ``close_location_value_p_ic`` / ``close_location_value_t_ic`` (H30 binds to
    ``clv_p_ic``). When both sides of a pair are finite they must match; keys stay
    distinct. Skip absent / non-finite sides. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    errs: list[str] = []
    pairs = (
        ("close_location_value_p_ic", "clv_p_ic", "close_location_value_p_ic_clv_p_ic_mismatch"),
        ("close_location_value_t_ic", "clv_t_ic", "close_location_value_t_ic_clv_t_ic_mismatch"),
    )
    for long_k, short_k, err_token in pairs:
        if long_k == short_k:
            errs.append("close_location_value_clv_alias_keys_collapsed")
            continue
        if long_k not in blob or short_k not in blob:
            continue
        if not _finite_scalar(blob.get(long_k)) or not _finite_scalar(blob.get(short_k)):
            continue
        a = float(blob[long_k])  # type: ignore[arg-type]
        b = float(blob[short_k])  # type: ignore[arg-type]
        if not math.isclose(a, b, rel_tol=0.0, abs_tol=1e-12):
            errs.append(err_token)
    return errs


def northset_clv_ic_pack_honesty_errors(blob: object) -> list[str]:
    """Soft-verify clv_* IC pack: p∈[0,1], t finite. Never equate to CLV mean."""
    if not isinstance(blob, dict):
        return []
    return _ic_pack_honesty_errors(
        blob,
        mean_ic_key="clv_mean_ic",
        rank_ic_key="clv_mean_rank_ic",
        t_key="clv_t_ic",
        p_key="clv_p_ic",
        n_key="clv_n_dates",
    )


def northset_microprice_bps_ic_pack_honesty_errors(blob: object) -> list[str]:
    """Soft-verify microprice_minus_mid_bps_* IC pack; never equate to microprice_p_ic alias alone."""
    if not isinstance(blob, dict):
        return []
    return _ic_pack_honesty_errors(
        blob,
        mean_ic_key="microprice_minus_mid_bps_mean_ic",
        rank_ic_key="microprice_minus_mid_bps_mean_rank_ic",
        t_key="microprice_minus_mid_bps_t_ic",
        p_key="microprice_minus_mid_bps_p_ic",
        n_key="microprice_minus_mid_bps_n_dates",
    )


def candle_all_finite_rate_prefix_honesty_errors(blob: object) -> list[str]:
    """Soft-verify every candle ``finite_rate_*`` key ∈ [0, 1] when finite.

    Candle stamps use the ``finite_rate_<feature>`` prefix (≠ northset
    ``*_finite_rate`` suffix catch-all). NaN skip; ±inf / OOB fail-closed.
    Research diagnostic only; off kyle invent.
    """
    if not isinstance(blob, dict):
        return []
    if blob.get("family") not in (None, "candle_order_book"):
        return []
    errs: list[str] = []
    for key, raw in blob.items():
        if not isinstance(key, str) or not key.startswith("finite_rate_"):
            continue
        if raw is None:
            continue
        try:
            x = float(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append(f"{key}_non_numeric")
            continue
        if x != x:
            continue
        if abs(x) == float("inf") or not (0.0 <= x <= 1.0):
            errs.append(f"{key}_out_of_unit_interval")
    return errs


def candle_ofi_qp_slope_ic_implies_mean_honesty_errors(blob: object) -> list[str]:
    """Soft-verify ofi / queue_priority / size-slope IC⇒mean on candle_order_book."""
    if not isinstance(blob, dict):
        return []
    if blob.get("family") != "candle_order_book":
        return []
    errs: list[str] = []
    specs = (
        ("ic_ofi", "mean_ofi", None, None),  # signed unbounded
        ("ic_queue_priority_proxy", "mean_queue_priority_proxy", 0.0, 1.0),
        ("ic_ask_queue_priority_proxy", "mean_ask_queue_priority_proxy", 0.0, 1.0),
        ("ic_bid_log_size_slope", "mean_bid_log_size_slope", None, None),
        ("ic_ask_log_size_slope", "mean_ask_log_size_slope", None, None),
        ("ic_microprice_minus_mid", "mean_microprice_minus_mid", None, None),
        ("ic_microprice_minus_mid_bps", "mean_microprice_minus_mid_bps", None, None),
        ("ic_bid_mean_log_tick_spacing", "mean_bid_mean_log_tick_spacing", None, None),
        ("ic_ask_mean_log_tick_spacing", "mean_ask_mean_log_tick_spacing", None, None),
        ("ic_bid_log_price_slope", "mean_bid_log_price_slope", None, None),
        ("ic_ask_log_price_slope", "mean_ask_log_price_slope", None, None),
    )
    for ic_key, mean_key, lo, hi in specs:
        if ic_key not in blob:
            continue
        if mean_key not in blob:
            errs.append(f"{mean_key}_missing_while_{ic_key}_scored")
            continue
        try:
            m = float(blob.get(mean_key))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append(f"{mean_key}_non_numeric")
            continue
        if m != m:
            errs.append(f"{mean_key}_nan_while_{ic_key}_scored")
        elif abs(m) == float("inf"):
            errs.append(f"{mean_key}_non_finite")
        elif lo is not None and hi is not None and not (lo <= m <= hi):
            errs.append(f"{mean_key}_out_of_unit_interval")
    return errs


def candle_ofi_and_queue_imbalance_means_honesty_errors(blob: object) -> list[str]:
    """Soft-verify candle fuse means when stamped (not only when IC scored).

    - ``mean_ofi`` finite when present (signed unbounded)
    - ``mean_queue_imbalance`` ∈ [-1, 1] when finite
    NaN skip. Complements IC⇒mean helpers. Research diagnostic only; off kyle invent.
    """
    if not isinstance(blob, dict):
        return []
    if blob.get("family") not in (None, "candle_order_book"):
        return []
    errs: list[str] = []
    if "mean_ofi" in blob:
        try:
            x = float(blob.get("mean_ofi"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append("mean_ofi_non_numeric")
        else:
            if x == x and abs(x) == float("inf"):
                errs.append("mean_ofi_non_finite")
    if "mean_queue_imbalance" in blob:
        try:
            q = float(blob.get("mean_queue_imbalance"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append("mean_queue_imbalance_non_numeric")
        else:
            if q == q and (abs(q) == float("inf") or not (-1.0 <= q <= 1.0)):
                errs.append("mean_queue_imbalance_out_of_signed_unit")
    return errs


def candle_feature_ofi_finite_honesty_errors(blob: object) -> list[str]:
    """Soft-verify FEATURE_COLS ``ofi`` mean finite when stamped or IC-scored.

    Delegates stamped-mean check to :func:`candle_ofi_and_queue_imbalance_means_honesty_errors`
    and IC⇒mean finite to :func:`candle_feature_cols_ic_implies_mean_honesty_errors`
    (ofi is unbounded). Research diagnostic only.
    """
    if not isinstance(blob, dict):
        return []
    errs: list[str] = []
    errs.extend(candle_ofi_and_queue_imbalance_means_honesty_errors(blob))
    # only keep ofi-related
    return [e for e in errs if "ofi" in e]


def imbalance_top_mean_ic_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: imbalance_top_mean_ic finite when present (signed OK).

    ±inf fail-closed; NaN/absent skip. Companion to H22 discovery path.
    Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    if "imbalance_top_mean_ic" not in blob:
        return []
    try:
        x = float(blob.get("imbalance_top_mean_ic"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return ["imbalance_top_mean_ic_non_numeric"]
    if x != x:
        return []
    if abs(x) == float("inf"):
        return ["imbalance_top_mean_ic_non_finite_fail_closed"]
    return []


_NORTHSET_PRICE_BASIS_ALLOWED = frozenset({"split_adjusted", "raw_fixture_opt_out"})
_NORTHSET_RETURN_BASIS_ALLOWED = frozenset(
    {"total_return", "split_adjusted", "raw_fixture_opt_out"}
)


def northset_family_book_source_honesty_errors(blob: object) -> list[str]:
    """Soft-verify always-on ``family`` / ``book_source`` / ``label`` stamps when present.

    - When ``family`` is present: must be ``"northset"`` (skip other families so
      candle blobs are not dual-flagged — callers pass the northset family blob)
    - When ``book_source`` is present: nonempty string
    - When ``label`` is present: nonempty string (empty label can pass the
      SYNTHETIC↔SYN* match when ``data_source`` is also non-SYN)

    Skip when ``family`` is ``candle_order_book`` (wrong surface). Research
    diagnostic only; never live Sharpe. Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    fam = blob.get("family") if "family" in blob else None
    if fam == "candle_order_book":
        return []

    errs: list[str] = []
    if "family" in blob and blob.get("family") is not None and blob.get("family") != "northset":
        errs.append("northset_family_invalid")

    if "book_source" in blob and blob.get("book_source") is not None:
        bs = blob.get("book_source")
        if not isinstance(bs, str) or not bs.strip():
            errs.append("northset_book_source_empty_or_not_str")

    if "label" in blob and blob.get("label") is not None:
        lab = blob.get("label")
        if not isinstance(lab, str) or not lab.strip():
            errs.append("northset_label_empty_or_not_str")
    return errs


def northset_price_return_basis_honesty_errors(blob: object) -> list[str]:
    """Soft-verify always-on ``price_basis`` / ``return_basis`` stamps when present.

    Contract (DATA_CONTRACTS price_basis/return_basis matrix):
    - ``price_basis`` ∈ {split_adjusted, raw_fixture_opt_out}
    - ``return_basis`` ∈ {total_return, split_adjusted, raw_fixture_opt_out}
    - both ``raw_fixture_opt_out`` ⇒ must match each other
    - ``price_basis == split_adjusted`` ⇒ ``return_basis`` ∈ {total_return, split_adjusted}
      when return_basis present

    Absent keys skip. Research diagnostic only; never live Sharpe. Off nest /
    kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    if blob.get("family") not in (None, "northset"):
        return []

    errs: list[str] = []
    pb = blob.get("price_basis") if "price_basis" in blob else None
    rb = blob.get("return_basis") if "return_basis" in blob else None

    if pb is not None and (not isinstance(pb, str) or pb not in _NORTHSET_PRICE_BASIS_ALLOWED):
        errs.append("northset_price_basis_invalid")
    if rb is not None and (not isinstance(rb, str) or rb not in _NORTHSET_RETURN_BASIS_ALLOWED):
        errs.append("northset_return_basis_invalid")

    if isinstance(pb, str) and isinstance(rb, str):
        if pb == "raw_fixture_opt_out" and rb != "raw_fixture_opt_out":
            errs.append("northset_raw_fixture_price_return_basis_mismatch")
        if pb == "split_adjusted" and rb not in ("total_return", "split_adjusted"):
            errs.append("northset_split_adjusted_return_basis_invalid")
    return errs


def northset_depth_honesty_errors(blob: object) -> list[str]:
    """Soft-verify always-on northset ``depth`` stamp when present.

    LOB levels used by attach / shape floors — must be finite integer ≥ 1.
    Candle has a parallel check in ``candle_order_book_sizing_honesty_errors``;
    never equate the two surfaces. Skip absent/NaN. Research diagnostic only;
    never live Sharpe. Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    if blob.get("family") not in (None, "northset"):
        return []
    if "depth" not in blob or blob.get("depth") is None:
        return []
    try:
        d = float(blob.get("depth"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return ["northset_depth_non_numeric"]
    if d != d:
        return []
    if abs(d) == float("inf"):
        return ["northset_depth_non_finite"]
    if d < 1.0 or abs(d - int(d)) > 1e-9:
        return ["northset_depth_lt_one_or_not_int"]
    return []


def northset_component_sources_honesty_errors(blob: object) -> list[str]:
    """Soft-verify always-on ``component_sources`` receipt map when present.

    Contract (DATA_CONTRACTS evidence matrix / ``bench_northset`` stamp):
    - value is a ``dict``
    - required keys: ``bars``, ``book``, ``session_candles``, ``session_book``
      with nonempty string values
    - ``session_candles`` == ``synthetic_reconstruction``
    - when ``use_session_l2`` is bool: ``session_book`` is
      ``synthetic_reconstruction`` if True else ``disabled``
    - when ``book_source`` is a nonempty str: ``book`` must equal it

    Absent ``component_sources`` → skip. Nest has no component map — never equate.
    Research diagnostic only; never live Sharpe. Off kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    if "component_sources" not in blob:
        return []
    cs = blob.get("component_sources")
    if not isinstance(cs, dict):
        return ["component_sources_not_dict"]

    errs: list[str] = []
    required = ("bars", "book", "session_candles", "session_book")
    for key in required:
        if key not in cs:
            errs.append(f"component_sources_missing_{key}")
            continue
        val = cs.get(key)
        if not isinstance(val, str) or not val.strip():
            errs.append(f"component_sources_{key}_empty_or_not_str")

    if errs:
        return errs

    if cs.get("session_candles") != "synthetic_reconstruction":
        errs.append("component_sources_session_candles_not_synthetic_reconstruction")

    use = blob.get("use_session_l2")
    if type(use) is bool:
        expect_book = "synthetic_reconstruction" if use else "disabled"
        if cs.get("session_book") != expect_book:
            errs.append("component_sources_session_book_mismatch_use_session_l2")

    book_source = blob.get("book_source")
    if isinstance(book_source, str) and book_source.strip() and cs.get("book") != book_source:
        errs.append("component_sources_book_ne_book_source")

    return errs


def northset_use_session_l2_gate_consistency_errors(blob: object) -> list[str]:
    """Soft-verify ``use_session_l2`` ↔ ``session_l2_identity_gate`` when both present.

    Stamp contract from ``bench_northset``:
    - ``use_session_l2 is True`` ⇔ gate ``"enforced"``
    - ``use_session_l2 is False`` ⇔ gate ``"skipped"``

    One present without the other → skip (pair incomplete). Invalid gate string
    still caught by floor/string-enum helpers. Research diagnostic only; never
    live Sharpe. Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    if "use_session_l2" not in blob or "session_l2_identity_gate" not in blob:
        return []
    use = blob.get("use_session_l2")
    gate = blob.get("session_l2_identity_gate")
    if type(use) is not bool:
        return []  # bool-flags helper owns type
    if gate not in ("enforced", "skipped"):
        return []  # enum helper owns invalid gate
    if use and gate != "enforced":
        return ["use_session_l2_true_gate_not_enforced"]
    if (not use) and gate != "skipped":
        return ["use_session_l2_false_gate_not_skipped"]
    return []


def northset_receipt_bool_flags_honesty_errors(blob: object) -> list[str]:
    """Soft-verify stamped northset receipt bool flags are real ``bool``.

    Covers keys stamped by ``bench_northset`` that are not already checked by
    ``book_hypothesis_eligible_honesty_errors``:

    - ``metrics_required_finite_ok``
    - ``shape_columns_ensured``
    - ``use_session_l2``
    - ``research_only``
    - ``include_kyle_ofi``
    - ``sweep_follow_control_sample_adequate``
    - ``sweep_reject_control_sample_adequate``

    When present, each must be ``type is bool`` (not int/str/None). True/False
    both valid. Absent skip. Top-level always-on only (≠ nest). Research
    diagnostic only; never live Sharpe. Off kyle_ofi overwrite / METRICS_*.
    """
    if not isinstance(blob, dict):
        return []
    errs: list[str] = []
    for key in (
        "metrics_required_finite_ok",
        "shape_columns_ensured",
        "use_session_l2",
        "research_only",
        "include_kyle_ofi",
        "sweep_follow_control_sample_adequate",
        "sweep_reject_control_sample_adequate",
    ):
        if key not in blob:
            continue
        if type(blob.get(key)) is not bool:
            errs.append(f"{key}_not_bool")
    return errs


def book_hypothesis_eligible_honesty_errors(blob: object) -> list[str]:
    """Soft-verify book_hypothesis_eligible / session_book_hypothesis_eligible bools.

    When present, each key must be a real ``bool`` (not int/str/None). Content
    True/False is always valid — eligibility is a receipt flag, not a score.
    Absent keys skipped. Research diagnostic only; never live Sharpe.
    Does not touch kyle_* / book_metrics METRICS_* keys.
    """
    if not isinstance(blob, dict):
        return []
    errs: list[str] = []
    for key in ("book_hypothesis_eligible", "session_book_hypothesis_eligible"):
        if key not in blob:
            continue
        val = blob.get(key)
        if type(val) is not bool:
            errs.append(f"{key}_not_bool")
    return errs


def northset_lag_corr_and_sweep_count_honesty_errors(blob: object) -> list[str]:
    """Soft-verify mid_lag1_corr, ofi_lag1_corr, n_sweep_high when present.

    - mid_lag1_corr ∈ [-1, 1] when finite (panel mean of per-name lag-1 corr)
    - ofi_lag1_corr ∈ [-1, 1] when finite
    - n_sweep_high ≥ 0 when present (integer count; non-numeric / ±inf dishonest)
    - mid_lag1_n_securities / ofi_lag1_n_securities ≥ 0 when finite
    NaN skipped for corr keys. Research diagnostic only; never live Sharpe.
    Does not touch kyle_* / book_metrics METRICS_* keys.
    """
    if not isinstance(blob, dict):
        return []
    errs: list[str] = []

    for key in ("mid_lag1_corr", "ofi_lag1_corr"):
        if key not in blob:
            continue
        try:
            x = float(blob.get(key))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append(f"{key}_non_numeric")
            continue
        if x != x:
            continue
        if abs(x) == float("inf"):
            errs.append(f"{key}_non_finite")
            continue
        if not (-1.0 <= x <= 1.0):
            errs.append(f"{key}_out_of_unit_interval")

    if "n_sweep_high" in blob:
        try:
            n = float(blob.get("n_sweep_high"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append("n_sweep_high_non_numeric")
        else:
            if n != n or abs(n) == float("inf"):
                errs.append("n_sweep_high_non_finite")
            elif n < 0.0:
                errs.append("n_sweep_high_negative")

    for key in ("mid_lag1_n_securities", "ofi_lag1_n_securities"):
        if key not in blob:
            continue
        try:
            n = float(blob.get(key))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append(f"{key}_non_numeric")
            continue
        if n != n:
            continue
        if abs(n) == float("inf"):
            errs.append(f"{key}_non_finite")
        elif n < 0.0:
            errs.append(f"{key}_negative")
        # When matching corr is finite, n_securities should be present (already is)
    return errs


def mid_lag1_corr_vs_ofi_lag1_corr_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate ``mid_lag1_corr`` with ``ofi_lag1_corr``.

    DATA_CONTRACTS: same panel lag-1 estimator shape, different series
    (mid path AR ≠ OFI path AR). When both keys are present:

    - Distinct key identity
    - Companion n-securities keys stay distinct when both stamped
    - Key-scoped finite probes (gap-style: mid-only blob must not count as ofi)
    - Numeric equality on clean synth is allowed

    Skip when either corr key absent. Research diagnostic only; never live Sharpe.
    Always-on surface — not nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_mid = "mid_lag1_corr"
    k_ofi = "ofi_lag1_corr"
    if k_mid not in blob or k_ofi not in blob:
        return []

    errs: list[str] = []
    if k_mid == k_ofi:
        errs.append("mid_ofi_lag1_corr_keys_collapsed")

    k_mid_n = "mid_lag1_n_securities"
    k_ofi_n = "ofi_lag1_n_securities"
    if k_mid_n in blob and k_ofi_n in blob and k_mid_n == k_ofi_n:
        errs.append("mid_ofi_lag1_n_securities_keys_collapsed")

    if not _finite_scalar({k_mid: 0.5}.get(k_mid)):
        errs.append("mid_lag1_corr_finite_probe_broken")
    if _finite_scalar({k_ofi: 0.5}.get(k_mid)):
        errs.append("ofi_lag1_corr_incorrectly_counts_as_finite_mid_lag1")
    if not _finite_scalar({k_ofi: 0.5}.get(k_ofi)):
        errs.append("ofi_lag1_corr_finite_probe_broken")
    if _finite_scalar({k_mid: 0.5}.get(k_ofi)):
        errs.append("mid_lag1_corr_incorrectly_counts_as_finite_ofi_lag1")
    return errs


def gap_finite_rate_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: gap_finite_rate ∈ [0, 1] when finite.

    Northset overnight gap finite fraction (identities.gap_finite_rate).
    NaN (empty bars) skipped. Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    val = blob.get("gap_finite_rate")
    if val is None:
        return []
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        # Bools/strings must not coerce into a passing rate.
        return ["gap_finite_rate_non_numeric"]
    x = float(val)
    if x != x:  # NaN
        return []
    if not (0.0 <= x <= 1.0):
        return ["gap_finite_rate_out_of_unit_interval"]
    return []


def vpin_mean_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: vpin_mean ∈ [0, 1] when finite.

    Daily fused VPIN mean on northset receipt (not session_book_vpin_mean).
    NaN skipped. Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    val = blob.get("vpin_mean")
    try:
        x = float(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return []
    if x != x:
        return []
    if not (0.0 <= x <= 1.0):
        return ["vpin_mean_out_of_unit_interval"]
    return []


def session_bulk_vpin_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: session_bulk_vpin ∈ [0, 1] when finite.

    Candle signed-volume VPIN scalar (session_vpin) — **not** H32 ``vpin_mean``
    and **not** H43 ``session_book_vpin_mean`` (DATA_CONTRACTS triad).
    NaN/absent skip. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    val = blob.get("session_bulk_vpin")
    try:
        x = float(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return []
    if x != x:
        return []
    if abs(x) == float("inf") or not (0.0 <= x <= 1.0):
        return ["session_bulk_vpin_out_of_unit_interval"]
    return []


def northset_spread_means_honesty_errors(blob: object) -> list[str]:
    """Soft-verify northset mean_quoted/effective/spread_bps when finite.

    - each ≥ 0 (and not ±inf), including mean_half_spread[_bps]
    - half ≈ ½ quoted; half_bps ≈ ½ spread_bps
    - do not equate quoted ≈ effective on this fuse (distinct columns)
    NaN keys skipped. Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    errs: list[str] = []

    def _finite(key: str) -> float | None:
        try:
            x = float(blob.get(key))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
        if x != x:
            return None
        if abs(x) == float("inf"):
            errs.append(f"{key}_non_finite")
            return None
        return x

    quoted = _finite("mean_quoted_spread")
    effective = _finite("mean_effective_spread")
    spread_bps = _finite("mean_spread_bps")
    half = _finite("mean_half_spread")
    half_bps = _finite("mean_half_spread_bps")
    for key, val in (
        ("mean_quoted_spread", quoted),
        ("mean_effective_spread", effective),
        ("mean_spread_bps", spread_bps),
        ("mean_half_spread", half),
        ("mean_half_spread_bps", half_bps),
    ):
        if val is not None and val < 0.0:
            errs.append(f"{key}_negative")
    # Post-collision-fix contract: effective_spread is the ask−bid alias, so a
    # finite quoted/effective divergence is a dishonesty signal (book effective
    # must not track the candle close−mid diagnostic).
    if (
        quoted is not None
        and effective is not None
        and not math.isclose(quoted, effective, rel_tol=1e-6, abs_tol=1e-12)
    ):
        errs.append("mean_quoted_effective_spread_mismatch")
    if quoted is not None and half is not None:
        tol = 1e-6 + 1e-6 * abs(quoted)
        if abs(half - 0.5 * quoted) > tol:
            errs.append("mean_half_spread_not_half_quoted")
    if spread_bps is not None and half_bps is not None:
        tol = 1e-6 + 1e-6 * abs(spread_bps)
        if abs(half_bps - 0.5 * spread_bps) > tol:
            errs.append("mean_half_spread_bps_not_half_spread_bps")
    return errs


def northset_session_l2_enforced_identity_rates_present_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: when session L2 identity gate is enforced, rates are present.

    If ``session_l2_identity_gate == "enforced"``, require the four session
    identity rate keys. Distinct from daily ``ohlc_identity_rate``.
    Research diagnostic only; never live Sharpe. Off kyle_ofi / METRICS_*.
    """
    if not isinstance(blob, dict):
        return []
    if blob.get("session_l2_identity_gate") != "enforced":
        return []
    errs: list[str] = []
    for key in (
        "session_ohlc_identity_rate",
        "session_reconstructs_daily_rate",
        "session_volume_conservation_rate",
        "session_chain_rate",
    ):
        if key not in blob:
            errs.append(f"{key}_missing_while_session_l2_identity_gate_enforced")
    return errs


def candle_notional_imbalance_ic_implies_mean_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: when notional IC is scored, mean_notional_imbalance ∈ [-1, 1].

    ``ic_notional_imbalance`` on candle_order_book ⇒ mean present and ∈ [-1, 1].
    Never equate IC to the mean. Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    if blob.get("family") != "candle_order_book":
        return []
    if "ic_notional_imbalance" not in blob:
        return []
    if "mean_notional_imbalance" not in blob:
        return ["mean_notional_imbalance_missing_while_notional_ic_scored"]
    try:
        m = float(blob.get("mean_notional_imbalance"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return ["mean_notional_imbalance_non_numeric"]
    if m != m:
        return ["mean_notional_imbalance_nan_while_notional_ic_scored"]
    if abs(m) == float("inf") or not (-1.0 <= m <= 1.0):
        return ["mean_notional_imbalance_out_of_unit_interval"]
    return []


def northset_session_identity_rates_honesty_errors(blob: object) -> list[str]:
    """Soft-verify session L2 identity rates ∈ [0, 1] when finite.

    Keys: session_ohlc_identity_rate, session_reconstructs_daily_rate,
    session_volume_conservation_rate, session_chain_rate. NaN skipped.
    Research diagnostic only;
    never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    errs: list[str] = []
    for key in (
        "session_ohlc_identity_rate",
        "session_reconstructs_daily_rate",
        "session_volume_conservation_rate",
        "session_chain_rate",
    ):
        try:
            x = float(blob.get(key))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if x != x:
            continue
        if not (0.0 <= x <= 1.0):
            errs.append(f"{key}_out_of_unit_interval")
    return errs


def northset_session_imbalance_mean_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: mean_session_imbalance_mean ∈ [-1, 1] when finite.

    Session-L2 path mean of ``session_imbalance_mean`` — not daily
    ``imbalance_top`` / ``queue_imbalance``. Skip if missing or non-finite.
    Research diagnostic only; never live Sharpe / promotion.
    """
    if not isinstance(blob, dict):
        return []
    val = blob.get("mean_session_imbalance_mean")
    try:
        x = float(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return []
    if x != x:  # NaN
        return []
    if not (-1.0 <= x <= 1.0):
        return ["mean_session_imbalance_mean_out_of_unit_interval"]
    return []


def northset_session_close_micro_bps_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: mean_session_close_micro_bps finite when present (not ±inf).

    Last-snap session micro — not daily ``microprice_minus_mid_bps`` mean.
    NaN (absent / session L2 off) is fine. Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    val = blob.get("mean_session_close_micro_bps")
    try:
        x = float(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return []
    if x != x:  # NaN
        return []
    if abs(x) == float("inf"):
        return ["mean_session_close_micro_bps_non_finite"]
    return []


def northset_session_close_depth_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: mean_session_close_bid/ask_depth ≥ 0 when finite.

    Last-snap depths — not daily ``bid_depth`` / ``ask_depth`` means.
    NaN (absent / session L2 off) is fine. Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    errors: list[str] = []
    for key in ("mean_session_close_bid_depth", "mean_session_close_ask_depth"):
        val = blob.get(key)
        try:
            x = float(val)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if x != x:  # NaN
            continue
        if x < 0.0 or abs(x) == float("inf"):
            errors.append(f"{key}_negative_or_non_finite")
    return errors


def northset_session_imbalance_std_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: mean_session_imbalance_std ≥ 0 when finite.

    Path dispersion — not path mean / last-snap close imbalance.
    NaN (absent / session L2 off) is fine. Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    val = blob.get("mean_session_imbalance_std")
    try:
        x = float(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return []
    if x != x:  # NaN
        return []
    if x < 0.0 or abs(x) == float("inf"):
        return ["mean_session_imbalance_std_negative_or_non_finite"]
    return []


def northset_session_close_imbalance_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: mean_session_close_imbalance ∈ [-1, 1] when finite.

    Last-snap close imbalance — not path mean / daily imbalance_top.
    NaN (absent / session L2 off) is fine. Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    val = blob.get("mean_session_close_imbalance")
    try:
        x = float(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return []
    if x != x:  # NaN
        return []
    if not (-1.0 <= x <= 1.0):
        return ["mean_session_close_imbalance_out_of_unit_interval"]
    return []


def northset_session_close_mid_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: mean_session_close_mid > 0 when finite.

    Last-snap mid — not daily mid/close. NaN skipped. Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    val = blob.get("mean_session_close_mid")
    try:
        x = float(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return []
    if x != x:  # NaN
        return []
    if not (x > 0.0) or abs(x) == float("inf"):
        return ["mean_session_close_mid_non_positive_or_non_finite"]
    return []


def northset_session_close_spread_bps_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: mean_session_close_spread_bps ≥ 0 when finite.

    Last-snap session close spread — not path ``mean_session_spread_bps_mean`` /
    daily ``mean_spread_bps``. Uses ≥0 (not >0): vendor_book_map can emit
    ``spread_bps=0`` when mid≤0 / locked book (``otherwise(0.0)``); synthetic
    clips to [1, 80] but honesty must not reject a real zero. NaN skipped.
    Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    val = blob.get("mean_session_close_spread_bps")
    try:
        x = float(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return []
    if x != x:  # NaN
        return []
    if x < 0.0 or abs(x) == float("inf"):
        return ["mean_session_close_spread_bps_negative_or_non_finite"]
    return []


def northset_session_spread_bps_mean_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: mean_session_spread_bps_mean ≥ 0 when finite.

    Session-L2 path mean of ``session_spread_bps_mean`` — not last-snap close /
    daily ``mean_spread_bps``. ≥0 matches book_metrics / vendor zero-spread.
    NaN skipped. Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    val = blob.get("mean_session_spread_bps_mean")
    try:
        x = float(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return []
    if x != x:  # NaN
        return []
    if x < 0.0 or abs(x) == float("inf"):
        return ["mean_session_spread_bps_mean_negative_or_non_finite"]
    return []


def northset_session_book_snaps_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: mean_session_book_snaps > 0 when finite.

    Session-L2 snap count mean — skip NaN (session L2 off). Research diagnostic
    only; never live Sharpe / promotion.
    """
    if not isinstance(blob, dict):
        return []
    val = blob.get("mean_session_book_snaps")
    if val is None:
        return []
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        # Present-but-malformed must not read as absent/NaN.
        return ["mean_session_book_snaps_non_numeric"]
    x = float(val)
    if x != x:  # NaN
        return []
    if not (x > 0.0) or abs(x) == float("inf"):
        return ["mean_session_book_snaps_non_positive_or_non_finite"]
    return []


def northset_session_book_snaps_n_session_consistency_errors(blob: object) -> list[str]:
    """Soft-verify mean_session_book_snaps vs n_session_book_rows / n_session_candles.

    Fail-closed when both sides of the session-L2 coverage pair are present and
    disagree:

    - Finite ``mean_session_book_snaps`` (path on) → ``n_session_book_rows`` must
      be present and ``> 0``; if ``n_session_candles`` is also present, rows must
      equal candles (1:1 session book vs session candle panel totals).
    - ``n_session_book_rows > 0`` with ``mean_session_book_snaps`` present → mean
      must be finite and ``> 0``.

    Skip when mean is NaN/absent and rows are 0/absent (session L2 off).
    Never equate ``mean_session_book_snaps`` to ``n_session_candles`` (per-parent
    mean ≠ panel row count). Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []

    has_mean = "mean_session_book_snaps" in blob
    has_rows = "n_session_book_rows" in blob and blob.get("n_session_book_rows") is not None
    has_candles = "n_session_candles" in blob and blob.get("n_session_candles") is not None

    mean_x: float | None = None
    if has_mean:
        raw_mean = blob.get("mean_session_book_snaps")
        if raw_mean is None:
            mean_x = None
        elif isinstance(raw_mean, bool) or not isinstance(raw_mean, (int, float)):
            # Present-but-malformed must not read as absent/NaN.
            return ["mean_session_book_snaps_non_numeric"]
        else:
            mean_x = float(raw_mean)

    # Parse through float first: int(float("inf")) raises OverflowError, which
    # a JSON receipt with Infinity counts would otherwise crash on.
    rows_n: int | None = None
    if has_rows:
        try:
            rows_f = float(blob.get("n_session_book_rows"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return ["n_session_book_rows_non_integer"]
        if rows_f != rows_f or abs(rows_f) == float("inf") or rows_f != int(rows_f):
            return ["n_session_book_rows_non_integer"]
        rows_n = int(rows_f)

    candles_n: int | None = None
    if has_candles:
        try:
            candles_f = float(blob.get("n_session_candles"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return ["n_session_candles_non_integer"]
        if candles_f != candles_f or abs(candles_f) == float("inf") or candles_f != int(candles_f):
            return ["n_session_candles_non_integer"]
        candles_n = int(candles_f)

    mean_finite = mean_x is not None and mean_x == mean_x and abs(mean_x) != float("inf")
    mean_active = mean_finite and mean_x > 0.0  # type: ignore[operator]
    rows_pos = rows_n is not None and rows_n > 0

    # Session L2 off: NaN/absent mean + zero/absent rows → skip
    if not mean_active and not rows_pos:
        return []

    errs: list[str] = []

    if mean_active:
        if not has_rows:
            errs.append("n_session_book_rows_missing_despite_mean_session_book_snaps")
        elif rows_n is not None and rows_n <= 0:
            errs.append("n_session_book_rows_non_positive_despite_mean_session_book_snaps")
        elif rows_n is not None and candles_n is not None and rows_n != candles_n:
            errs.append("n_session_book_rows_ne_n_session_candles")

    if rows_pos and has_mean and (not mean_finite or not (mean_x > 0.0)):  # type: ignore[operator]
        errs.append("mean_session_book_snaps_non_positive_despite_n_session_book_rows")

    return errs


def northset_session_ofi_sum_ic_honesty_errors(blob: object) -> list[str]:
    """Soft-verify session_ofi_sum_* IC companions when present.

    - ``session_ofi_sum_mean_ic`` / ``session_ofi_sum_mean_rank_ic`` / ``_t_ic`` finite
    - ``session_ofi_sum_p_ic`` ∈ [0, 1] when finite
    - ``session_ofi_sum_n_dates`` ≥ 0 when finite
    Never equate to daily ``ofi_mean_ic`` / ``ofi_p_ic`` (different feature).
    Research diagnostic only; never live Sharpe. Off kyle_ofi / METRICS_*.
    """
    if not isinstance(blob, dict):
        return []
    errs: list[str] = []
    for key in (
        "session_ofi_sum_mean_ic",
        "session_ofi_sum_mean_rank_ic",
        "session_ofi_sum_t_ic",
    ):
        if key not in blob:
            continue
        try:
            x = float(blob.get(key))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append(f"{key}_non_numeric")
            continue
        if x != x:
            continue
        if abs(x) == float("inf"):
            errs.append(f"{key}_non_finite_fail_closed")
    if "session_ofi_sum_p_ic" in blob:
        try:
            p = float(blob.get("session_ofi_sum_p_ic"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append("session_ofi_sum_p_ic_non_numeric")
        else:
            if p == p and abs(p) != float("inf") and not (0.0 <= p <= 1.0):
                errs.append("session_ofi_sum_p_ic_out_of_unit_interval")
            elif p == p and abs(p) == float("inf"):
                errs.append("session_ofi_sum_p_ic_non_finite_fail_closed")
    if "session_ofi_sum_n_dates" in blob:
        try:
            n = float(blob.get("session_ofi_sum_n_dates"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append("session_ofi_sum_n_dates_non_numeric")
        else:
            if n == n and abs(n) != float("inf") and n < 0.0:
                errs.append("session_ofi_sum_n_dates_negative")
            elif n == n and abs(n) == float("inf"):
                errs.append("session_ofi_sum_n_dates_non_finite")
    return errs


def candle_spread_over_mid_ic_implies_mean_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: when spread_over_mid IC is scored, mean ≥ 0.

    ``ic_spread_over_mid`` on candle_order_book ⇒ ``mean_spread_over_mid`` present
    and ≥ 0 when finite. Never equate IC to the mean. Research diagnostic only.
    """
    if not isinstance(blob, dict):
        return []
    if blob.get("family") != "candle_order_book":
        return []
    if "ic_spread_over_mid" not in blob:
        return []
    if "mean_spread_over_mid" not in blob:
        return ["mean_spread_over_mid_missing_while_spread_over_mid_ic_scored"]
    try:
        m = float(blob.get("mean_spread_over_mid"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return ["mean_spread_over_mid_non_numeric"]
    if m != m:
        return ["mean_spread_over_mid_nan_while_spread_over_mid_ic_scored"]
    if abs(m) == float("inf") or m < 0.0:
        return ["mean_spread_over_mid_negative_or_non_finite"]
    return []


def northset_session_ofi_sum_mean_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: session_ofi_sum_mean finite when present (not ±inf).

    Session-path mean of ``session_ofi_sum`` — signed OFI sum is unbounded, so
    no unit-interval claim; only reject ±inf. NaN (session L2 off) skipped.
    Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    val = blob.get("session_ofi_sum_mean")
    try:
        x = float(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return []
    if x != x:  # NaN
        return []
    if abs(x) == float("inf"):
        return ["session_ofi_sum_mean_non_finite"]
    return []


def northset_session_book_vpin_mean_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: session_book_vpin_mean ∈ [0, 1] when finite.

    Session book VPIN proxy is ``|ofi_sum| / ofi_abs_sum`` (synthetic_lob) →
    unit interval by construction. NaN skipped. Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    val = blob.get("session_book_vpin_mean")
    try:
        x = float(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return []
    if x != x:  # NaN
        return []
    if not (0.0 <= x <= 1.0):
        return ["session_book_vpin_mean_out_of_unit_interval"]
    return []


def northset_session_ofi_abs_sum_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: mean_session_ofi_abs_sum ≥ 0 when finite.

    Path mean of ``session_ofi_abs_sum`` (VPIN denominator companion) — not
    ``|session_ofi_sum_mean|``. NaN skipped. Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    val = blob.get("mean_session_ofi_abs_sum")
    try:
        x = float(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return []
    if x != x:
        return []
    if x < 0.0 or abs(x) == float("inf"):
        return ["mean_session_ofi_abs_sum_negative_or_non_finite"]
    return []


def northset_session_ofi_abs_dominates_sum_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: |session_ofi_sum_mean| ≤ mean_session_ofi_abs_sum when both finite.

    Path identity for VPIN denom companion. Never equate ofi_abs mean to
    |ofi_sum_mean| (Jensen) as a claimed equality. Research diagnostic only.
    """
    if not isinstance(blob, dict):
        return []
    try:
        s = float(blob.get("session_ofi_sum_mean"))  # type: ignore[arg-type]
        a = float(blob.get("mean_session_ofi_abs_sum"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return []
    if s != s or a != a:
        return []
    if abs(s) == float("inf") or abs(a) == float("inf"):
        return []
    if abs(s) > a + 1e-9:
        return ["session_ofi_abs_sum_less_than_abs_ofi_sum_mean"]
    return []


def northset_session_close_mid_micro_pair_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: when close_micro is finite, close_mid must also be finite.

    Companion keys from the same last-snap session aggregate. Both NaN (session
    L2 off) is fine; micro finite + mid NaN/±inf is dishonest pairing.
    Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []

    def _parse(key: str) -> float | None:
        val = blob.get(key)
        try:
            x = float(val)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
        return x

    micro = _parse("mean_session_close_micro_bps")
    mid = _parse("mean_session_close_mid")
    if micro is None or micro != micro:  # missing/NaN micro → skip
        return []
    if abs(micro) == float("inf"):
        return []  # micro non-finite handled by micro helper
    # micro finite → mid must be finite (NaN/missing/±inf fail)
    if mid is None or mid != mid or abs(mid) == float("inf"):
        return ["mean_session_close_mid_nan_while_micro_finite"]
    return []


NORTHSET_SESSION_MEANS_HONESTY_HELPERS = (
    northset_session_imbalance_mean_honesty_errors,
    northset_session_close_micro_bps_honesty_errors,
    northset_session_close_depth_honesty_errors,
    northset_session_imbalance_std_honesty_errors,
    northset_session_close_imbalance_honesty_errors,
    northset_session_close_mid_honesty_errors,
    northset_session_close_mid_micro_pair_honesty_errors,
    northset_session_close_spread_bps_honesty_errors,
    northset_session_spread_bps_mean_honesty_errors,
    northset_session_book_snaps_honesty_errors,
    northset_session_book_snaps_n_session_consistency_errors,
    northset_session_ofi_sum_mean_honesty_errors,
    northset_session_ofi_sum_ic_honesty_errors,
    northset_session_ofi_abs_sum_honesty_errors,
    northset_session_ofi_abs_dominates_sum_honesty_errors,
    northset_session_book_vpin_mean_honesty_errors,
)


def northset_session_means_honesty_errors(blob: object) -> list[str]:
    """Dispatcher: fan into all session-mean soft-verify helpers.

    Keeps individual helpers for unit tests; verify.py can call this once.
    Research diagnostic only; never live Sharpe / promotion.
    """
    errors: list[str] = []
    for fn in NORTHSET_SESSION_MEANS_HONESTY_HELPERS:
        errors.extend(fn(blob))
    return errors


# ---------------------------------------------------------------------------
# Soft-verify: nested northset["kyle_ofi"] residual_flow / dispersion honesty
# (Sergeant / research diagnostic only — never live Sharpe / promotion.)
# ---------------------------------------------------------------------------

_KYLE_RESIDUAL_SPEARMAN_KEYS = (
    "residual_ofi_ex_depth_fwd_delta_mid_mean_spearman",
    "residual_depth_ex_ofi_fwd_delta_mid_mean_spearman",
    "residual_depth_ex_ofi_fwd_ret_1_mean_spearman",
    "residual_ofi_ex_depth_fwd_ret_1_mean_spearman",
    "residual_depth_ex_ofi_fwd_ret_2_mean_spearman",
    "residual_ofi_ex_depth_fwd_ret_2_mean_spearman",
    "residual_depth_ex_ofi_fwd_ret_3_mean_spearman",
    "residual_ofi_ex_depth_fwd_ret_3_mean_spearman",
)

_KYLE_FORBIDDEN_TOKENS = ("sharpe", "sortino", "calmar", "pnl", "nav", "live_pnl_claim")


def _kyle_ofi_blob(blob: object) -> dict | None:
    """Return nested kyle_ofi dict from a northset family blob, or the blob itself."""
    if not isinstance(blob, dict):
        return None
    if blob.get("family") == "kyle_ofi":
        return blob
    nest = blob.get("kyle_ofi")
    return nest if isinstance(nest, dict) else None


def kyle_residual_flow_honesty_errors(blob: object) -> list[str]:
    """Soft-verify residual_flow receipt keys on nested ``kyle_ofi`` (or bare receipt).

    When any residual_* spearman key is present and finite:
    - ``research_only`` must be True
    - ``claim`` must be ``research_diagnostic_only``
    - companion ``*_t`` / ``*_n_dates`` keys must exist for that spearman
    - no Sharpe/pnl-token keys
    Missing nest / non-finite residual keys → skip (empty).
    """
    kyle = _kyle_ofi_blob(blob)
    if kyle is None:
        return []
    active = []
    for key in _KYLE_RESIDUAL_SPEARMAN_KEYS:
        if key not in kyle:
            continue
        try:
            x = float(kyle[key])  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if x == x:  # finite
            active.append(key)
    if not active:
        return []

    errors: list[str] = []
    if kyle.get("research_only") is not True:
        errors.append("kyle_ofi_residual_research_only_missing")
    if kyle.get("claim") != "research_diagnostic_only":
        errors.append("kyle_ofi_residual_claim_invalid")
    for key in active:
        stem = key[: -len("_mean_spearman")]
        if f"{stem}_t" not in kyle:
            errors.append(f"kyle_ofi_residual_missing_t:{stem}")
        if f"{stem}_n_dates" not in kyle:
            errors.append(f"kyle_ofi_residual_missing_n_dates:{stem}")
    leaked = [k for k in kyle if any(tok in str(k).lower() for tok in _KYLE_FORBIDDEN_TOKENS)]
    if leaked:
        errors.append("kyle_ofi_residual_forbidden_metric_keys")
    return errors


def northset_kyle_r2_unit_honesty_errors(blob: object) -> list[str]:
    """Soft-verify always-on ``kyle_r2`` / ``kyle_ofi_r2`` ∈ [0, 1] when finite.

    Name-mean OLS R² from ``kyle_lambda`` helpers (signed_volume vs ofi paths).
    NaN/absent skip; ±inf fail-closed. Research diagnostic only; never live Sharpe.
    Always-on surface — not nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    errs: list[str] = []
    for key in ("kyle_r2", "kyle_ofi_r2"):
        if key not in blob:
            continue
        try:
            x = float(blob.get(key))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append(f"{key}_non_numeric")
            continue
        if x != x:
            continue
        if abs(x) == float("inf") or not (0.0 <= x <= 1.0):
            errs.append(f"{key}_out_of_unit_interval")
    return errs


def kyle_r2_vs_kyle_ofi_r2_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate always-on ``kyle_r2`` with ``kyle_ofi_r2``.

    DATA_CONTRACTS: both are name-mean OLS R², different flow columns
    (``signed_volume`` vs ``ofi``). When both keys are present:

    - Distinct key identity
    - Key-scoped finite probes (ofi-only blob must not count as kyle_r2)
    - Numeric equality on clean synth is allowed

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Always-on — not nest invent / kyle_ofi.py overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_sv = "kyle_r2"
    k_ofi = "kyle_ofi_r2"
    if k_sv not in blob or k_ofi not in blob:
        return []

    errs: list[str] = []
    if k_sv == k_ofi:
        errs.append("kyle_r2_kyle_ofi_r2_keys_collapsed")

    if not _finite_scalar({k_sv: 0.5}.get(k_sv)):
        errs.append("kyle_r2_finite_probe_broken")
    if _finite_scalar({k_ofi: 0.5}.get(k_sv)):
        errs.append("kyle_ofi_r2_incorrectly_counts_as_finite_kyle_r2")
    if not _finite_scalar({k_ofi: 0.5}.get(k_ofi)):
        errs.append("kyle_ofi_r2_finite_probe_broken")
    if _finite_scalar({k_sv: 0.5}.get(k_ofi)):
        errs.append("kyle_r2_incorrectly_counts_as_finite_kyle_ofi_r2")
    return errs


def kyle_lambda_dispersion_honesty_errors(blob: object) -> list[str]:
    """Soft-verify Kyle λ dispersion stamps when present on nested ``kyle_ofi``.

    If ``kyle_lambda_depth_p50`` (or ofi twin) is finite: require research_only,
    research_diagnostic_only claim, and no Sharpe/pnl keys. Skip if absent/NaN.
    """
    kyle = _kyle_ofi_blob(blob)
    if kyle is None:
        return []
    markers = ("kyle_lambda_depth_p50", "kyle_lambda_ofi_p50", "kyle_lambda_dispersion_window")
    present = False
    for key in markers:
        if key not in kyle:
            continue
        if key == "kyle_lambda_dispersion_window":
            present = True
            break
        try:
            x = float(kyle[key])  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if x == x:
            present = True
            break
    if not present:
        return []
    errors: list[str] = []
    if kyle.get("research_only") is not True:
        errors.append("kyle_ofi_dispersion_research_only_missing")
    if kyle.get("claim") != "research_diagnostic_only":
        errors.append("kyle_ofi_dispersion_claim_invalid")
    leaked = [k for k in kyle if any(tok in str(k).lower() for tok in _KYLE_FORBIDDEN_TOKENS)]
    if leaked:
        errors.append("kyle_ofi_dispersion_forbidden_metric_keys")
    return errors


def kyle_lambda_ofi_depth_corr_honesty_errors(blob: object) -> list[str]:
    """Soft-verify OFI↔depth Kyle-λ corr stamps on nested ``kyle_ofi``.

    When ``kyle_lambda_ofi_depth_spearman`` or ``kyle_lambda_ofi_depth_pearson``
    is finite: require ``research_only``, ``claim=research_diagnostic_only``,
    companions ``kyle_lambda_ofi_depth_n_dates`` / ``*_prod_hac_t`` / ``*_prod_hac_p``,
    and no Sharpe/pnl-token keys. Missing / non-finite markers → skip.
    """
    kyle = _kyle_ofi_blob(blob)
    if kyle is None:
        return []
    markers = ("kyle_lambda_ofi_depth_spearman", "kyle_lambda_ofi_depth_pearson")
    present = False
    for key in markers:
        if key not in kyle:
            continue
        try:
            x = float(kyle[key])  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if x == x:
            present = True
            break
    if not present:
        return []
    errors: list[str] = []
    if kyle.get("research_only") is not True:
        errors.append("kyle_ofi_depth_corr_research_only_missing")
    if kyle.get("claim") != "research_diagnostic_only":
        errors.append("kyle_ofi_depth_corr_claim_invalid")
    for req in (
        "kyle_lambda_ofi_depth_n_dates",
        "kyle_lambda_ofi_depth_prod_hac_t",
        "kyle_lambda_ofi_depth_prod_hac_p",
    ):
        if req not in kyle:
            errors.append(f"kyle_ofi_depth_corr_missing:{req}")
    leaked = [k for k in kyle if any(tok in str(k).lower() for tok in _KYLE_FORBIDDEN_TOKENS)]
    if leaked:
        errors.append("kyle_ofi_depth_corr_forbidden_metric_keys")
    return errors


def kyle_ofi_join_coverage_honesty_errors(blob: object) -> list[str]:
    """Soft-verify nested ``kyle_ofi`` join_coverage + book_source honesty.

    When ``join_coverage`` is finite: reuse :func:`join_coverage_honesty_errors`
    (∈(0,1], or [floor,1] if ``min_join_coverage`` / ``book_join_coverage_floor``
    present) and require nonempty ``book_source``. NaN / absent → skip.
    Research diagnostic only; never live Sharpe / promotion.
    """
    kyle = _kyle_ofi_blob(blob)
    if kyle is None:
        return []
    val = kyle.get("join_coverage")
    if val is None:
        return []
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        # Present-but-malformed must not read as absent/NaN.
        return ["kyle_ofi_join_coverage_non_numeric"]
    x = float(val)
    if x != x:  # NaN
        return []
    errors: list[str] = list(join_coverage_honesty_errors(kyle))
    src = kyle.get("book_source")
    if not (isinstance(src, str) and str(src).strip()):
        errors.append("kyle_ofi_join_coverage_book_source_missing")
    return errors


def kyle_lambda_date_series_honesty_errors(blob: object) -> list[str]:
    """Soft-verify dump/date-series companion stamps on nested ``kyle_ofi``.

    When ``kyle_lambda_date_series_n_depth`` or ``kyle_lambda_date_series_n_ofi``
    is present and finite: require ``research_only``, ``claim=research_diagnostic_only``,
    and no Sharpe/pnl-token keys. Aligns with CLI ``--dump-lambda-series``
    research_only panel contract. NaN / absent → skip.
    """
    kyle = _kyle_ofi_blob(blob)
    if kyle is None:
        return []
    markers = ("kyle_lambda_date_series_n_depth", "kyle_lambda_date_series_n_ofi")
    present = False
    for key in markers:
        if key not in kyle:
            continue
        try:
            x = float(kyle[key])  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if x == x:
            present = True
            break
    if not present:
        return []
    errors: list[str] = []
    if kyle.get("research_only") is not True:
        errors.append("kyle_ofi_date_series_research_only_missing")
    if kyle.get("claim") != "research_diagnostic_only":
        errors.append("kyle_ofi_date_series_claim_invalid")
    leaked = [k for k in kyle if any(tok in str(k).lower() for tok in _KYLE_FORBIDDEN_TOKENS)]
    if leaked:
        errors.append("kyle_ofi_date_series_forbidden_metric_keys")
    return errors


def kyle_ofi_synthetic_source_honesty_errors(blob: object) -> list[str]:
    """Soft-verify nested ``kyle_ofi`` book_dgp / dgp / data_source SYNTHETIC consistency.

    Contract mirrors ``bench_kyle_ofi_fused`` stamping:
    - ``book_dgp`` / ``dgp`` ``synthetic_lob`` ⇔ ``data_source == "SYNTHETIC"``
    - when both ``dgp`` and ``book_dgp`` present, they must match
    - non-synthetic ``book_dgp`` ⇒ ``data_source`` equals ``book_source`` (when both set)
    Absent / empty markers → skip. Research diagnostic only; never live Sharpe.
    """
    kyle = _kyle_ofi_blob(blob)
    if kyle is None:
        return []

    def _str(key: str) -> str | None:
        v = kyle.get(key)
        if v is None:
            return None
        s = str(v).strip()
        return s if s else None

    book_dgp = _str("book_dgp")
    dgp = _str("dgp")
    data_source = _str("data_source")
    book_source = _str("book_source")
    if book_dgp is None and dgp is None and data_source is None:
        return []

    errors: list[str] = []
    if book_dgp is not None and dgp is not None and book_dgp != dgp:
        errors.append("kyle_ofi_dgp_book_dgp_mismatch")

    synth_dgp = None
    if book_dgp is not None:
        synth_dgp = book_dgp == "synthetic_lob"
    elif dgp is not None:
        synth_dgp = dgp == "synthetic_lob"

    if synth_dgp is True and data_source is not None and data_source != "SYNTHETIC":
        errors.append("kyle_ofi_synthetic_dgp_data_source_not_SYNTHETIC")
    if data_source == "SYNTHETIC":
        # require synthetic_lob on whichever dgp field is present
        if book_dgp is not None and book_dgp != "synthetic_lob":
            errors.append("kyle_ofi_SYNTHETIC_data_source_book_dgp_not_synthetic_lob")
        if dgp is not None and dgp != "synthetic_lob":
            errors.append("kyle_ofi_SYNTHETIC_data_source_dgp_not_synthetic_lob")
        if book_dgp is None and dgp is None:
            errors.append("kyle_ofi_SYNTHETIC_data_source_missing_dgp")
    if synth_dgp is False and data_source is not None:
        if data_source == "SYNTHETIC":
            errors.append("kyle_ofi_nonsynthetic_dgp_data_source_SYNTHETIC")
        elif book_source is not None and data_source != book_source:
            errors.append("kyle_ofi_nonsynthetic_data_source_ne_book_source")
    return errors


def kyle_ofi_label_synthetic_honesty_errors(blob: object) -> list[str]:
    """Soft-verify nest ``label`` SYN* vs ``data_source`` / dgp stamping.

    When ``label`` uppercases to a SYN* prefix (bench default path):
    - ``book_dgp``/``dgp`` ``synthetic_lob`` ⇒ ``data_source`` must be ``SYNTHETIC``
    - non-synthetic dgp ⇒ ``data_source`` must equal ``book_source`` (when set)
    Absent / non-SYN label → skip. Thin fail-closed; never live Sharpe.
    """
    kyle = _kyle_ofi_blob(blob)
    if kyle is None:
        return []
    label = kyle.get("label")
    if label is None or not str(label).strip():
        return []
    if not str(label).upper().startswith("SYN"):
        return []

    def _str(key: str) -> str | None:
        v = kyle.get(key)
        if v is None:
            return None
        s = str(v).strip()
        return s if s else None

    book_dgp = _str("book_dgp")
    dgp = _str("dgp")
    data_source = _str("data_source")
    book_source = _str("book_source")
    if data_source is None:
        return []

    synth_dgp = None
    if book_dgp is not None:
        synth_dgp = book_dgp == "synthetic_lob"
    elif dgp is not None:
        synth_dgp = dgp == "synthetic_lob"

    errors: list[str] = []
    if synth_dgp is True and data_source != "SYNTHETIC":
        errors.append("kyle_ofi_SYN_label_data_source_not_SYNTHETIC")
    if synth_dgp is False:
        if data_source == "SYNTHETIC":
            errors.append("kyle_ofi_SYN_label_nonsynthetic_dgp_data_source_SYNTHETIC")
        elif book_source is not None and data_source != book_source:
            errors.append("kyle_ofi_SYN_label_data_source_ne_book_source")
    if synth_dgp is None and data_source != "SYNTHETIC":
        # SYN* label with no dgp stamped still expects SYNTHETIC data_source
        errors.append("kyle_ofi_SYN_label_data_source_not_SYNTHETIC")
    return errors


def _kyle_ofi_has_nest_diagnostic_marker(kyle: dict) -> bool:
    """True when any kyle nest diagnostic stamp is present/finite."""
    for key in _KYLE_RESIDUAL_SPEARMAN_KEYS:
        if key not in kyle:
            continue
        try:
            x = float(kyle[key])  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if x == x:
            return True
    for key in (
        "kyle_lambda_depth_p50",
        "kyle_lambda_ofi_p50",
        "kyle_lambda_ofi_depth_spearman",
        "kyle_lambda_ofi_depth_pearson",
        "kyle_lambda_date_series_n_depth",
        "kyle_lambda_date_series_n_ofi",
        "join_coverage",
        "kyle_lambda_depth_mean",
        "kyle_lambda_ofi_mean",
    ):
        if key not in kyle:
            continue
        try:
            x = float(kyle[key])  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if x == x:
            return True
    if "kyle_lambda_dispersion_window" in kyle:
        return True
    for key in ("book_dgp", "dgp", "data_source", "label", "book_source"):
        v = kyle.get(key)
        if isinstance(v, str) and v.strip():
            return True
    return False


def kyle_ofi_nest_claim_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: any nest diagnostic marker ⇒ research_only + claim.

    Umbrella fail-closed across residual/dispersion/ofi_depth/join/date_series/
    source stamps so a partial nest cannot omit honesty labels. Skip when no
    markers. Research diagnostic only; never live Sharpe.
    """
    kyle = _kyle_ofi_blob(blob)
    if kyle is None:
        return []
    if not _kyle_ofi_has_nest_diagnostic_marker(kyle):
        return []
    errors: list[str] = []
    if kyle.get("research_only") is not True:
        errors.append("kyle_ofi_nest_research_only_missing")
    if kyle.get("claim") != "research_diagnostic_only":
        errors.append("kyle_ofi_nest_claim_invalid")
    leaked = [k for k in kyle if any(tok in str(k).lower() for tok in _KYLE_FORBIDDEN_TOKENS)]
    if leaked:
        errors.append("kyle_ofi_nest_forbidden_metric_keys")
    return errors


_KYLE_OFI_IC_METHOD_ALLOWED = frozenset({"date_level_spearman_hac"})


def kyle_ofi_ic_method_honesty_errors(blob: object) -> list[str]:
    """Soft-verify nested ``kyle_ofi`` ``ic_method`` stamp honesty.

    Contract: bench/residual paths stamp ``ic_method=date_level_spearman_hac``.
    - If ``ic_method`` present: must be in the allowed set (fail-closed otherwise).
    - If any IC diagnostic marker is finite (residual spearman, λ means, ofi_depth
      corr, dispersion window/p50) and ``ic_method`` absent → missing error.
    Absent markers and absent method → skip. Research diagnostic only.
    """
    kyle = _kyle_ofi_blob(blob)
    if kyle is None:
        return []

    def _finite(key: str) -> bool:
        if key not in kyle:
            return False
        try:
            x = float(kyle[key])  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return False
        return x == x

    ic_markers = False
    for key in _KYLE_RESIDUAL_SPEARMAN_KEYS:
        if _finite(key):
            ic_markers = True
            break
    if not ic_markers:
        for key in (
            "kyle_lambda_depth_mean",
            "kyle_lambda_ofi_mean",
            "kyle_lambda_depth_p50",
            "kyle_lambda_ofi_p50",
            "kyle_lambda_ofi_depth_spearman",
            "kyle_lambda_ofi_depth_pearson",
        ):
            if _finite(key):
                ic_markers = True
                break
    if not ic_markers and "kyle_lambda_dispersion_window" in kyle:
        ic_markers = True

    method = kyle.get("ic_method")
    errors: list[str] = []
    if method is None or (isinstance(method, str) and not method.strip()):
        if ic_markers:
            errors.append("kyle_ofi_ic_method_missing")
        return errors
    method_s = str(method).strip()
    if method_s not in _KYLE_OFI_IC_METHOD_ALLOWED:
        errors.append("kyle_ofi_ic_method_invalid")
    return errors


def kyle_ofi_family_honesty_errors(blob: object) -> list[str]:
    """Soft-verify nested ``kyle_ofi`` ``family`` stamp when diagnostics present.

    When any nest diagnostic marker is present/finite: ``family`` must be exactly
    ``kyle_ofi``. Also fail-closed when a kyle-shaped receipt (λ/residual/ofi_depth
    keys) carries a wrong ``family`` — ``_kyle_ofi_blob`` would otherwise skip.
    Bare empty / northset-only scalars → skip. Research diagnostic only.
    """
    if not isinstance(blob, dict):
        return []
    nest = blob.get("kyle_ofi")
    if isinstance(nest, dict):
        kyle = nest
    elif blob.get("family") == "kyle_ofi" or any(
        k in blob
        for k in (
            "kyle_lambda_depth_mean",
            "kyle_lambda_ofi_mean",
            "kyle_lambda_ofi_depth_spearman",
            "kyle_lambda_dispersion_window",
            "residual_ofi_ex_depth_fwd_delta_mid_mean_spearman",
        )
    ):
        kyle = blob
    else:
        return []
    if not _kyle_ofi_has_nest_diagnostic_marker(kyle):
        return []
    if kyle.get("family") != "kyle_ofi":
        return ["kyle_ofi_family_invalid"]
    return []


def kyle_ofi_hac_lags_honesty_errors(blob: object) -> list[str]:
    """Soft-verify optional ``hac_lags`` stamp on nested ``kyle_ofi``.

    When ``hac_lags`` is present: must be finite and ≥ 0 (integer-valued).
    Absent → skip (receipt may omit the key without editing ``kyle_ofi.py``).
    Also: when both depth rolling HAC lo/hi finite, require lo ≤ hi (and ofi twin).
    Research diagnostic only; never live Sharpe.
    """
    kyle = _kyle_ofi_blob(blob)
    if kyle is None:
        return []
    errors: list[str] = []
    if "hac_lags" in kyle and kyle.get("hac_lags") is not None:
        try:
            x = float(kyle["hac_lags"])  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return ["kyle_ofi_hac_lags_non_numeric"]
        if x != x or abs(x) == float("inf"):
            errors.append("kyle_ofi_hac_lags_non_finite")
        elif x < 0:
            errors.append("kyle_ofi_hac_lags_negative")
        elif abs(x - round(x)) > 1e-12:
            errors.append("kyle_ofi_hac_lags_not_integer")

    def _pair_lo_hi(lo_key: str, hi_key: str, err: str) -> None:
        if lo_key not in kyle or hi_key not in kyle:
            return
        try:
            lo = float(kyle[lo_key])  # type: ignore[arg-type]
            hi = float(kyle[hi_key])  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return
        if lo != lo or hi != hi:
            return
        if lo > hi:
            errors.append(err)

    _pair_lo_hi(
        "kyle_lambda_depth_rolling_hac_lo",
        "kyle_lambda_depth_rolling_hac_hi",
        "kyle_ofi_depth_rolling_hac_lo_gt_hi",
    )
    _pair_lo_hi(
        "kyle_lambda_ofi_rolling_hac_lo",
        "kyle_lambda_ofi_rolling_hac_hi",
        "kyle_ofi_ofi_rolling_hac_lo_gt_hi",
    )
    return errors


def kyle_ofi_min_names_honesty_errors(blob: object) -> list[str]:
    """Soft-verify nested ``kyle_ofi`` ``min_names`` stamp.

    When ``min_names`` present: finite integer-valued and ≥ 1.
    When bench-level markers present (``kyle_lambda_depth_mean`` / ``n_fused``)
    and ``min_names`` absent → missing. Else skip. Research diagnostic only.
    """
    kyle = _kyle_ofi_blob(blob)
    if kyle is None:
        return []
    benchish = False
    for key in ("kyle_lambda_depth_mean", "kyle_lambda_ofi_mean", "n_fused", "n_scored"):
        if key not in kyle:
            continue
        try:
            x = float(kyle[key])  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if x == x:
            benchish = True
            break
    if "min_names" not in kyle or kyle.get("min_names") is None:
        return ["kyle_ofi_min_names_missing"] if benchish else []
    try:
        x = float(kyle["min_names"])  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return ["kyle_ofi_min_names_non_numeric"]
    errors: list[str] = []
    if x != x or abs(x) == float("inf"):
        errors.append("kyle_ofi_min_names_non_finite")
    elif x < 1:
        errors.append("kyle_ofi_min_names_lt_one")
    elif abs(x - round(x)) > 1e-12:
        errors.append("kyle_ofi_min_names_not_integer")
    return errors


def kyle_ofi_n_fused_scored_honesty_errors(blob: object) -> list[str]:
    """Soft-verify nested ``kyle_ofi`` ``n_fused`` / ``n_scored`` consistency.

    When either count is present: both must be finite integer-valued ≥ 0 and
    ``n_scored ≤ n_fused``. One without the other → missing companion.
    Absent both → skip. Research diagnostic only; never live Sharpe.
    """
    kyle = _kyle_ofi_blob(blob)
    if kyle is None:
        return []
    has_f = "n_fused" in kyle and kyle.get("n_fused") is not None
    has_s = "n_scored" in kyle and kyle.get("n_scored") is not None
    if not has_f and not has_s:
        return []
    errors: list[str] = []
    if has_f ^ has_s:
        errors.append("kyle_ofi_n_fused_scored_pair_incomplete")
        return errors

    def _parse(key: str) -> float | None:
        try:
            x = float(kyle[key])  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errors.append(f"kyle_ofi_{key}_non_numeric")
            return None
        if x != x or abs(x) == float("inf"):
            errors.append(f"kyle_ofi_{key}_non_finite")
            return None
        if x < 0:
            errors.append(f"kyle_ofi_{key}_negative")
        if abs(x - round(x)) > 1e-12:
            errors.append(f"kyle_ofi_{key}_not_integer")
        return x

    fused = _parse("n_fused")
    scored = _parse("n_scored")
    if fused is not None and scored is not None and scored > fused:
        errors.append("kyle_ofi_n_scored_gt_n_fused")
    return errors


def kyle_ofi_book_panel_path_honesty_errors(blob: object) -> list[str]:
    """Soft-verify nested ``kyle_ofi`` ``book_panel_path`` stamp.

    Absent key or explicit ``None`` → skip (synthetic benches omit a path).
    When present and not ``None``: must be a nonempty string. Empty / whitespace
    fail-closed. Research diagnostic only; never live Sharpe.
    """
    kyle = _kyle_ofi_blob(blob)
    if kyle is None:
        return []
    if "book_panel_path" not in kyle:
        return []
    v = kyle.get("book_panel_path")
    if v is None:
        return []
    if not isinstance(v, str) or not str(v).strip():
        return ["kyle_ofi_book_panel_path_empty"]
    return []


def kyle_ofi_min_join_coverage_pair_honesty_errors(blob: object) -> list[str]:
    """Soft-verify ``min_join_coverage`` / ``book_join_coverage_floor`` vs join_coverage.

    When a floor stamp is present: finite and ∈ [0, 1]. When ``join_coverage`` is
    also finite: require ``join_coverage >= floor`` (and ≤ 1). Absent floor → skip
    (``kyle_ofi_join_coverage_honesty_errors`` still covers open-unit join alone).
    Research diagnostic only.
    """
    kyle = _kyle_ofi_blob(blob)
    if kyle is None:
        return []
    raw_floor = kyle.get("min_join_coverage")
    if raw_floor is None:
        raw_floor = kyle.get("book_join_coverage_floor")
    if raw_floor is None:
        return []
    try:
        floor = float(raw_floor)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return ["kyle_ofi_min_join_coverage_non_numeric"]
    errors: list[str] = []
    if floor != floor or abs(floor) == float("inf"):
        errors.append("kyle_ofi_min_join_coverage_non_finite")
        return errors
    if not (0.0 <= floor <= 1.0):
        errors.append("kyle_ofi_min_join_coverage_outside_unit_interval")
    if "join_coverage" not in kyle or kyle.get("join_coverage") is None:
        return errors
    try:
        cov = float(kyle["join_coverage"])  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return errors
    if cov != cov or abs(cov) == float("inf"):
        return errors
    if cov + 1e-12 < floor:
        errors.append("kyle_ofi_join_coverage_below_min_join_coverage")
    if cov > 1.0 + 1e-12:
        errors.append("kyle_ofi_join_coverage_above_one_with_floor")
    return errors


def kyle_ofi_dispersion_window_honesty_errors(blob: object) -> list[str]:
    """Soft-verify ``kyle_lambda_dispersion_window`` int stamp on nested ``kyle_ofi``.

    When dispersion markers present (``kyle_lambda_depth_p50`` / ``kyle_lambda_ofi_p50``
    finite, or window key expected): window must be present, finite, integer-valued,
    and ≥ 1. Absent markers and absent window → skip. Research diagnostic only.
    """
    kyle = _kyle_ofi_blob(blob)
    if kyle is None:
        return []

    def _finite(key: str) -> bool:
        if key not in kyle:
            return False
        try:
            x = float(kyle[key])  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return False
        return x == x

    disp_markers = _finite("kyle_lambda_depth_p50") or _finite("kyle_lambda_ofi_p50")
    if (
        "kyle_lambda_dispersion_window" not in kyle
        or kyle.get("kyle_lambda_dispersion_window") is None
    ):
        return ["kyle_ofi_dispersion_window_missing"] if disp_markers else []
    try:
        x = float(kyle["kyle_lambda_dispersion_window"])  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return ["kyle_ofi_dispersion_window_non_numeric"]
    errors: list[str] = []
    if x != x or abs(x) == float("inf"):
        errors.append("kyle_ofi_dispersion_window_non_finite")
    elif x < 1:
        errors.append("kyle_ofi_dispersion_window_lt_one")
    elif abs(x - round(x)) > 1e-12:
        errors.append("kyle_ofi_dispersion_window_not_integer")
    return errors


def kyle_ofi_diagnostic_string_honesty_errors(blob: object) -> list[str]:
    """Soft-verify optional nest ``diagnostic`` string stamp.

    Full ``bench_kyle_ofi_fused`` omits ``diagnostic`` (skip). Sub-receipts
    (residual / λ / ofi_depth_corr) stamp it: when the key is present it must be
    a nonempty string without Sharpe/pnl tokens. Research diagnostic only.
    """
    kyle = _kyle_ofi_blob(blob)
    if kyle is None:
        return []
    if "diagnostic" not in kyle:
        return []
    v = kyle.get("diagnostic")
    if v is None or not isinstance(v, str) or not str(v).strip():
        return ["kyle_ofi_diagnostic_empty"]
    s = str(v).strip()
    if any(tok in s.lower() for tok in _KYLE_FORBIDDEN_TOKENS):
        return ["kyle_ofi_diagnostic_forbidden_token"]
    return []


def kyle_ofi_lambda_decile_order_honesty_errors(blob: object) -> list[str]:
    """Soft-verify Kyle λ dispersion decile order on nested ``kyle_ofi``.

    For depth and ofi sides independently: when p10/p50/p90 are all finite,
    require ``p10 ≤ p50 ≤ p90``. Partial / absent / NaN triples → skip that side.
    Research diagnostic only; never live Sharpe.
    """
    kyle = _kyle_ofi_blob(blob)
    if kyle is None:
        return []

    def _finite(key: str) -> float | None:
        if key not in kyle:
            return None
        try:
            x = float(kyle[key])  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
        if x != x or abs(x) == float("inf"):
            return None
        return x

    errors: list[str] = []
    for _side, keys, err in (
        (
            "depth",
            (
                "kyle_lambda_depth_p10",
                "kyle_lambda_depth_p50",
                "kyle_lambda_depth_p90",
            ),
            "kyle_ofi_depth_decile_order_invalid",
        ),
        (
            "ofi",
            (
                "kyle_lambda_ofi_p10",
                "kyle_lambda_ofi_p50",
                "kyle_lambda_ofi_p90",
            ),
            "kyle_ofi_ofi_decile_order_invalid",
        ),
    ):
        present = [v for v in (_finite(k) for k in keys) if v is not None]
        if len(present) != len(keys):
            continue
        p10, p50, p90 = present
        if not (p10 <= p50 <= p90):
            errors.append(err)
    return errors


def kyle_ofi_std_iqr_honesty_errors(blob: object) -> list[str]:
    """Soft-verify nested ``kyle_ofi`` λ ``std`` / ``iqr`` stamps are ≥ 0 when finite.

    Covers depth/ofi ``kyle_lambda_*_std`` and ``kyle_lambda_*_iqr``. Absent / NaN
    → skip that key. Research diagnostic only; never live Sharpe.
    """
    kyle = _kyle_ofi_blob(blob)
    if kyle is None:
        return []
    errors: list[str] = []
    for key in (
        "kyle_lambda_depth_std",
        "kyle_lambda_ofi_std",
        "kyle_lambda_depth_iqr",
        "kyle_lambda_ofi_iqr",
    ):
        if key not in kyle:
            continue
        try:
            x = float(kyle[key])  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errors.append(f"kyle_ofi_{key}_non_numeric")
            continue
        if x != x:
            continue
        if abs(x) == float("inf"):
            errors.append(f"kyle_ofi_{key}_non_finite")
        elif x < 0:
            errors.append(f"kyle_ofi_{key}_negative")
    return errors


def kyle_ofi_rolling_mean_hac_band_honesty_errors(blob: object) -> list[str]:
    """Soft-verify rolling mean present/finite when HAC band lo/hi are finite.

    For depth and ofi sides: if both ``*_rolling_hac_lo`` and ``*_rolling_hac_hi``
    are finite, require ``*_rolling_mean`` present and finite. Also lo ≤ hi
    (parity with hac_lags helper). Absent band → skip. Research diagnostic only.
    """
    kyle = _kyle_ofi_blob(blob)
    if kyle is None:
        return []

    def _finite(key: str) -> float | None:
        if key not in kyle:
            return None
        try:
            x = float(kyle[key])  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
        if x != x or abs(x) == float("inf"):
            return None
        return x

    errors: list[str] = []
    for side in ("depth", "ofi"):
        lo = _finite(f"kyle_lambda_{side}_rolling_hac_lo")
        hi = _finite(f"kyle_lambda_{side}_rolling_hac_hi")
        if lo is None or hi is None:
            continue
        if lo > hi:
            errors.append(f"kyle_ofi_{side}_rolling_hac_lo_gt_hi")
        mean = _finite(f"kyle_lambda_{side}_rolling_mean")
        if mean is None:
            errors.append(f"kyle_ofi_{side}_rolling_mean_missing_with_hac_band")
    return errors


def kyle_ofi_n_dates_companion_honesty_errors(blob: object) -> list[str]:
    """Soft-verify ``*_n_dates`` ≥ 1 when companion IC / t stamps are finite.

    Companions:
    - finite ``*_mean_spearman`` ⇒ ``*_n_dates`` present, finite, integer-valued, ≥ 1
    - finite IC-style ``*_t`` (excl. ``*hac*`` / ``*rolling*``) ⇒ matching ``*_n_dates``
      for residual / flow IC stems, or side-level ``kyle_lambda_{depth,ofi}_n_dates``
      for bare ``kyle_lambda_{depth,ofi}_t`` (not per-target fwd λ t-stats)
    Absent companions → skip. Research diagnostic only; never live Sharpe.
    """
    kyle = _kyle_ofi_blob(blob)
    if kyle is None:
        return []

    def _finite(key: str) -> float | None:
        if key not in kyle:
            return None
        try:
            x = float(kyle[key])  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
        if x != x or abs(x) == float("inf"):
            return None
        return x

    def _n_dates_stem_for_t(stem: str) -> str | None:
        if stem in {"kyle_lambda_depth", "kyle_lambda_ofi"}:
            return stem
        ic_prefixes = (
            "residual_",
            "ofi_delta_mid_",
            "signed_depth_",
            "depth_flow_",
            "ofi_flow_",
            "ofi_fwd_",
            "signed_depth_fwd_",
        )
        if stem.startswith(ic_prefixes):
            return stem
        return None

    required: set[str] = set()
    for key in kyle:
        if not isinstance(key, str):
            continue
        if key.endswith("_mean_spearman") and _finite(key) is not None:
            required.add(key[: -len("_mean_spearman")])
        elif (
            key.endswith("_t")
            and "hac" not in key
            and "rolling" not in key
            and _finite(key) is not None
        ):
            mapped = _n_dates_stem_for_t(key[: -len("_t")])
            if mapped is not None:
                required.add(mapped)

    errors: list[str] = []
    for stem in sorted(required):
        nkey = f"{stem}_n_dates"
        if nkey not in kyle or kyle.get(nkey) is None:
            errors.append(f"kyle_ofi_n_dates_missing:{stem}")
            continue
        try:
            n = float(kyle[nkey])  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errors.append(f"kyle_ofi_n_dates_non_numeric:{stem}")
            continue
        if n != n or abs(n) == float("inf"):
            errors.append(f"kyle_ofi_n_dates_non_finite:{stem}")
        elif n < 1:
            errors.append(f"kyle_ofi_n_dates_lt_one:{stem}")
        elif abs(n - round(n)) > 1e-12:
            errors.append(f"kyle_ofi_n_dates_not_integer:{stem}")
    return errors


def kyle_ofi_pvalue_honesty_errors(blob: object) -> list[str]:
    """Soft-verify nested ``kyle_ofi`` p-value stamps ∈ [0, 1] when finite.

    Any key ending in ``_p`` (IC / HAC p-values) that parses as finite must lie
    in the closed unit interval. Absent / NaN → skip that key. Research
    diagnostic only; never live Sharpe.
    """
    kyle = _kyle_ofi_blob(blob)
    if kyle is None:
        return []
    errors: list[str] = []
    for key, raw in kyle.items():
        if not isinstance(key, str) or not key.endswith("_p"):
            continue
        # deciles are *_p10/*_p50/*_p90 — they do not end with lone "_p"
        try:
            x = float(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errors.append(f"kyle_ofi_pvalue_non_numeric:{key}")
            continue
        if x != x:
            continue
        if abs(x) == float("inf"):
            errors.append(f"kyle_ofi_pvalue_non_finite:{key}")
        elif not (0.0 <= x <= 1.0):
            errors.append(f"kyle_ofi_pvalue_outside_unit_interval:{key}")
    return errors


def kyle_ofi_spearman_honesty_errors(blob: object) -> list[str]:
    """Soft-verify nested ``kyle_ofi`` Spearman (and Pearson) corr stamps ∈ [-1, 1].

    Any key ending in ``_mean_spearman``, ``_spearman``, or ``_pearson`` that is
    finite must lie in the closed interval. Absent / NaN → skip. Research
    diagnostic only; never live Sharpe.
    """
    kyle = _kyle_ofi_blob(blob)
    if kyle is None:
        return []
    errors: list[str] = []
    for key, raw in kyle.items():
        if not isinstance(key, str):
            continue
        if not (
            key.endswith("_mean_spearman") or key.endswith("_spearman") or key.endswith("_pearson")
        ):
            continue
        try:
            x = float(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errors.append(f"kyle_ofi_corr_non_numeric:{key}")
            continue
        if x != x:
            continue
        if abs(x) == float("inf"):
            errors.append(f"kyle_ofi_corr_non_finite:{key}")
        elif not (-1.0 <= x <= 1.0):
            errors.append(f"kyle_ofi_corr_outside_unit_interval:{key}")
    return errors


def kyle_ofi_tstat_honesty_errors(blob: object) -> list[str]:
    """Soft-verify nested ``kyle_ofi`` t-stat stamps are finite when present.

    - Any key ending in ``_t`` that is present/non-null must parse finite (not NaN/±inf).
    - Finite ``*_mean_spearman`` ⇒ companion ``*_t`` present and finite (IC companion
      to p / n_dates soft-verify).
    Research diagnostic only; never live Sharpe.
    """
    kyle = _kyle_ofi_blob(blob)
    if kyle is None:
        return []

    def _finite(key: str) -> float | None:
        if key not in kyle or kyle.get(key) is None:
            return None
        try:
            x = float(kyle[key])  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
        if x != x or abs(x) == float("inf"):
            return None
        return x

    errors: list[str] = []
    for key, raw in kyle.items():
        if not isinstance(key, str) or not key.endswith("_t"):
            continue
        if raw is None:
            continue
        try:
            x = float(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errors.append(f"kyle_ofi_tstat_non_numeric:{key}")
            continue
        if x != x or abs(x) == float("inf"):
            errors.append(f"kyle_ofi_tstat_non_finite:{key}")

    for key in list(kyle.keys()):
        if not isinstance(key, str) or not key.endswith("_mean_spearman"):
            continue
        if _finite(key) is None:
            continue
        stem = key[: -len("_mean_spearman")]
        tkey = f"{stem}_t"
        if tkey not in kyle or kyle.get(tkey) is None:
            errors.append(f"kyle_ofi_tstat_missing:{stem}")
            continue
        # non-finite already flagged above if key present; ensure companion miss covered
        if _finite(tkey) is None and (
            f"kyle_ofi_tstat_non_finite:{tkey}" not in errors
            and f"kyle_ofi_tstat_non_numeric:{tkey}" not in errors
        ):
            errors.append(f"kyle_ofi_tstat_non_finite:{tkey}")
    return errors


def kyle_ofi_label_nonempty_honesty_errors(blob: object) -> list[str]:
    """Soft-verify nest ``label`` nonempty when diagnostic markers are present.

    Scan residual: SYN* / claim helpers skip blank labels. When any nest
    diagnostic marker is present, ``label`` must be a nonempty string.
    Absent markers → skip. Research diagnostic only; never live Sharpe.
    """
    kyle = _kyle_ofi_blob(blob)
    if kyle is None:
        return []
    if not _kyle_ofi_has_nest_diagnostic_marker(kyle):
        return []
    label = kyle.get("label")
    if label is None or not isinstance(label, str) or not str(label).strip():
        return ["kyle_ofi_label_empty_with_markers"]
    return []


def kyle_ofi_date_series_counts_honesty_errors(blob: object) -> list[str]:
    """Soft-verify ``kyle_lambda_date_series_n_{depth,ofi}`` ≥ 0 ints when present.

    Complements claim soft-verify on those stamps. Absent → skip.
    """
    kyle = _kyle_ofi_blob(blob)
    if kyle is None:
        return []
    errors: list[str] = []
    for key in ("kyle_lambda_date_series_n_depth", "kyle_lambda_date_series_n_ofi"):
        if key not in kyle or kyle.get(key) is None:
            continue
        try:
            x = float(kyle[key])  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errors.append(f"kyle_ofi_{key}_non_numeric")
            continue
        if x != x or abs(x) == float("inf"):
            errors.append(f"kyle_ofi_{key}_non_finite")
        elif x < 0:
            errors.append(f"kyle_ofi_{key}_negative")
        elif abs(x - round(x)) > 1e-12:
            errors.append(f"kyle_ofi_{key}_not_integer")
    return errors


def kyle_ofi_nest_honesty_errors(blob: object) -> list[str]:
    """Dispatcher: full kyle nest soft-verify suite (incl. label nonempty + date_series n)."""
    errors: list[str] = []
    errors.extend(kyle_residual_flow_honesty_errors(blob))
    errors.extend(kyle_lambda_dispersion_honesty_errors(blob))
    errors.extend(kyle_lambda_ofi_depth_corr_honesty_errors(blob))
    errors.extend(kyle_ofi_join_coverage_honesty_errors(blob))
    errors.extend(kyle_lambda_date_series_honesty_errors(blob))
    errors.extend(kyle_ofi_synthetic_source_honesty_errors(blob))
    errors.extend(kyle_ofi_label_synthetic_honesty_errors(blob))
    errors.extend(kyle_ofi_nest_claim_honesty_errors(blob))
    errors.extend(kyle_ofi_ic_method_honesty_errors(blob))
    errors.extend(kyle_ofi_family_honesty_errors(blob))
    errors.extend(kyle_ofi_hac_lags_honesty_errors(blob))
    errors.extend(kyle_ofi_min_names_honesty_errors(blob))
    errors.extend(kyle_ofi_n_fused_scored_honesty_errors(blob))
    errors.extend(kyle_ofi_book_panel_path_honesty_errors(blob))
    errors.extend(kyle_ofi_min_join_coverage_pair_honesty_errors(blob))
    errors.extend(kyle_ofi_dispersion_window_honesty_errors(blob))
    errors.extend(kyle_ofi_diagnostic_string_honesty_errors(blob))
    errors.extend(kyle_ofi_lambda_decile_order_honesty_errors(blob))
    errors.extend(kyle_ofi_std_iqr_honesty_errors(blob))
    errors.extend(kyle_ofi_rolling_mean_hac_band_honesty_errors(blob))
    errors.extend(kyle_ofi_n_dates_companion_honesty_errors(blob))
    errors.extend(kyle_ofi_pvalue_honesty_errors(blob))
    errors.extend(kyle_ofi_spearman_honesty_errors(blob))
    errors.extend(kyle_ofi_tstat_honesty_errors(blob))
    errors.extend(kyle_ofi_label_nonempty_honesty_errors(blob))
    errors.extend(kyle_ofi_date_series_counts_honesty_errors(blob))
    return errors


# --- Mac-only honesty helpers merged by Lt ---
# Spread-alias tolerance constants (restored: the merge dropped them while the
# helpers below call math.isclose with these names).
_NORTHSET_SPREAD_REL_TOL = 1e-9
_NORTHSET_SPREAD_ABS_TOL = 1e-12


def northset_half_spread_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: ``mean_half_spread ≈ 0.5 * mean_quoted_spread`` when both finite.

    Missing / non-finite either side → skip. Research diagnostic only; never
    live Sharpe / promotion.
    """
    pair = _finite_pair(blob, "mean_half_spread", "mean_quoted_spread")
    if pair is None:
        return []
    half, quoted = pair
    if not math.isclose(
        half,
        0.5 * quoted,
        rel_tol=_NORTHSET_SPREAD_REL_TOL,
        abs_tol=_NORTHSET_SPREAD_ABS_TOL,
    ):
        return ["mean_half_spread_not_half_of_mean_quoted_spread"]
    return []


def northset_all_share_unit_honesty_errors(blob: object) -> list[str]:
    """Soft-verify every top-level ``*_share`` ∈ [0, 1] when finite.

    Covers overnight_share, tob_*_share, sweep_*_reclaim/follow_share, etc.
    Skip kyle_*/METRICS_*. Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    errs: list[str] = []
    for key, raw in blob.items():
        if not isinstance(key, str) or not key.endswith("_share"):
            continue
        if key.startswith("kyle_") or "METRICS_" in key:
            continue
        if raw is None:
            continue
        try:
            x = float(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append(f"{key}_non_numeric")
            continue
        if x != x:
            continue
        if abs(x) == float("inf") or not (0.0 <= x <= 1.0):
            errs.append(f"{key}_out_of_unit_interval")
    return errs


def northset_all_fraction_unit_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: every top-level ``*_fraction`` key ∈ [0, 1] when finite.

    Covers sweep fold-positive fractions and any future ``*_fraction`` rates
    without per-key helpers. Nested dicts are ignored (top-level only).
    """
    if not isinstance(blob, dict):
        return []
    errors: list[str] = []
    for key, raw in blob.items():
        if not str(key).endswith("_fraction"):
            continue
        if raw is None:
            continue
        try:
            val = float(raw)
        except (TypeError, ValueError):
            errors.append(f"{key}_non_numeric")
            continue
        if val != val:  # NaN ok
            continue
        if not (0.0 <= val <= 1.0):
            errors.append(f"{key}_out_of_unit_interval")
    return errors


def northset_sweep_fold_positive_rates_honesty_errors(blob: object) -> list[str]:
    """Soft-verify sweep fold-positive rates ∈ [0, 1] when finite.

    Explicit residual (also covered by ``*_fraction`` catch-all):
    ``sweep_min_fold_positive_fraction``, ``sweep_reject_fold_positive_fraction``,
    ``sweep_follow_fold_positive_fraction``. Min threshold must be in (0, 1]
    when finite (0 is not a meaningful fold floor). Off kyle invent.
    Research diagnostic only.
    """
    if not isinstance(blob, dict):
        return []
    errs: list[str] = []
    keys = (
        "sweep_min_fold_positive_fraction",
        "sweep_reject_fold_positive_fraction",
        "sweep_follow_fold_positive_fraction",
    )
    for key in keys:
        if key not in blob:
            continue
        try:
            x = float(blob.get(key))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append(f"{key}_non_numeric")
            continue
        if x != x:
            continue
        if abs(x) == float("inf") or not (0.0 <= x <= 1.0):
            errs.append(f"{key}_out_of_unit_interval")
            continue
        if key == "sweep_min_fold_positive_fraction" and not (0.0 < x <= 1.0):
            errs.append("sweep_min_fold_positive_fraction_not_positive_unit")
    return errs


def northset_queue_imbalance_mean_alias_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: ``queue_imbalance_mean`` ∈ [-1, 1] when finite.

    Alias companion to ``mean_queue_imbalance`` / session queue stamps — same
    signed-unit contract on the northset receipt key name.
    """
    if not isinstance(blob, dict):
        return []
    raw = blob.get("queue_imbalance_mean")
    if raw is None:
        return []
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return ["queue_imbalance_mean_non_numeric"]
    if val != val:
        return []
    if not (-1.0 <= val <= 1.0):
        return ["queue_imbalance_mean_out_of_signed_unit"]
    return []


def candle_spread_alias_honesty_errors(blob: object) -> list[str]:
    """Soft-verify candle spread alias identities + nonneg when stamped.

    - each of quoted / effective / half / spread_bps / half_bps ≥ 0 when finite
    - ``mean_quoted_spread ≈ mean_effective_spread`` when both finite
    - ``mean_half_spread ≈ 0.5 * mean_quoted_spread``
    - ``mean_spread_bps ≈ 2 * mean_half_spread_bps``
    - ``mean_spread_bps ≈ 1e4 * mean_spread_over_mid`` when both finite
      (book_metrics: spread_bps = 1e4 * spread/mid; spread_over_mid = spread/mid)
    Never invent missing keys. Research diagnostic only; never live Sharpe.
    Off kyle invent / IC↔gap mesh.
    """
    if not isinstance(blob, dict):
        return []
    if blob.get("family") != "candle_order_book":
        return []
    errs: list[str] = []
    for key in (
        "mean_quoted_spread",
        "mean_effective_spread",
        "mean_half_spread",
        "mean_half_spread_bps",
        "mean_spread_bps",
    ):
        if key not in blob:
            continue
        try:
            x = float(blob.get(key))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append(f"{key}_non_numeric")
            continue
        if x != x:
            continue
        if abs(x) == float("inf") or x < 0.0:
            errs.append(f"{key}_negative_or_non_finite")
    qe = _finite_pair(blob, "mean_quoted_spread", "mean_effective_spread")
    if qe is not None:
        quoted, effective = qe
        if not math.isclose(
            quoted,
            effective,
            rel_tol=_NORTHSET_SPREAD_REL_TOL,
            abs_tol=_NORTHSET_SPREAD_ABS_TOL,
        ):
            errs.append("mean_quoted_spread_not_equal_mean_effective_spread")
    errs.extend(northset_half_spread_honesty_errors(blob))
    errs.extend(northset_spread_bps_honesty_errors(blob))
    # spread_bps = 1e4 * spread_over_mid by construction (same mid)
    pair_bps_mid = _finite_pair(blob, "mean_spread_bps", "mean_spread_over_mid")
    if pair_bps_mid is not None:
        spread_bps, over_mid = pair_bps_mid
        if not math.isclose(
            spread_bps,
            1e4 * over_mid,
            rel_tol=_NORTHSET_SPREAD_REL_TOL,
            abs_tol=max(_NORTHSET_SPREAD_ABS_TOL, 1e-6),
        ):
            errs.append("mean_spread_bps_not_1e4_times_mean_spread_over_mid")
    return errs


def candle_spread_bps_ic_matches_spread_over_mid_ic_honesty_errors(blob: object) -> list[str]:
    """Soft-verify candle ``ic_spread_bps`` ≈ ``ic_spread_over_mid`` when both finite.

    book_metrics: ``spread_bps = 1e4 * spread_over_mid`` — a positive scale, so
    date-level Spearman (and Pearson) ICs must match. Complements the mean-side
    ``mean_spread_bps ≈ 1e4 * mean_spread_over_mid`` check in
    :func:`candle_spread_alias_honesty_errors`.

    When both IC keys are finite:
    - bare Spearman ICs must be isclose
    - ``ic_*_pearson`` companions must be isclose when both finite

    Skip absent / non-finite either side. Candle family only.
    Research diagnostic only; never live Sharpe. Off sibling invent / half_spread
    Sergeant lane / kyle invent.
    """
    if not isinstance(blob, dict):
        return []
    if blob.get("family") not in (None, "candle_order_book"):
        return []
    errs: list[str] = []

    def _finite(key: str) -> float | None:
        if key not in blob:
            return None
        try:
            x = float(blob.get(key))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append(f"{key}_non_numeric")
            return None
        if x != x or abs(x) == float("inf"):
            return None
        return x

    spearman_bps = _finite("ic_spread_bps")
    spearman_mid = _finite("ic_spread_over_mid")
    if (
        spearman_bps is not None
        and spearman_mid is not None
        and not math.isclose(spearman_bps, spearman_mid, rel_tol=1e-9, abs_tol=1e-12)
    ):
        errs.append("ic_spread_bps_diverges_from_ic_spread_over_mid")

    pearson_bps = _finite("ic_spread_bps_pearson")
    pearson_mid = _finite("ic_spread_over_mid_pearson")
    if (
        pearson_bps is not None
        and pearson_mid is not None
        and not math.isclose(pearson_bps, pearson_mid, rel_tol=1e-9, abs_tol=1e-12)
    ):
        errs.append("ic_spread_bps_pearson_diverges_from_ic_spread_over_mid_pearson")
    return errs


def candle_frac_and_spread_x_honesty_errors(blob: object) -> list[str]:
    """Soft-verify candle structure fracs + spread×range companion.

    When stamped on ``candle_order_book`` (or family absent):
    - ``mean_candle_range_frac`` / ``mean_candle_body_frac`` ∈ [0, 1] when finite
    - ``mean_imbalance_x_body_frac`` ∈ [-1, 1] when finite
    - ``mean_spread_x_range`` ≥ 0 when finite
    NaN skipped. Research diagnostic only; never live Sharpe. Off kyle invent.
    """
    if not isinstance(blob, dict):
        return []
    fam = blob.get("family")
    if fam not in (None, "candle_order_book"):
        return []
    errs: list[str] = []
    for key in ("mean_candle_range_frac", "mean_candle_body_frac"):
        if key not in blob:
            continue
        try:
            x = float(blob.get(key))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append(f"{key}_non_numeric")
            continue
        if x != x:
            continue
        if abs(x) == float("inf") or not (0.0 <= x <= 1.0):
            errs.append(f"{key}_out_of_unit_interval")
    if "mean_imbalance_x_body_frac" in blob:
        try:
            x = float(blob.get("mean_imbalance_x_body_frac"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append("mean_imbalance_x_body_frac_non_numeric")
        else:
            if x == x and abs(x) != float("inf") and not (-1.0 <= x <= 1.0):
                errs.append("mean_imbalance_x_body_frac_out_of_signed_unit")
            elif x == x and abs(x) == float("inf"):
                errs.append("mean_imbalance_x_body_frac_non_finite")
    if "mean_spread_x_range" in blob:
        try:
            x = float(blob.get("mean_spread_x_range"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append("mean_spread_x_range_non_numeric")
        else:
            if x == x and abs(x) == float("inf"):
                errs.append("mean_spread_x_range_non_finite")
            elif x == x and x < 0.0:
                errs.append("mean_spread_x_range_negative")
    return errs


def candle_direction_mean_honesty_errors(blob: object) -> list[str]:
    """Soft-verify ``mean_candle_direction`` ∈ [-1, 1] when finite.

    FEATURE_COLS ``candle_direction`` is ternary ∈ {-1, 0, 1}; the receipt mean
    must therefore lie in [-1, 1]. NaN/absent skip. Research diagnostic only.
    Off kyle invent.
    """
    if not isinstance(blob, dict):
        return []
    fam = blob.get("family")
    if fam not in (None, "candle_order_book"):
        return []
    if "mean_candle_direction" not in blob:
        return []
    try:
        x = float(blob.get("mean_candle_direction"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return ["mean_candle_direction_non_numeric"]
    if x != x:
        return []
    if abs(x) == float("inf") or not (-1.0 <= x <= 1.0):
        return ["mean_candle_direction_out_of_signed_unit"]
    return []


def candle_wick_skew_and_body_ret_means_honesty_errors(blob: object) -> list[str]:
    """Soft-verify unbounded candle FEATURE_COLS means finite when stamped.

    Keys: ``mean_wick_skew``, ``mean_candle_body_ret``, ``mean_signed_vol_x_imbalance``.
    Present numeric values must not be ±inf (NaN skip). Research diagnostic only;
    never live Sharpe. Off kyle invent.
    """
    if not isinstance(blob, dict):
        return []
    if blob.get("family") not in (None, "candle_order_book"):
        return []
    errs: list[str] = []
    for key in (
        "mean_wick_skew",
        "mean_candle_body_ret",
        "mean_signed_vol_x_imbalance",
    ):
        if key not in blob:
            continue
        try:
            x = float(blob.get(key))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append(f"{key}_non_numeric")
            continue
        if x != x:
            continue
        if abs(x) == float("inf"):
            errs.append(f"{key}_non_finite")
    return errs


def candle_signed_vol_x_imbalance_mean_honesty_errors(blob: object) -> list[str]:
    """Alias: soft-verify ``mean_signed_vol_x_imbalance`` finite via the finite-means pack."""
    if not isinstance(blob, dict):
        return []
    slim = (
        {
            "family": blob.get("family"),
            "mean_signed_vol_x_imbalance": blob["mean_signed_vol_x_imbalance"],
        }
        if "mean_signed_vol_x_imbalance" in blob
        else {"family": blob.get("family")}
    )
    return candle_wick_skew_and_body_ret_means_honesty_errors(slim)


def northset_microprice_weight_balance_honesty_errors(blob: object) -> list[str]:
    """Northset-named entry point for mean microprice weight-balance honesty.

    Identical contract to :func:`mean_microprice_weight_balance_honesty_errors`
    (∈ [0, 1] when finite; missing / NaN → skip). Research diagnostic only.
    """
    return mean_microprice_weight_balance_honesty_errors(blob)


def northset_spread_bps_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: ``mean_spread_bps ≈ 2 * mean_half_spread_bps`` when both finite.

    ``half_spread_bps = 1e4 * half_spread / mid`` and ``spread_bps = 2 * half_spread_bps``
    by construction; a receipt violating that is inconsistent. Missing /
    non-finite either side → skip. Research diagnostic only; never live Sharpe.
    """
    pair = _finite_pair(blob, "mean_spread_bps", "mean_half_spread_bps")
    if pair is None:
        return []
    spread_bps, half_bps = pair
    if not math.isclose(
        spread_bps,
        2.0 * half_bps,
        rel_tol=_NORTHSET_SPREAD_REL_TOL,
        abs_tol=_NORTHSET_SPREAD_ABS_TOL,
    ):
        return ["mean_spread_bps_not_double_mean_half_spread_bps"]
    return []


def northset_spread_receipt_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: ``mean_effective_spread ≈ mean_quoted_spread`` when both finite.

    Book ``effective_spread`` is an alias of the quoted spread
    (best_ask − best_bid); the candle diagnostic is ``mean_close_mid_abs_rel``
    only, never a second "effective" spread. Missing / non-finite either side
    → skip (thin external book panels stamp NaN). Research diagnostic only;
    never live Sharpe / promotion.
    """
    pair = _finite_pair(blob, "mean_effective_spread", "mean_quoted_spread")
    if pair is None:
        return []
    effective, quoted = pair
    if not math.isclose(
        effective,
        quoted,
        rel_tol=_NORTHSET_SPREAD_REL_TOL,
        abs_tol=_NORTHSET_SPREAD_ABS_TOL,
    ):
        return ["mean_effective_spread_diverges_from_mean_quoted_spread"]
    return []


def _finite_pair(blob: object, key_a: str, key_b: str) -> tuple[float, float] | None:
    """Return finite ``(a, b)`` for two blob keys, or None to skip soft-verify.

    Missing / non-numeric / NaN / ±inf on either side → None (never invent a
    comparison from absent evidence).
    """
    if not isinstance(blob, dict):
        return None
    a = blob.get(key_a)
    b = blob.get(key_b)
    if not _finite_scalar(a) or not _finite_scalar(b):
        return None
    return float(a), float(b)  # type: ignore[arg-type]


def northset_structure_finite_rate_distinct_from_candle_honesty_errors(blob: object) -> list[str]:
    """Soft-verify northset ``structure_finite_rate`` is distinct from candle companions.

    Northset aggregates concentration_top / queue_priority / side_notional /
    tob_size_share finite rates — never candle ``finite_rate_*`` keys
    (microprice_minus_mid / size concentration). When any northset companion
    rate is present with ``structure_finite_rate``:

    - candle ``finite_rate_*`` keys must be absent (family mix = dishonest)
    - if all four companions + aggregate are finite, aggregate ≈ nanmean(companions)

    NaN/absent skip. Research diagnostic only; never live Sharpe. Off kyle / METRICS_*.
    """
    if not isinstance(blob, dict):
        return []
    northset_companions = (
        "concentration_top_finite_rate",
        "queue_priority_finite_rate",
        "side_notional_finite_rate",
        "tob_size_share_finite_rate",
    )
    candle_companions = (
        "finite_rate_microprice_minus_mid",
        "finite_rate_bid_size_concentration_top",
        "finite_rate_ask_size_concentration_top",
    )
    has_northset = any(k in blob for k in northset_companions)
    if not has_northset or "structure_finite_rate" not in blob:
        return []
    errs: list[str] = []
    for key in candle_companions:
        if key in blob:
            errs.append("northset_structure_finite_rate_mixed_with_candle_finite_rate_companions")
            break
    vals: list[float] = []
    for key in northset_companions:
        if key not in blob:
            continue
        try:
            x = float(blob.get(key))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if x != x or abs(x) == float("inf"):
            continue
        vals.append(x)
    try:
        agg = float(blob.get("structure_finite_rate"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return errs
    if agg != agg or abs(agg) == float("inf"):
        return errs
    if len(vals) == 4:
        import math

        expected = sum(vals) / 4.0
        if not math.isclose(agg, expected, rel_tol=1e-9, abs_tol=1e-12):
            errs.append("northset_structure_finite_rate_not_nanmean_of_companion_rates")
    return errs


def candle_structure_finite_rate_covers_companions_honesty_errors(blob: object) -> list[str]:
    """Soft-verify candle ``structure_finite_rate`` ≈ nanmean of finite_rate companions.

    Bench stamps ``structure_finite_rate`` via nanmean of:
    ``finite_rate_microprice_minus_mid``,
    ``finite_rate_bid_size_concentration_top``,
    ``finite_rate_ask_size_concentration_top``.

    When the aggregate and all three companions are finite, require equality.
    Partial / NaN companions → skip (thin books). Candle family only.
    Research diagnostic only; never live Sharpe. Off kyle invent / METRICS_*.
    """
    if not isinstance(blob, dict):
        return []
    fam = blob.get("family")
    if fam not in (None, "candle_order_book"):
        return []
    companions = (
        "finite_rate_microprice_minus_mid",
        "finite_rate_bid_size_concentration_top",
        "finite_rate_ask_size_concentration_top",
    )
    if "structure_finite_rate" not in blob:
        return []
    if not all(k in blob for k in companions):
        return []
    try:
        agg = float(blob.get("structure_finite_rate"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return ["structure_finite_rate_non_numeric"]
    if agg != agg or abs(agg) == float("inf"):
        return []
    vals: list[float] = []
    for key in companions:
        try:
            x = float(blob.get(key))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return []
        if x != x or abs(x) == float("inf"):
            return []
        vals.append(x)
    import math

    expected = sum(vals) / float(len(vals))
    if not math.isclose(agg, expected, rel_tol=1e-9, abs_tol=1e-12):
        return ["candle_structure_finite_rate_not_nanmean_of_finite_rate_companions"]
    return []


def northset_queue_priority_le_size_concentration_honesty_errors(blob: object) -> list[str]:
    """Soft-verify queue priority means do not exceed same-side size concentration.

    Book identity: queue_priority ≤ size_concentration_top on each side when both
    finite (touch share of depth ≤ top-level concentration). Receipt means:

    - ``mean_queue_priority_proxy`` ≤ ``mean_bid_size_concentration_top``
    - ``mean_ask_queue_priority_proxy`` ≤ ``mean_ask_size_concentration_top``

    Skip pairs where either key absent/non-finite. Research diagnostic only;
    never live Sharpe. Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    errs: list[str] = []
    pairs = (
        (
            "mean_queue_priority_proxy",
            "mean_bid_size_concentration_top",
            "mean_queue_priority_proxy_gt_mean_bid_size_concentration_top",
        ),
        (
            "mean_ask_queue_priority_proxy",
            "mean_ask_size_concentration_top",
            "mean_ask_queue_priority_proxy_gt_mean_ask_size_concentration_top",
        ),
    )
    for q_key, c_key, err in pairs:
        if q_key not in blob or c_key not in blob:
            continue
        try:
            q = float(blob.get(q_key))  # type: ignore[arg-type]
            c = float(blob.get(c_key))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if q != q or c != c or abs(q) == float("inf") or abs(c) == float("inf"):
            continue
        if q > c + 1e-9:
            errs.append(err)
    return errs


def northset_queue_priority_bid_ask_pair_honesty_errors(blob: object) -> list[str]:
    """Soft-verify ask queue priority is a separate companion from bid/proxy.

    When ``mean_queue_priority_proxy`` is present and finite, require
    ``mean_ask_queue_priority_proxy`` also present (missing ask = dishonest
    collapse). Both ∈ [0, 1] when finite (delegates bounds to
    mean_queue_priority_honesty_errors). Never force equality between bid and
    ask means (Jensen / side asymmetry OK). Research diagnostic only.
    Off kyle_ofi / METRICS_*.
    """
    if not isinstance(blob, dict):
        return []
    if "mean_queue_priority_proxy" not in blob:
        return []
    try:
        bid = float(blob.get("mean_queue_priority_proxy"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return []
    if bid != bid or abs(bid) == float("inf"):
        return []
    if "mean_ask_queue_priority_proxy" not in blob:
        return ["mean_ask_queue_priority_proxy_missing_while_bid_proxy_finite"]
    ask = blob.get("mean_ask_queue_priority_proxy")
    if not _finite_scalar(ask):
        # Present but non-finite is inconsistent, not a skip: the bid proxy is
        # finite, so the ask companion must be finite too.
        return ["mean_ask_queue_priority_proxy_nan_while_bid_proxy_finite"]
    return []


def candle_mwb_scored_implies_mean_unit_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: when MWB IC is scored on candle, mean MWB ∈ [0, 1].

    Fuse path contract: ``ic_microprice_weight_balance`` present ⇒
    ``mean_microprice_weight_balance`` present and ∈ [0, 1] when finite.
    Never equate IC to the mean. NaN mean with finite IC is dishonest.
    Research diagnostic only; never live Sharpe. Off kyle_ofi / METRICS_*.
    """
    if not isinstance(blob, dict):
        return []
    if blob.get("family") != "candle_order_book":
        return []
    if "ic_microprice_weight_balance" not in blob:
        return []
    if "mean_microprice_weight_balance" not in blob:
        return ["mean_microprice_weight_balance_missing_while_mwb_ic_scored"]
    try:
        m = float(blob.get("mean_microprice_weight_balance"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return ["mean_microprice_weight_balance_non_numeric"]
    if m != m:
        return ["mean_microprice_weight_balance_nan_while_mwb_ic_scored"]
    if abs(m) == float("inf") or not (0.0 <= m <= 1.0):
        return ["mean_microprice_weight_balance_out_of_unit_interval"]
    return []


def candle_microprice_weight_balance_ic_honesty_errors(blob: object) -> list[str]:
    """Soft-verify ic_microprice_weight_balance companions when scored.

    When ``ic_microprice_weight_balance`` is present: finite-when-present for IC/t;
    ``ic_microprice_weight_balance_p`` ∈ [0, 1]; ``_n_dates`` ≥ 0. Never equate
    to mean_microprice_weight_balance (mean ≠ IC). Research diagnostic only.
    """
    if not isinstance(blob, dict):
        return []
    if "ic_microprice_weight_balance" not in blob:
        return []
    # Reuse pattern via the general IC helper for this key slice
    slice_blob = {
        k: v
        for k, v in blob.items()
        if isinstance(k, str) and k.startswith("ic_microprice_weight_balance")
    }
    return candle_feature_cols_ic_honesty_errors(slice_blob)


def candle_feature_cols_ic_completeness_honesty_errors(blob: object) -> list[str]:
    """Soft-verify candle FEATURE_COLS IC keys are present when the family scored.

    When ``family == "candle_order_book"`` and (``n_scored`` > 0 or any spearman
    ``ic_*`` key is present), every column in ``FEATURE_COLS`` must expose the
    Spearman/Pearson companion pack: ``ic_<col>``, ``ic_<col>_pearson``,
    ``ic_<col>_t``, ``ic_<col>_p``, and ``ic_<col>_n_dates`` (values may be NaN).
    Skip other families. Research diagnostic only; never live Sharpe.
    Off kyle_ofi / METRICS_*.
    """
    if not isinstance(blob, dict):
        return []
    if blob.get("family") != "candle_order_book":
        return []
    try:
        from quant_fund.microstructure.bench import FEATURE_COLS
    except Exception:
        return []
    n_scored = blob.get("n_scored")
    try:
        scored = float(n_scored) if n_scored is not None else float("nan")
    except (TypeError, ValueError):
        scored = float("nan")
    has_ic = any(
        isinstance(k, str)
        and k.startswith("ic_")
        and not k.endswith(("_t", "_p", "_n_dates", "_pearson"))
        and k != "ic_method"
        for k in blob
    )
    if not (has_ic or (scored == scored and scored > 0.0)):
        return []
    errs: list[str] = []
    for col in FEATURE_COLS:
        for suf in ("", "_pearson", "_t", "_p", "_n_dates"):
            key = f"ic_{col}{suf}"
            if key not in blob:
                errs.append(f"{key}_missing_from_candle_feature_cols_receipt")
    return errs


def candle_feature_cols_ic_honesty_errors(blob: object) -> list[str]:
    """Soft-verify candle_order_book FEATURE_COLS IC receipt companions.

    When present on a candle family blob:
    - ``ic_*_p`` ∈ [0, 1] when finite (Spearman HAC p-values)
    - ``ic_*_n_dates`` ≥ 0 when finite
    - bare ``ic_<feature>`` / ``ic_*_t`` / ``ic_*_pearson`` finite-when-present (signed OK; ±inf fail)
    - ``mean_abs_ic`` ∈ [0, 1] when finite
    - bare ``ic_*`` / ``ic_*_pearson`` / ``best_feature_ic`` ∈ [-1, 1] when finite
    - ``best_feature_ic_key`` / ``best_feature_ic`` pair identity: nonempty key requires
      finite matching value (max |spearman|); finite value requires nonempty key
    - ``mean_abs_ic`` ≈ mean(|ic_*| spearman) when both present
    - ``|best_feature_ic|`` ≥ ``mean_abs_ic`` when both finite (best covers mean;
      still holds when feature ``ic_*`` keys are absent)

    Never equate with northset ofi/microprice/clv IC keys. NaN/absent skip.
    Research diagnostic only; never live Sharpe. Off kyle_ofi / METRICS_*.
    """
    if not isinstance(blob, dict):
        return []
    errs: list[str] = []
    spearman: dict[str, float] = {}

    for key, raw in blob.items():
        if not isinstance(key, str) or not key.startswith("ic_"):
            continue
        try:
            x = float(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            # Skip non-numeric meta (e.g. ic_method=date_level_spearman_hac).
            continue
        if x != x:
            continue
        if abs(x) == float("inf"):
            errs.append(f"{key}_non_finite_fail_closed")
            continue
        if key.endswith("_p"):
            if not (0.0 <= x <= 1.0):
                errs.append(f"{key}_out_of_unit_interval")
        elif key.endswith("_n_dates"):
            if x < 0.0:
                errs.append(f"{key}_negative")
        elif key.endswith(("_t", "_pearson")):
            # pearson correlation ∈ [-1, 1]; t may be unbounded
            if key.endswith("_pearson") and not (-1.0 <= x <= 1.0):
                errs.append(f"{key}_out_of_unit_interval")
        else:
            # bare ic_<feature> spearman mean ∈ [-1, 1]
            if not (-1.0 <= x <= 1.0):
                errs.append(f"{key}_out_of_unit_interval")
            else:
                spearman[key] = x

    if "mean_abs_ic" in blob:
        try:
            mai = float(blob.get("mean_abs_ic"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append("mean_abs_ic_non_numeric")
        else:
            if mai == mai and abs(mai) != float("inf"):
                if mai < 0.0:
                    errs.append("mean_abs_ic_negative")
                elif mai > 1.0:
                    errs.append("mean_abs_ic_out_of_unit_interval")
            elif mai == mai and abs(mai) == float("inf"):
                errs.append("mean_abs_ic_non_finite_fail_closed")

    import math

    best_key = blob.get("best_feature_ic_key")
    has_best_key = best_key is not None and best_key != ""
    has_best_ic = "best_feature_ic" in blob and blob.get("best_feature_ic") is not None

    if has_best_key:
        if (
            not isinstance(best_key, str)
            or not best_key.startswith("ic_")
            or best_key.endswith(("_t", "_p", "_n_dates", "_pearson"))
        ):
            errs.append("best_feature_ic_key_not_ic_spearman_key")
        elif best_key not in blob:
            errs.append("best_feature_ic_key_missing_from_receipt")
        if not has_best_ic:
            errs.append("best_feature_ic_missing_despite_best_feature_ic_key")
        else:
            try:
                best_val = float(blob.get("best_feature_ic"))  # type: ignore[arg-type]
            except (TypeError, ValueError):
                errs.append("best_feature_ic_non_numeric")
            else:
                if best_val != best_val:
                    errs.append("best_feature_ic_nan_despite_best_feature_ic_key")
                elif abs(best_val) == float("inf"):
                    errs.append("best_feature_ic_non_finite_fail_closed")
                elif not (-1.0 <= best_val <= 1.0) or not (-1.0 <= best_val <= 1.0):
                    errs.append("best_feature_ic_out_of_unit_interval")
                elif spearman:
                    max_abs = max(abs(v) for v in spearman.values())
                    if not math.isclose(abs(best_val), max_abs, rel_tol=1e-9, abs_tol=1e-12):
                        errs.append("best_feature_ic_not_max_abs_spearman_ic")
                    keyed = spearman.get(best_key) if isinstance(best_key, str) else None
                    if keyed is None or not math.isclose(
                        best_val, keyed, rel_tol=1e-9, abs_tol=1e-12
                    ):
                        errs.append("best_feature_ic_mismatch_best_feature_ic_key")
    elif has_best_ic:
        try:
            best_val = float(blob.get("best_feature_ic"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append("best_feature_ic_non_numeric")
        else:
            if best_val == best_val and abs(best_val) != float("inf"):
                errs.append("best_feature_ic_key_missing_despite_best_feature_ic")
            elif best_val == best_val and abs(best_val) == float("inf"):
                errs.append("best_feature_ic_non_finite_fail_closed")

    # mean_abs_ic identity: ≈ mean(|spearman ic_*|) when both sides present
    if "mean_abs_ic" in blob and spearman:
        try:
            mai = float(blob.get("mean_abs_ic"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            pass  # already flagged non-numeric above
        else:
            if mai == mai and abs(mai) != float("inf"):
                expected = sum(abs(v) for v in spearman.values()) / len(spearman)
                if not math.isclose(mai, expected, rel_tol=1e-9, abs_tol=1e-12):
                    errs.append("mean_abs_ic_not_mean_abs_spearman_ic")

    # |best_feature_ic| covers mean_abs_ic even when feature ic_* keys absent
    if "best_feature_ic" in blob and "mean_abs_ic" in blob:
        try:
            best_val = float(blob.get("best_feature_ic"))  # type: ignore[arg-type]
            mai = float(blob.get("mean_abs_ic"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            pass  # non-numeric already flagged above when applicable
        else:
            if (
                best_val == best_val
                and mai == mai
                and abs(best_val) != float("inf")
                and abs(mai) != float("inf")
                and abs(best_val) + 1e-12 < mai
            ):
                errs.append("best_feature_ic_abs_lt_mean_abs_ic")

    return errs


_CANDLE_FEATURE_IC_METHOD_ALLOWED = frozenset({"date_level_spearman_hac"})


def _candle_has_feature_ic_marker(blob: dict) -> bool:
    """True if any FEATURE_COLS-style ic_<col> spearman (not meta suffix) is present."""
    for key in blob:
        if not isinstance(key, str) or not key.startswith("ic_"):
            continue
        if key == "ic_method":
            continue
        if key.endswith(("_t", "_p", "_n_dates", "_pearson")):
            continue
        return True
    return False


def candle_order_book_dgp_data_source_honesty_errors(blob: object) -> list[str]:
    """Soft-verify candle_order_book ``dgp`` / ``book_dgp`` ↔ ``data_source``.

    Parallel to always-on northset helper — never equate the two surfaces.
    When present on candle family blob:

    - ``dgp`` and ``book_dgp`` both present → must match
    - ``data_source == "SYNTHETIC"`` ⇒ present dgp fields are ``synthetic_lob``
    - ``book_dgp``/``dgp`` ``synthetic_lob`` ⇒ ``data_source`` is ``SYNTHETIC`` when set
    - non-synthetic dgp + ``data_source`` set ⇒ not ``SYNTHETIC``; if ``book_source``
      also set, ``data_source == book_source``

    Skip non-candle families / all three absent. Research diagnostic only; never
    live Sharpe. Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    if blob.get("family") not in (None, "candle_order_book"):
        return []
    # require candle marker or explicit family
    if (
        blob.get("family") is None
        and "best_feature_ic" not in blob
        and "mean_abs_ic" not in blob
        and not any(k in blob for k in ("finite_rate_microprice_minus_mid", "ic_method"))
    ):
        return []

    malformed: list[str] = []

    def _str(key: str) -> str | None:
        if key not in blob or blob.get(key) is None:
            return None
        val = blob.get(key)
        if not isinstance(val, str):
            # Present-but-non-string provenance must not read as absent.
            malformed.append(f"{key}_non_str")
            return None
        s = val.strip()
        return s if s else None

    book_dgp = _str("book_dgp")
    dgp = _str("dgp")
    data_source = _str("data_source")
    book_source = _str("book_source")
    if book_dgp is None and dgp is None and data_source is None and not malformed:
        return []

    errs: list[str] = list(malformed)
    if book_dgp is not None and dgp is not None and book_dgp != dgp:
        errs.append("candle_dgp_book_dgp_mismatch")

    synth_dgp: bool | None = None
    if book_dgp is not None:
        synth_dgp = book_dgp == "synthetic_lob"
    elif dgp is not None:
        synth_dgp = dgp == "synthetic_lob"

    if synth_dgp is True and data_source is not None and data_source != "SYNTHETIC":
        errs.append("candle_synthetic_dgp_data_source_not_SYNTHETIC")
    if data_source == "SYNTHETIC":
        if book_dgp is not None and book_dgp != "synthetic_lob":
            errs.append("candle_SYNTHETIC_data_source_book_dgp_not_synthetic_lob")
        if dgp is not None and dgp != "synthetic_lob":
            errs.append("candle_SYNTHETIC_data_source_dgp_not_synthetic_lob")
        if book_dgp is None and dgp is None:
            errs.append("candle_SYNTHETIC_data_source_missing_dgp")
    if synth_dgp is False and data_source is not None:
        if data_source == "SYNTHETIC":
            errs.append("candle_nonsynthetic_dgp_data_source_SYNTHETIC")
        elif book_source is not None and data_source != book_source:
            errs.append("candle_nonsynthetic_data_source_ne_book_source")
    return errs


def candle_order_book_family_provenance_honesty_errors(blob: object) -> list[str]:
    """Soft-verify candle_order_book ``family`` / ``book_source`` / ``label`` stamps.

    When present on a candle family blob:
    - ``family`` == ``candle_order_book``
    - ``book_source`` nonempty string
    - ``label`` nonempty string

    Skip non-candle families. Never equate with northset / kyle nest twins.
    Research diagnostic only; never live Sharpe. Off kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    fam = blob.get("family") if "family" in blob else None
    if fam is not None and fam != "candle_order_book":
        return []
    # If family absent, only run when candle markers present
    if (
        fam is None
        and not any(
            k in blob
            for k in ("best_feature_ic", "mean_abs_ic", "finite_rate_microprice_minus_mid")
        )
        and "book_source" not in blob
        and "label" not in blob
    ):
        return []

    errs: list[str] = []
    if (
        "family" in blob
        and blob.get("family") is not None
        and (blob.get("family") != "candle_order_book")
    ):
        errs.append("candle_family_invalid")
    if "book_source" in blob and blob.get("book_source") is not None:
        bs = blob.get("book_source")
        if not isinstance(bs, str) or not bs.strip():
            errs.append("candle_book_source_empty_or_not_str")
    if "label" in blob and blob.get("label") is not None:
        lab = blob.get("label")
        if not isinstance(lab, str) or not lab.strip():
            errs.append("candle_label_empty_or_not_str")
    return errs


def candle_order_book_sizing_honesty_errors(blob: object) -> list[str]:
    """Soft-verify candle_order_book sizing stamps: min_names / depth / n_bars / n_fused / n_scored.

    Candle bench stamps these (CLI echoes n_fused + min_names). When present:
    - ``min_names`` finite integer ≥ 1
    - ``depth`` finite integer ≥ 1 (LOB levels used by attach)
    - ``n_bars`` / ``n_fused`` / ``n_scored`` non-negative integers (NaN skip per key)
    - ``n_scored ≤ n_fused ≤ n_bars`` when pairs present

    Candle-family only (skip other families). Never equate with northset / kyle nest
    twins. Research diagnostic only; never live Sharpe. Off kyle_ofi / METRICS_*.
    """
    if not isinstance(blob, dict):
        return []
    fam = blob.get("family")
    if fam is not None and fam != "candle_order_book":
        return []

    errs: list[str] = []

    if "min_names" in blob and blob.get("min_names") is not None:
        try:
            mn = float(blob.get("min_names"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append("candle_min_names_non_numeric")
        else:
            if mn != mn or abs(mn) == float("inf"):
                errs.append("candle_min_names_non_finite")
            elif mn < 1.0 or abs(mn - int(mn)) > 1e-9:
                errs.append("candle_min_names_lt_one_or_not_int")

    if "depth" in blob and blob.get("depth") is not None:
        try:
            d = float(blob.get("depth"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append("candle_depth_non_numeric")
        else:
            if d != d or abs(d) == float("inf"):
                errs.append("candle_depth_non_finite")
            elif d < 1.0 or abs(d - int(d)) > 1e-9:
                errs.append("candle_depth_lt_one_or_not_int")

    def _nonneg_int(key: str) -> int | None:
        if key not in blob:
            return None
        val = blob.get(key)
        try:
            x = float(val)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append(f"candle_{key}_non_numeric")
            return None
        if x != x:
            return None
        if abs(x) == float("inf") or x < 0 or abs(x - int(x)) > 1e-9:
            errs.append(f"candle_{key}_not_nonneg_int")
            return None
        return int(x)

    nb = _nonneg_int("n_bars")
    nf = _nonneg_int("n_fused")
    ns = _nonneg_int("n_scored")
    if nb is not None and ns is not None and ns > nb:
        errs.append("candle_n_scored_gt_n_bars")
    if nf is not None and ns is not None and ns > nf:
        errs.append("candle_n_scored_gt_n_fused")
    if nb is not None and nf is not None and nf > nb:
        errs.append("candle_n_fused_gt_n_bars")
    return errs


def candle_order_book_ic_method_honesty_errors(blob: object) -> list[str]:
    """Soft-verify candle_order_book ``ic_method`` for FEATURE_COLS date-IC / HAC.

    When any ``ic_<feature>`` spearman key is present:
    - ``ic_method`` must be present and ``date_level_spearman_hac``
    Finite ``ic_<feature>`` with companion ``ic_<feature>_n_dates`` present → n_dates ≥ 1.
    Missing nest / no IC markers → skip. Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    # Optional family gate: if family stamped, must be candle_order_book
    fam = blob.get("family")
    if fam is not None and fam != "candle_order_book":
        return []
    if not _candle_has_feature_ic_marker(blob):
        return []
    errors: list[str] = []
    method = blob.get("ic_method")
    if method is None or (isinstance(method, str) and not method.strip()):
        errors.append("candle_order_book_ic_method_missing")
    elif not isinstance(method, str) or method not in _CANDLE_FEATURE_IC_METHOD_ALLOWED:
        errors.append("candle_order_book_ic_method_invalid")
    for key, val in blob.items():
        if not isinstance(key, str) or not key.startswith("ic_"):
            continue
        if key.endswith(("_t", "_p", "_n_dates", "_pearson")):
            continue
        try:
            x = float(val)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if x != x:
            continue
        nd_key = f"{key}_n_dates"
        if nd_key not in blob:
            continue
        try:
            nd = float(blob.get(nd_key))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errors.append(f"{nd_key}_non_numeric")
            continue
        if nd != nd or nd < 1.0:
            errors.append(f"{nd_key}_lt_1")
    return errors


NORTHSET_RECEIPT_HONESTY_HELPERS: tuple = (
    mean_book_age_seconds_honesty_errors,
    book_age_seconds_honesty_errors,
    mean_microprice_minus_mid_honesty_errors,
    northset_log_size_slope_honesty_errors,
    northset_qlike_means_honesty_errors,
    northset_range_spread_honesty_errors,
    amihud_mean_honesty_errors,
    northset_queue_sweep_ofi_honesty_errors,
    northset_queue_priority_le_size_concentration_honesty_errors,
    northset_queue_priority_bid_ask_pair_honesty_errors,
    depth_shape_finite_rate_honesty_errors,
    northset_structure_finite_rate_distinct_from_candle_honesty_errors,
    candle_structure_finite_rate_covers_companions_honesty_errors,
    northset_overnight_rv_semi_honesty_errors,
    northset_book_shape_finite_rates_honesty_errors,
    session_volume_conservation_rate_honesty_errors,
    session_reconstructs_daily_rate_honesty_errors,
    session_ohlc_vs_reconstructs_never_equate_honesty_errors,
    session_ohlc_vs_volume_conservation_never_equate_honesty_errors,
    session_ohlc_vs_session_chain_never_equate_honesty_errors,
    session_volume_conservation_vs_reconstructs_never_equate_honesty_errors,
    session_chain_vs_session_identity_siblings_never_equate_honesty_errors,
    book_uncrossed_rate_honesty_errors,
    ohlc_identity_rate_honesty_errors,
    book_uncrossed_vs_imbalance_p_ic_never_equate_honesty_errors,
    ohlc_identity_vs_imbalance_p_ic_never_equate_honesty_errors,
    session_bulk_vpin_vs_siblings_never_equate_honesty_errors,
    ohlc_identity_vs_book_uncrossed_never_equate_honesty_errors,
    ohlc_identity_vs_gap_finite_never_equate_honesty_errors,
    gap_finite_rate_vs_book_uncrossed_never_equate_honesty_errors,
    session_ohlc_vs_gap_finite_never_equate_honesty_errors,
    session_ohlc_vs_book_uncrossed_never_equate_honesty_errors,
    book_uncrossed_vs_session_chain_never_equate_honesty_errors,
    imbalance_top_p_ic_vs_session_chain_never_equate_honesty_errors,
    imbalance_top_p_ic_vs_session_ohlc_never_equate_honesty_errors,
    imbalance_top_p_ic_vs_session_volume_conservation_never_equate_honesty_errors,
    book_uncrossed_vs_session_reconstructs_never_equate_honesty_errors,
    book_uncrossed_vs_volume_conservation_never_equate_honesty_errors,
    gap_finite_rate_vs_session_chain_never_equate_honesty_errors,
    gap_finite_rate_vs_session_volume_conservation_never_equate_honesty_errors,
    gap_finite_rate_vs_session_reconstructs_never_equate_honesty_errors,
    imbalance_top_p_ic_vs_gap_finite_never_equate_honesty_errors,
    imbalance_top_p_ic_vs_session_reconstructs_never_equate_honesty_errors,
    microprice_p_ic_vs_gap_finite_never_equate_honesty_errors,
    microprice_p_ic_vs_session_chain_never_equate_honesty_errors,
    clv_p_ic_vs_gap_finite_never_equate_honesty_errors,
    dm_split_vs_park_p_vs_gap_finite_never_equate_honesty_errors,
    sweep_follow_event_p_vs_gap_finite_never_equate_honesty_errors,
    sweep_reject_cost_adjusted_mean_bps_vs_gap_finite_never_equate_honesty_errors,
    sweep_reject_control_diff_p_vs_gap_finite_never_equate_honesty_errors,
    wick_skew_p_ic_vs_gap_finite_never_equate_honesty_errors,
    ofi_p_ic_vs_gap_finite_never_equate_honesty_errors,
    dm_gk_vs_park_p_vs_gap_finite_never_equate_honesty_errors,
    sweep_reject_event_p_vs_gap_finite_never_equate_honesty_errors,
    sweep_follow_placebo_p_vs_gap_finite_never_equate_honesty_errors,
    sweep_follow_fold_positive_fraction_vs_gap_finite_never_equate_honesty_errors,
    vpin_p_ic_vs_gap_finite_never_equate_honesty_errors,
    sweep_reject_signed_p_ic_vs_gap_finite_never_equate_honesty_errors,
    sweep_follow_signed_p_ic_vs_gap_finite_never_equate_honesty_errors,
    sweep_reject_control_diff_p_vs_sweep_follow_control_diff_p_never_equate_honesty_errors,
    sweep_reject_fold_positive_fraction_vs_sweep_follow_fold_positive_fraction_never_equate_honesty_errors,
    sweep_reject_event_p_vs_sweep_follow_event_p_never_equate_honesty_errors,
    sweep_reject_signed_p_ic_vs_sweep_follow_signed_p_ic_never_equate_honesty_errors,
    sweep_reject_cost_adjusted_mean_bps_vs_sweep_follow_cost_adjusted_mean_bps_never_equate_honesty_errors,
    sweep_reject_placebo_p_vs_sweep_follow_placebo_p_never_equate_honesty_errors,
    sweep_reject_placebo_p_vs_gap_finite_never_equate_honesty_errors,
    sweep_follow_cost_adjusted_mean_bps_vs_gap_finite_never_equate_honesty_errors,
    sweep_reject_fold_positive_fraction_vs_gap_finite_never_equate_honesty_errors,
    sweep_follow_control_diff_p_vs_gap_finite_never_equate_honesty_errors,
    session_book_vpin_p_ic_vs_gap_finite_never_equate_honesty_errors,
    ohlc_identity_vs_session_ohlc_never_equate_honesty_errors,
    ohlc_identity_vs_session_reconstructs_never_equate_honesty_errors,
    ohlc_identity_vs_session_chain_never_equate_honesty_errors,
    ohlc_identity_vs_volume_conservation_never_equate_honesty_errors,
    session_chain_rate_honesty_errors,
    session_mean_jump_ratio_honesty_errors,
    sweep_follow_signed_mean_ic_honesty_errors,
    sweep_reject_signed_mean_ic_honesty_errors,
    sweep_reject_event_mean_bps_honesty_errors,
    sweep_follow_event_mean_bps_honesty_errors,
    sweep_follow_cost_adjusted_mean_bps_honesty_errors,
    sweep_reject_cost_adjusted_mean_bps_honesty_errors,
    northset_n_bars_scored_honesty_errors,
    ofi_p_ic_honesty_errors,
    northset_session_ofi_sum_ic_honesty_errors,
    northset_vpin_sweep_fold_honesty_errors,
    microprice_p_ic_honesty_errors,
    clv_p_ic_honesty_errors,
    northset_product_stamp_honesty_errors,
    northset_impact_proxy_warning_honesty_errors,
    close_location_value_clv_alias_identity_honesty_errors,
    ofi_mean_ic_honesty_errors,
    northset_ofi_lag_ic_honesty_errors,
    northset_vpin_ic_pack_honesty_errors,
    northset_queue_imbalance_ic_honesty_errors,
    imbalance_top_mean_ic_honesty_errors,
    northset_microprice_bps_ic_pack_honesty_errors,
    northset_clv_ic_pack_honesty_errors,
    northset_imbalance_top_ic_pack_honesty_errors,
    northset_session_book_vpin_ic_pack_honesty_errors,
    northset_candle_body_ret_ic_pack_honesty_errors,
    northset_wick_skew_ic_pack_honesty_errors,
    northset_bid_log_size_slope_ic_pack_honesty_errors,
    northset_imbalance_depth_ic_pack_honesty_errors,
    northset_amihud_ic_pack_honesty_errors,
    amihud_mean_ic_vs_amihud_abs_mean_ic_never_equate_honesty_errors,
    northset_volume_over_range_ic_packs_honesty_errors,
    northset_sweep_signed_ic_packs_honesty_errors,
    northset_session_close_ic_packs_honesty_errors,
    northset_all_n_dates_nonneg_honesty_errors,
    northset_all_p_ic_unit_interval_honesty_errors,
    northset_receipt_dgp_data_source_honesty_errors,
    northset_receipt_string_enum_honesty_errors,
    northset_sweep_evidence_scope_honesty_errors,
    northset_dm_park_honesty_errors,
    northset_sweep_evidence_blob_honesty_errors,
    northset_all_rate_unit_honesty_errors,
    northset_all_share_unit_honesty_errors,
    northset_all_fraction_unit_honesty_errors,
    northset_sweep_fold_positive_rates_honesty_errors,
    northset_queue_imbalance_mean_alias_honesty_errors,
    northset_all_floor_unit_honesty_errors,
    northset_all_finite_rate_unit_honesty_errors,
    northset_all_mean_rank_ic_unit_honesty_errors,
    northset_all_mean_ic_finite_honesty_errors,
    northset_all_t_ic_finite_honesty_errors,
    book_hypothesis_eligible_honesty_errors,
    northset_family_book_source_honesty_errors,
    northset_price_return_basis_honesty_errors,
    northset_depth_honesty_errors,
    northset_component_sources_honesty_errors,
    northset_use_session_l2_gate_consistency_errors,
    northset_include_kyle_ofi_nest_presence_honesty_errors,
    northset_sweep_control_sample_adequate_honesty_errors,
    northset_shape_columns_ensured_book_panel_path_honesty_errors,
    northset_receipt_bool_flags_honesty_errors,
    northset_lag_corr_and_sweep_count_honesty_errors,
    mid_lag1_corr_vs_ofi_lag1_corr_never_equate_honesty_errors,
    northset_kyle_r2_unit_honesty_errors,
    kyle_r2_vs_kyle_ofi_r2_never_equate_honesty_errors,
    gap_finite_rate_honesty_errors,
    vpin_mean_honesty_errors,
    session_bulk_vpin_honesty_errors,
    northset_spread_means_honesty_errors,
    northset_session_identity_rates_honesty_errors,
    northset_session_l2_enforced_identity_rates_present_honesty_errors,
    northset_half_spread_honesty_errors,
    northset_microprice_weight_balance_honesty_errors,
    northset_spread_bps_honesty_errors,
    northset_spread_receipt_honesty_errors,
)


def northset_receipt_honesty_errors(blob: object) -> list[str]:
    """Dispatcher: northset receipt soft-verify helpers not covered by session means.

    Fans out to rate/IC/spread/shape/sweep companions. Session path means stay in
    northset_session_means_honesty_errors. Kyle nest stays in kyle_ofi_nest_*.
    Research diagnostic only; never live Sharpe.
    """
    errs: list[str] = []
    for fn in NORTHSET_RECEIPT_HONESTY_HELPERS:
        errs.extend(fn(blob))
    return errs
