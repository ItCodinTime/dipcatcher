"""Soft key-presence batteries for VaR, CRPS e-process, and ES receipts.

Split out of the original module. Import the parent path; it re-exports these names.
"""

from __future__ import annotations

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


__all__ = [
    "DIST_CRPS_EPROCESS_REQUIRED_WHEN_DM",
    "DIST_CRPS_SCALED_EPROCESS_REQUIRED_WHEN_DM",
    "DM_CRPS_MARKER_KEY",
    "DM_CRPS_SCALED_MARKER_KEY",
    "ES_MARKER_KEYS",
    "KUPIEC_MARKER_KEYS",
    "TAIL_ES_BATTERY_REQUIRED_WHEN_ES_MARKERS",
    "TAIL_VAR_BATTERY_REQUIRED_WHEN_KUPIEC",
    "dist_crps_eprocess_keys_present",
    "dist_crps_eprocess_missing_keys",
    "tail_es_battery_keys_present",
    "tail_es_battery_missing_keys",
    "tail_var_battery_keys_present",
    "tail_var_battery_missing_keys",
]
