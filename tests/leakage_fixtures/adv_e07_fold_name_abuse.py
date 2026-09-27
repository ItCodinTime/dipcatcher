"""ADVERSARIAL §1a-E7 (POSITIVE, LH003): fold-NAMED function fitting a global.

The name `run_fold_preprocessing` is not a fold scope: the fit consumes the
module-level full sample, not a caller-sliced fold parameter.
"""

from __future__ import annotations

X_FULL = [[0.1, 0.2], [0.3, 0.4]]


def run_fold_preprocessing(scaler):
    scaler.fit(X_FULL)
    return scaler
