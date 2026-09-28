"""Global seed management. GPU determinism is not claimed."""

from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int) -> None:
    """Seed the process RNGs.

    PYTHONHASHSEED is read by the interpreter at startup, so setting it here
    only affects subprocesses spawned after this call — this interpreter's
    hash order is already fixed.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
