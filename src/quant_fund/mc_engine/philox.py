"""Counter-based Philox streams keyed by path index.

NumPy's Philox (Salmon et al., 2011, as exposed by ``numpy.random.Philox``)
is a counter-based generator: the output of a counter value does not depend
on how many previous draws some other worker happened to request.

Layout used here, written into the 4-word Philox counter before every path:

- word 0 starts at 0 and advances as ``random_raw`` consumes the path
- word 1 is the path index (or the antithetic base index)
- word 2 is a stream id
- word 3 stays 0

Stream ids ``1`` through ``15`` are reserved for this engine. Plug-in
scenario generators that draw their own Philox numbers with the same seed
should use ``stream_id >= USER_STREAM_ID_MIN`` so they do not collide with
engine shocks.

Normals use Box-Muller on ``(raw + 0.5) / 2**64`` uniforms. That consumes a
fixed number of words per normal. NumPy's ``Generator.standard_normal`` uses
ziggurat rejection and a data-dependent word count, which cannot be given a
fixed counter budget, so it is not used.
"""

from __future__ import annotations

import numpy as np
from numpy.random import Philox
from numpy.typing import NDArray

USER_STREAM_ID_MIN = 16
STREAM_SHOCK = 1
_U64_SCALE = 1.0 / 18446744073709551616.0  # 2**64
_EMPTY_BUFFER_POS = 4

UInt64Array = NDArray[np.uint64]
FloatArray = NDArray[np.float64]


def _as_indices(path_indices: NDArray[np.int64] | NDArray[np.uint64]) -> UInt64Array:
    arr = np.asarray(path_indices)
    if arr.ndim != 1:
        raise ValueError("path_indices must be a 1-d array")
    if arr.size == 0:
        return np.zeros(0, dtype=np.uint64)
    if np.issubdtype(arr.dtype, np.signedinteger):
        if int(arr.min()) < 0:
            raise ValueError("path_indices must be non-negative")
    elif not np.issubdtype(arr.dtype, np.unsignedinteger):
        raise ValueError("path_indices must be an integer array")
    return arr.astype(np.uint64, copy=False)


def philox_raw(
    seed: int,
    path_indices: NDArray[np.int64] | NDArray[np.uint64],
    n_draws: int,
    *,
    stream_id: int,
) -> UInt64Array:
    """``n_draws`` Philox words for each path index. Shape ``(n_paths, n_draws)``."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative int")
    if isinstance(stream_id, bool) or not isinstance(stream_id, int) or stream_id < 0:
        raise ValueError("stream_id must be a non-negative int")
    if isinstance(n_draws, bool) or not isinstance(n_draws, int) or n_draws < 1:
        raise ValueError("n_draws must be a positive int")
    indices = _as_indices(path_indices)
    out = np.empty((indices.size, n_draws), dtype=np.uint64)
    if indices.size == 0:
        return out
    bitgen = Philox(seed)
    template = bitgen.state
    counter0 = template["state"]["counter"].copy()
    stream = np.uint64(stream_id)
    for row, index in enumerate(indices.tolist()):
        state = bitgen.state
        counter = counter0.copy()
        counter[0] = np.uint64(0)
        counter[1] = np.uint64(index)
        counter[2] = stream
        counter[3] = np.uint64(0)
        state["state"]["counter"] = counter
        state["buffer_pos"] = _EMPTY_BUFFER_POS
        state["has_uint32"] = 0
        state["uinteger"] = 0
        bitgen.state = state
        out[row] = bitgen.random_raw(n_draws)
    return out


def philox_uniforms(
    seed: int,
    path_indices: NDArray[np.int64] | NDArray[np.uint64],
    n_draws: int,
    *,
    stream_id: int,
) -> FloatArray:
    """Uniforms in ``(0, 1)`` from Philox words. Shape ``(n_paths, n_draws)``."""
    raw = philox_raw(seed, path_indices, n_draws, stream_id=stream_id)
    uniforms = (raw.astype(np.float64) + 0.5) * _U64_SCALE
    return np.asarray(uniforms, dtype=np.float64)


def philox_normals(
    seed: int,
    path_indices: NDArray[np.int64] | NDArray[np.uint64],
    n_normals: int,
    *,
    stream_id: int,
) -> FloatArray:
    """Box-Muller normals. Shape ``(n_paths, n_normals)``.

    An odd ``n_normals`` draws one extra uniform pair and drops the spare
    coordinate, so the counter budget stays even.
    """
    if isinstance(n_normals, bool) or not isinstance(n_normals, int) or n_normals < 1:
        raise ValueError("n_normals must be a positive int")
    n_uniforms = n_normals + (n_normals & 1)
    raw = philox_raw(seed, path_indices, n_uniforms, stream_id=stream_id)
    if raw.shape[0] == 0:
        return np.zeros((0, n_normals), dtype=np.float64)
    uniforms = (raw.astype(np.float64) + 0.5) * _U64_SCALE
    left = uniforms[:, 0::2]
    right = uniforms[:, 1::2]
    radius = np.sqrt(-2.0 * np.log(left))
    theta = (2.0 * np.pi) * right
    paired = np.empty((raw.shape[0], n_uniforms), dtype=np.float64)
    paired[:, 0::2] = radius * np.cos(theta)
    paired[:, 1::2] = radius * np.sin(theta)
    return np.asarray(paired[:, :n_normals], dtype=np.float64)
