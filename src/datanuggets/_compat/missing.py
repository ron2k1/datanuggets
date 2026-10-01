"""Missing values and non-finite sums as R on x86 gives them, down to the NaN bits.

R's NA is a signaling NaN whose payload is 1954, and which NaN a long double sum ends on follows
the x87's rules, which _non_finite_sum emulates on the bits.
"""

from __future__ import annotations

import struct

import numpy as np

from datanuggets._compat import ldjit as _ldjit

_QUIET = 1 << 51  # the top fraction bit, set in a quiet NaN
_FRACTION = (1 << 52) - 1
_INDEFINITE = 0xFFF8000000000000  # the x87's default NaN, which Inf - Inf gives
_R_NAN = _ldjit.R_NAN  # the same bits as a float: what 0 / 0 gives R on x86
_NA_BITS = 0x7FF00000000007A2  # R's NA_real_: a signaling NaN whose payload is 1954


def _non_finite_sum(v, *, quiet: bool = False) -> float:
    """R's long double running sum of doubles, at least one of them Inf or NaN, down to which
    NaN it is. The finite ones can't overflow a long double, so only the rest decide, one x87
    add of a double at a time. Emulated on the bits, since another FPU picks other NaNs:

    - a finite or infinite sum plus a NaN is that NaN, quieted
    - a sum plus an infinity is that infinity, or the default NaN if the signs differ
    - a NaN sum stays put, except a quiet NaN with a larger significand replaces it (the
      positive one on a tie). A signaling NaN never does, and R's NA is one.

    quiet=True quiets every NaN first, as mean()'s x / n does in SSE. Dividing by a count keeps
    the result, so without it this is colMeans and rowMeans too."""
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    acc = None  # the running sum's bits once it's non-finite
    for b in v[~np.isfinite(v)].view(np.uint64).tolist():
        if b & _FRACTION:  # a NaN
            b |= _QUIET if quiet else 0
            if acc is None or not acc & _FRACTION:
                acc = b | _QUIET
            elif b & _QUIET and (b & _FRACTION, not b >> 63) > (acc & _FRACTION, not acc >> 63):
                acc = b
        elif acc is None:
            acc = b
        elif not acc & _FRACTION and acc != b:
            acc = _INDEFINITE
    if acc is None:
        raise ValueError("no Inf or NaN to sum")
    return struct.unpack("<d", struct.pack("<Q", acc))[0]
