import ctypes
from enum import Enum
from pathlib import Path

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel

from schnorr_euchner import OutcomeStatus, SearchStats

# Resolve paths to the compiled .so file
_HERE = Path(__file__).resolve().parent
_ILS_SO = _HERE / "_ils_search.so"  # Assuming CMake prefixes with _

DOUBLE_2D = np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags="C_CONTIGUOUS")
DOUBLE_1D = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS")
INT64_1D = np.ctypeslib.ndpointer(dtype=np.int64, ndim=1, flags="C_CONTIGUOUS")

_ils_lib = ctypes.CDLL(str(_ILS_SO))

# void ils_search_c(const double *H, const double *y, int m, int n, long long max_iters, long long *best_x, long long *iters, long long *cands, int *status)
_ils_lib.ils_search_c.argtypes = [
    DOUBLE_2D,  # H_in
    DOUBLE_1D,  # y_in
    ctypes.c_int,  # m
    ctypes.c_int,  # n
    ctypes.c_longlong,  # max_iterations
    INT64_1D,  # best_x
    ctypes.POINTER(ctypes.c_longlong),  # out_iterations
    ctypes.POINTER(ctypes.c_longlong),  # out_candidates
    ctypes.POINTER(ctypes.c_int),  # out_status
]


def ils_search_c(
    H: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    max_iterations: int | None = None,
) -> tuple[npt.NDArray[np.int64], SearchStats]:
    """
    Solves an Integer Least Squares (ILS) problem using Schnorr-Euchner enumeration natively in C.
    """
    m, n = H.shape

    if max_iterations is None:
        max_iterations = 25 * n**2

    H_c = np.ascontiguousarray(H, dtype=np.float64)
    y_c = np.ascontiguousarray(y, dtype=np.float64)

    best_x = np.empty(n, dtype=np.int64)

    out_iters = ctypes.c_longlong(0)
    out_cands = ctypes.c_longlong(0)
    out_status = ctypes.c_int(0)

    # Execute highly optimized native search
    _ils_lib.ils_search_c(
        H_c,
        y_c,
        m,
        n,
        max_iterations,
        best_x,
        ctypes.byref(out_iters),
        ctypes.byref(out_cands),
        ctypes.byref(out_status),
    )

    # Translate the integer status code back to the Enum
    outcome = (
        OutcomeStatus.DONE
        if out_status.value == 2
        else OutcomeStatus.MAX_ITERATIONS_REACHED
    )

    stats = SearchStats(
        iterations=out_iters.value, candidates=out_cands.value, outcome=outcome
    )

    return best_x, stats
