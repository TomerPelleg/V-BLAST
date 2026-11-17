import ctypes
from pathlib import Path

import numpy as np

from columns_ordering_algorithms.permutation import Permutation

# Resolve paths to the compiled .so files
_HERE = Path(__file__).resolve().parent
_QR_SO = _HERE / "_qr_baselines.so"

# Shared ctypes definitions
DOUBLE_2D = np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags="C_CONTIGUOUS")
INT_1D = np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS")

_qr_lib = ctypes.CDLL(str(_QR_SO))

# Blocked QR Wrapper (BLAS-3)
# void qr_blocked_mkl(const double *A_in, int m, int n, int *perm)
_qr_lib.qr_blocked_mkl.argtypes = [
    DOUBLE_2D,
    ctypes.c_int,
    ctypes.c_int,
    INT_1D,
]


def qr_blocked_c(A: np.ndarray) -> Permutation:
    A_c = np.ascontiguousarray(A, dtype=np.float64)
    m, n = A_c.shape
    perm_array = np.empty(n, dtype=np.int32)

    _qr_lib.qr_blocked_mkl(A_c, m, n, perm_array)

    perm = Permutation(length=n, elements=perm_array.tolist())
    return perm


# Unblocked QR Wrapper (BLAS-2)
# void qr_unblocked_mkl(const double *A_in, int m, int n, int *perm)
_qr_lib.qr_unblocked_mkl.argtypes = [
    DOUBLE_2D,
    ctypes.c_int,
    ctypes.c_int,
    INT_1D,
]


def qr_unblocked_c(A: np.ndarray) -> Permutation:
    A_c = np.ascontiguousarray(A, dtype=np.float64)
    m, n = A_c.shape
    perm_array = np.empty(n, dtype=np.int32)

    _qr_lib.qr_unblocked_mkl(A_c, m, n, perm_array)

    perm = Permutation(length=n, elements=perm_array.tolist())
    return perm
