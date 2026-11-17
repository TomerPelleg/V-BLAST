import ctypes
from pathlib import Path

import numpy as np

from columns_ordering_algorithms.permutation import Permutation

_HERE = Path(__file__).resolve().parent
_HH_SO = _HERE / "_householder.so"
_CP_SO = _HERE / "_chang_paige.so"
_CI_SO = _HERE / "_computing_inverse.so"

# Shared ctypes definitions
DOUBLE_2D = np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags="C_CONTIGUOUS")
INT_1D = np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS")

# Householder Wrapper
_hh_lib = ctypes.CDLL(str(_HH_SO))
# void householder_full_mkl(const double *A_in, int m, int n, int *perm)
_hh_lib.householder_full_mkl.argtypes = [
    DOUBLE_2D,
    ctypes.c_int,
    ctypes.c_int,
    INT_1D,
]


def householder_vblast_c(A: np.ndarray) -> Permutation:
    A_c = np.ascontiguousarray(A, dtype=np.float64)
    m, n = A_c.shape
    perm_array = np.empty(n, dtype=np.int32)

    _hh_lib.householder_full_mkl(A_c, m, n, perm_array)

    perm = Permutation(length=n, elements=perm_array.tolist())
    return perm


# Chang-Paige Wrapper
_cp_lib = ctypes.CDLL(str(_CP_SO))
# void chang_paige_full_mkl(const double *A_in, int m, int n, int k_cols, int *perm)
_cp_lib.chang_paige_full_mkl.argtypes = [
    DOUBLE_2D,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    INT_1D,
]


def chang_paige_vblast_c(A: np.ndarray) -> Permutation:
    A_c = np.ascontiguousarray(A, dtype=np.float64)
    m, n = A_c.shape
    perm_array = np.empty(n, dtype=np.int32)

    # We pass 'n' for k_cols since we are decomposing the full matrix
    _cp_lib.chang_paige_full_mkl(A_c, m, n, n, perm_array)

    perm = Permutation(length=n, elements=perm_array.tolist())
    return perm


# Computing Inverse Wrapper
_ci_lib = ctypes.CDLL(str(_CI_SO))
# void computing_inverse_full_mkl(const double *A, int m, int n, int *perm)
_ci_lib.computing_inverse_full_mkl.argtypes = [
    DOUBLE_2D,
    ctypes.c_int,
    ctypes.c_int,
    INT_1D,
]


def vblast_by_computing_inverse_c(A: np.ndarray) -> Permutation:
    A_c = np.ascontiguousarray(A, dtype=np.float64)
    m, n = A_c.shape
    perm_array = np.empty(n, dtype=np.int32)

    _ci_lib.computing_inverse_full_mkl(A_c, m, n, perm_array)

    perm = Permutation(length=n, elements=perm_array.tolist())
    return perm
