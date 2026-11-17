import ctypes
import os
from pathlib import Path

import numpy as np

from columns_ordering_algorithms.permutation import Permutation

# Resolve paths to the compiled .so files
_HERE = Path(__file__).resolve().parent
_RS_SO = _HERE / "_repeated_random_sampling.so"
_SQRD_SO = _HERE / "_sqrd.so"

# Shared ctypes definitions
DOUBLE_2D = np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags="C_CONTIGUOUS")
INT_1D = np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS")

# Repeated Random Sampling Wrapper
_rs_lib = ctypes.CDLL(str(_RS_SO))
_rs_lib = ctypes.CDLL(str(_RS_SO))
# void repeated_random_sampling_full_mkl(...)
_rs_lib.repeated_random_sampling_full_mkl.argtypes = [
    DOUBLE_2D,  # A_in
    ctypes.c_int,  # m_rows
    ctypes.c_int,  # n
    ctypes.c_int,  # b
    ctypes.c_uint32,  # seed
    ctypes.c_void_p,  # omegas_in (Optional)
    INT_1D,  # perm
]


def repeated_random_sampling_c(
    A: np.ndarray, b: int, seed: int = None, omegas: np.ndarray = None
) -> Permutation:
    A_c = np.ascontiguousarray(A, dtype=np.float64)
    m, n = A_c.shape
    perm_array = np.empty(n, dtype=np.int32)

    if omegas is not None:
        omegas_c = np.ascontiguousarray(omegas, dtype=np.float64)
        omegas_ptr = omegas_c.ctypes.data_as(ctypes.c_void_p)
        seed_val = 0
    else:
        omegas_ptr = None
        seed_val = (
            seed
            if seed is not None
            else int.from_bytes(os.urandom(4), byteorder="little")
        )

    _rs_lib.repeated_random_sampling_full_mkl(
        A_c, m, n, b, seed_val, omegas_ptr, perm_array
    )

    perm = Permutation(length=n, elements=perm_array.tolist())
    return perm


# SQRD Wrapper
_sqrd_lib = ctypes.CDLL(str(_SQRD_SO))
# void sqrd(const double *A_in, int m, int n, int *perm)
_sqrd_lib.sqrd.argtypes = [
    DOUBLE_2D,
    ctypes.c_int,
    ctypes.c_int,
    INT_1D,
]


def sqrd_c(A: np.ndarray) -> Permutation:
    A_c = np.ascontiguousarray(A, dtype=np.float64)
    m, n = A_c.shape
    perm_array = np.empty(n, dtype=np.int32)

    _sqrd_lib.sqrd(A_c, m, n, perm_array)

    perm = Permutation(length=n, elements=perm_array.tolist())
    return perm
