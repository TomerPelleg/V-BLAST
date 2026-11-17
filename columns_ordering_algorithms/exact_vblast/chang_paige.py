import numpy as np
import scipy.linalg.lapack as lapack

from columns_ordering_algorithms.permutation import Permutation


def find_rightmost_k_columns_square(A: np.ndarray, k: int) -> Permutation:
    """
    :param A: A real matrix with m rows and n columns
    :param k: number of columns to find
    :return: a permutation of [0,...,n-1] which has in the last k places the k columns with the best residual
    l2 norm (ordered).
    """
    m, n = A.shape
    best_permutation = Permutation(length=n)
    Q, R = np.linalg.qr(A)
    L = np.linalg.inv(R).T

    if k <= 0:
        raise ValueError(f"Requested {k} columns which is negative")
    if k > n:
        raise ValueError(f"Requested {k} columns for matrix with only {n} columns")

    # Initialize the squared norms for all columns once at the very beginning
    l2_norms_sq = np.sum(L**2, axis=0)

    for cur_column in reversed(range(n - k, n)):
        # 1. Find the best column using our tracked squared norms
        p = np.argmin(l2_norms_sq[: cur_column + 1])

        # 2. Bubble the chosen column to the right
        for idx in range(p, cur_column):
            # Swap columns in L
            L[:, [idx, idx + 1]] = L[:, [idx + 1, idx]]

            # Track the swap in the permutation object
            best_permutation.swap(idx, idx + 1)

            # Track the swap in our running squared norms array
            l2_norms_sq[idx], l2_norms_sq[idx + 1] = (
                l2_norms_sq[idx + 1],
                l2_norms_sq[idx],
            )

            a = L[idx, idx + 1]
            b = L[idx + 1, idx + 1]

            if abs(a) > 1e-12:
                # Generate Givens rotation
                c, s, _ = lapack.dlartg(b, a)

                row_idx = L[idx, :].copy()
                row_idx_1 = L[idx + 1, :].copy()

                # Apply rotation to rows idx and idx+1.
                # Since a Givens rotation is an orthogonal transformation, doing this
                # strictly preserves the L2 norms of all active columns.
                # No norm recalculation is needed!
                L[idx, :] = c * row_idx - s * row_idx_1
                L[idx + 1, :] = s * row_idx + c * row_idx_1

        # 3. Downdate the norms row-by-row for the NEXT iteration
        if cur_column > 0:
            # We subtract the squared values of the row we are discarding
            l2_norms_sq[:cur_column] -= L[cur_column, :cur_column] ** 2

            # np.maximum clips to 0 to prevent negative values from floating-point rounding errors
            l2_norms_sq[:cur_column] = np.maximum(l2_norms_sq[:cur_column], 0.0)

    return best_permutation


def chang_paige_vblast(A: np.ndarray) -> Permutation:
    """
    :param A: A real matrix with m rows and n columns
    :return: a permutation of [0,...,n-1] such that if we re-order the columns of A according to the permutation,
    we get the wanted VBLAST order
    """
    m, n = A.shape
    return find_rightmost_k_columns_square(A, n)
