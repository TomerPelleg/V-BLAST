import numpy as np

from columns_ordering_algorithms.permutation import Permutation, inplace_swap_cols


def householder_vblast(A: np.ndarray) -> Permutation:
    """
    :param A: A real matrix with m rows and n columns
    :return: a permutation of [1,...,n] such that if we re-order the columns of A according to the permutation,
    we get the wanted VBLAST order
    """
    m, n = A.shape
    best_permutation = Permutation(length=n)
    Q, R = np.linalg.qr(A)
    L = np.linalg.inv(R).T

    # Initialize the squared norms for all columns once at the very beginning
    l2_norms_sq = np.sum(L**2, axis=0)

    for cur_column in reversed(range(n)):
        # 1. Find the best column using our tracked squared norms
        p = np.argmin(l2_norms_sq[: cur_column + 1])

        # 2. Swap columns in Permutation, L, and our tracked norms array
        best_permutation.swap(cur_column, p)
        inplace_swap_cols(L, cur_column, p)
        l2_norms_sq[cur_column], l2_norms_sq[p] = (
            l2_norms_sq[p],
            l2_norms_sq[cur_column],
        )

        norm_x = np.sqrt(l2_norms_sq[cur_column])

        # 3. Apply Householder reflection
        if norm_x > 0:
            x = L[: cur_column + 1, cur_column]

            # copysign forces addition of like-signs to prevent catastrophic cancellation
            alpha = np.copysign(norm_x, x[cur_column])

            v = x.copy()
            v[cur_column] += alpha
            v_norm = np.linalg.norm(v)

            if v_norm > 0:
                v /= v_norm
                # Because H is orthogonal, applying it to L strictly preserves the L2
                # norms of all the active columns. Our l2_norms_sq array is still 100% accurate!
                L[: cur_column + 1, :] -= 2.0 * np.outer(v, v @ L[: cur_column + 1, :])

        # 4. Downdate the norms row-by-row for the NEXT iteration
        if cur_column > 0:
            # Since we are ignoring the bottom row going forward, we just subtract
            # its squared values from our running totals.
            l2_norms_sq[:cur_column] -= L[cur_column, :cur_column] ** 2

            # np.maximum clips to 0 to prevent negative values from floating-point rounding errors
            l2_norms_sq[:cur_column] = np.maximum(l2_norms_sq[:cur_column], 0.0)

    return best_permutation
