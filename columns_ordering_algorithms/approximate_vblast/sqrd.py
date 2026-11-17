import numpy as np

from columns_ordering_algorithms.permutation import Permutation, inplace_swap_cols


def sqrd(A: np.ndarray) -> Permutation:
    """
    Computes a right-looking QR decomposition with column pivoting.
    At each step, it selects the column with the minimum norm (SQRD order).

    :param A: A real matrix with m rows and n columns
    :return: a permutation of [1,...,n] such that if we re-order the columns of A
    according to the permutation, we get the wanted SQRD order
    """
    m, n = A.shape
    sqrd_permutation = Permutation(length=n)

    R = A.copy()

    # Initialize the squared norms for all columns once at the very beginning
    l2_norms_sq = np.sum(R**2, axis=0)

    for cur_column in range(n):
        relative_p = np.argmin(l2_norms_sq[cur_column:])
        p = cur_column + relative_p

        if p != cur_column:
            sqrd_permutation.swap(cur_column, p)
            inplace_swap_cols(R, cur_column, p)
            l2_norms_sq[cur_column], l2_norms_sq[p] = (
                l2_norms_sq[p],
                l2_norms_sq[cur_column],
            )

        if cur_column == m:
            return sqrd_permutation

        x = R[cur_column:, cur_column]
        norm_x = np.linalg.norm(x)

        if norm_x > 0:
            alpha = np.copysign(norm_x, x[0])

            v = x.copy()
            v[0] += alpha
            v_norm = np.linalg.norm(v)

            if v_norm > 0:
                v /= v_norm
                R[cur_column:, cur_column:] -= 2.0 * np.outer(
                    v, v @ R[cur_column:, cur_column:]
                )

        if cur_column < n - 1:
            l2_norms_sq[cur_column + 1 :] -= R[cur_column, cur_column + 1 :] ** 2

            l2_norms_sq[cur_column + 1 :] = np.maximum(
                l2_norms_sq[cur_column + 1 :], 0.0
            )

    return sqrd_permutation
