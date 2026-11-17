import numpy as np

from columns_ordering_algorithms.permutation import Permutation


def vblast_by_computing_inverse(A: np.ndarray) -> Permutation:
    """
    Finds the column permutation to maximize trailing diagonal elements of R.
    Optimized to O(n^3) using Schur complement rank-1 updates.
    """
    A = np.atleast_2d(A)
    _, n_cols = A.shape

    # Compute the initial inverse matrix M = (A^T A)^-1
    # We use pinv to handle singular or rank-deficient matrices safely.
    M = np.linalg.pinv(A.T @ A)

    remaining_cols = list(range(n_cols))
    perm = []

    for _ in range(n_cols):
        diag_M = np.diag(M)

        min_idx = np.argmin(diag_M)

        chosen_col = remaining_cols[min_idx]
        perm.insert(0, chosen_col)

        remaining_cols.pop(min_idx)

        if len(remaining_cols) > 0:
            m_k = M[:, min_idx]
            m_kk = M[min_idx, min_idx]

            M_new = M - np.outer(m_k, m_k) / m_kk

            M = np.delete(M_new, min_idx, axis=0)
            M = np.delete(M, min_idx, axis=1)

    return Permutation(length=n_cols, elements=perm)
