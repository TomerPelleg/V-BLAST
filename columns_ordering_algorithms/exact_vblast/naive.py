import numpy as np

from columns_ordering_algorithms.permutation import Permutation, inplace_swap_cols


def naive_vblast(A: np.ndarray) -> Permutation:
    """
    :param A: A real matrix with m rows and n columns
    :return: a permutation of [1,...,n] such that if we re-order the columns of A according to the permutation,
    we get the wanted VBLAST order
    """
    m, n = A.shape
    best_permutation = Permutation(length=n)
    Q, R = np.linalg.qr(A)

    progressive_A = np.copy(A)

    for cur_column in reversed(range(n)):
        best_column = -1
        best_element = 0
        for candidate_column in range(cur_column + 1):
            candidate_A = np.copy(progressive_A)
            inplace_swap_cols(candidate_A, cur_column, candidate_column)

            candidate_Q, candidate_R = np.linalg.qr(candidate_A)

            if abs(candidate_R[cur_column, cur_column]) > best_element:
                best_element = abs(candidate_R[cur_column, cur_column])
                best_column = candidate_column

        best_permutation.swap(best_column, cur_column)
        inplace_swap_cols(progressive_A, cur_column, best_column)

    return best_permutation
