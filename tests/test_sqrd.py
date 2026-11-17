import numpy as np
import pytest

from columns_ordering_algorithms.approximate_vblast.c_wrapper import sqrd_c
from columns_ordering_algorithms.approximate_vblast.sqrd import sqrd
from columns_ordering_algorithms.permutation import (
    Permutation,
    inplace_swap_cols,
    permute_matrix_columns,
)


def textbook_sqrd(A: np.ndarray) -> Permutation:
    """
    Comptues SQRD decomposition according to Wubben original paper
    """
    m, n = A.shape
    sqrd_permutation = Permutation(length=n)

    Q = A.copy()
    R = np.zeros((n, n))

    for cur_column in range(n):
        relative_p = np.argmin(np.sum(Q**2, axis=0)[cur_column:])
        p = cur_column + relative_p

        if p != cur_column:
            sqrd_permutation.swap(cur_column, p)
            inplace_swap_cols(Q, cur_column, p)
            inplace_swap_cols(R, cur_column, p)

        qi = Q[:, cur_column]
        norm_q = np.linalg.norm(qi)
        R[cur_column, cur_column] = norm_q

        Q[:, cur_column] = qi / norm_q

        for j in range(cur_column + 1, n):
            R[cur_column, j] = np.dot(Q[:, j], qi)
            Q[:, j] -= R[cur_column, j] * qi

    assert np.allclose(Q.T @ Q, np.eye(n))
    assert np.allclose(R, np.triu(R))
    assert np.allclose(permute_matrix_columns(A, sqrd_permutation), Q @ R)

    return sqrd_permutation


@pytest.mark.parametrize(
    "m, n",
    [
        (10, 10),
        (17, 3),
        (20, 10),
        (50, 50),
        (60, 50),
    ],
)
@pytest.mark.parametrize("seed", [42, 123, 999, 2023, 8675309])
def test_sqrd_equivalence(m: int, n: int, seed: int):
    np.random.seed(seed)

    A = np.random.randn(m, n) * 10

    textbook_perm = textbook_sqrd(A.copy())
    computed_sqrd_perm = sqrd(A.copy())

    assert textbook_perm == computed_sqrd_perm


@pytest.mark.parametrize(
    "m, n",
    [
        (10, 10),
        (17, 3),
        (20, 10),
        (50, 50),
        (60, 50),
    ],
)
@pytest.mark.parametrize("seed", [42, 123, 999, 2023, 8675309])
def test_sqrd_c_equivalence(m: int, n: int, seed: int):
    np.random.seed(seed)

    A = np.random.randn(m, n) * 10

    textbook_perm = sqrd_c(A.copy())
    computed_sqrd_perm = sqrd(A.copy())

    assert textbook_perm == computed_sqrd_perm