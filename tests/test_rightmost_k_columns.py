import pytest
import numpy as np
from scipy.linalg import qr

from columns_ordering_algorithms.permutation import Permutation, permute_matrix_columns
from columns_ordering_algorithms.exact_vblast.chang_paige import (
    find_rightmost_k_columns_square,
)


def test_permutation_validity_and_length():
    """Test that the returned object is a valid Permutation with length n."""
    m, n, k = 5, 4, 2
    A = np.random.randn(m, n)

    perm = find_rightmost_k_columns_square(A, k)

    assert isinstance(perm, Permutation)
    assert perm.length == n
    assert len(perm.elements) == n
    assert sorted(perm.elements) == list(range(n))


def test_single_column_k_1():
    """
    Test k=1: Verify the rightmost column maximizes |R[n-1, n-1]|
    compared to putting any other column at the end.
    """
    np.random.seed(42)
    m, n = 5, 4
    A = np.random.randn(m, n)

    perm = find_rightmost_k_columns_square(A, k=1)

    # QR of the permuted matrix
    A_perm = permute_matrix_columns(A, perm)
    _, R_chosen = qr(A_perm, mode="economic")
    chosen_diag_val = abs(R_chosen[-1, -1])

    # Check against all other possible columns placed at index n-1
    for col_idx in range(n):
        # Swap col_idx into the final position
        test_cols = list(range(n))
        test_cols.remove(col_idx)
        test_cols.append(col_idx)

        _, R_test = qr(A[:, test_cols], mode="economic")
        test_diag_val = abs(R_test[-1, -1])

        # Chosen column's diagonal value should be >= any other candidate
        assert chosen_diag_val >= test_diag_val - 1e-10


def test_greedy_pivoting_property_k_greater_than_1():
    """
    Test sequential greediness for k > 1:
    At step i (from n-1 down to n-k), the element |R[i, i]| is maximized
    among all remaining available columns.
    """
    np.random.seed(101)
    m, n, k = 6, 5, 3
    A = np.random.randn(m, n)

    perm = find_rightmost_k_columns_square(A, k=k)
    A_perm = permute_matrix_columns(A, perm)
    _, R_perm = qr(A_perm, mode="economic")

    # Verify rightmost selection at step n-1
    # then work backward for step n-2, n-3, ...
    for step_from_end in range(k):
        idx = (n - 1) - step_from_end
        current_val = abs(R_perm[idx, idx])

        # Evaluate remaining left columns placed at position `idx`
        for candidate_idx in range(idx):
            # Swap candidate column into position `idx`
            cols = list(range(n))
            cols[idx], cols[candidate_idx] = cols[candidate_idx], cols[idx]

            _, R_test = qr(A_perm[:, cols[: idx + 1]], mode="economic")
            test_val = abs(R_test[idx, idx])

            assert current_val >= test_val - 1e-10


def test_edge_case_k_equals_n():
    """Test full matrix column pivoting where k = n."""
    _, n = 4, 4
    A = np.array(
        [
            [1.0, 10.0, 0.1, 5.0],
            [0.0, 2.0, 0.0, 1.0],
            [0.0, 0.0, 3.0, 0.5],
            [1.0, 0.0, 0.0, 2.0],
        ]
    )

    perm = find_rightmost_k_columns_square(A, k=n)
    assert perm.length == n
    assert len(set(perm.elements)) == n


def test_invalid_k_raises_error():
    """Test out-of-bounds k values."""
    A = np.random.randn(4, 3)

    with pytest.raises((ValueError, IndexError)):
        find_rightmost_k_columns_square(A, k=0)

    with pytest.raises((ValueError, IndexError)):
        find_rightmost_k_columns_square(A, k=4)  # k > n
