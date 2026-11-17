import numpy as np
import pytest

from columns_ordering_algorithms.approximate_vblast.repeated_random_sampling import (
    repeated_random_sampling,
)
from columns_ordering_algorithms.permutation import Permutation, permute_matrix_columns


def test_repeated_random_sampling_non_square_dimensions():
    """
    Bug 3 Test: Verifies repeated_random_sampling handles m != n matrices
    without shape mismatch in omega @ relevant_L.
    """
    m, n = 10, 4  # Tall matrix
    np.random.seed(42)
    A = np.random.randn(m, n)

    # Should not raise ValueError for dimension mismatch
    perm = repeated_random_sampling(A, b=2)
    # noqa: F821
    assert isinstance(perm, Permutation)
    assert perm.length == n


def test_permute_matrix_columns_reassignment():
    """
    Bug 2 Test: Verifies that permute_matrix_columns returns a new matrix
    and requires reassignment to take effect.
    """
    matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
    perm = Permutation(length=2, elements=[1, 0])

    # Calling without assignment does NOT mutate in-place
    result = permute_matrix_columns(matrix, perm)

    # Original matrix remains unchanged
    np.testing.assert_array_equal(matrix, np.array([[1.0, 2.0], [3.0, 4.0]]))
    # Returned matrix is correctly permuted
    np.testing.assert_array_equal(result, np.array([[2.0, 1.0], [4.0, 3.0]]))


def test_repeated_random_sampling_fat_matrix_validation():
    """
    Bug 4 Test: Tests behavior when m < n (fat matrix).
    """
    m, n = 3, 5
    A = np.random.randn(m, n)

    # Expect LinAlgError or ValueError unless m < n is explicitly handled
    with pytest.raises((ValueError, np.linalg.LinAlgError)):
        repeated_random_sampling(A, b=2)


def test_repeated_random_sampling_diagonal_scaling():
    """
    Tests the randomized V-BLAST ordering by using a diagonal matrix with
    vastly different scales. Even with the random projection (omega),
    the massive scale differences guarantee a specific deterministic output.
    """
    # If A is diagonal, Q is Identity, and R is A.
    # L = inv(R).T -> L is also diagonal with elements 1/A_ii
    #
    # A = diag(1000, 10, 0.1)
    # R = diag(1000, 10, 0.1)
    # L = diag(0.001, 0.1, 10)
    #
    # V-BLAST looks for the MINIMUM norm in L at each step.
    # Step 1: L norms are approx (0.001, 0.1, 10). Min is Col 0.
    #         Col 0 goes to the end (index 2).
    #         Permutation starts swapping 0 and 2.
    # Step 2: Next min is Col 1. It goes to index 1 (stays).
    # Step 3: Col 2 is left at index 0.
    # Resulting permutation should map standard [0, 1, 2] to [2, 1, 0]

    A = np.diag([1000.0, 10.0, 0.1])

    # Fix seed just in case, though the scale difference (100x) makes it robust
    # against almost any normal random matrix.
    np.random.seed(42)

    # Set b=1 to force it to find 1 column per block
    perm = repeated_random_sampling(A, b=1)

    assert perm.elements == [2, 1, 0]


def test_repeated_random_sampling_b_equals_n():
    """
    If b >= n, the Random Sampling should process the entire matrix
    in a single block iteration. It delegates entirely to find_rightmost_min_k_columns.
    """
    # We use a setup where L will have easily predictable norms
    np.random.seed(101)

    # A orthogonal matrix Q multiplied by specific singular values
    # We want L column norms to be strictly increasing, so A singular values are decreasing
    A = np.diag([100.0, 1.0, 0.01])

    # Calling with b=3 means it finds all 3 columns in the first outer loop
    perm = repeated_random_sampling(A, b=3)

    # Again, L = diag(0.01, 1.0, 100.0)
    # Norms: col0 < col1 < col2
    # So col0 goes to index 2, col1 to index 1, col2 to index 0
    assert perm.elements == [2, 1, 0]
