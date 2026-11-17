import cvxpy as cp
import numpy as np
import pytest

from schnorr_euchner import ils_search_c
from schnorr_euchner.se import get_next_offset, ils_search


def test_positive_first_sequence():
    """Verifies the alternation sequence when positive_first=True.

    Expected: 0 -> 1 -> -1 -> 2 -> -2 -> 3 -> -3 -> 4
    """
    expected_sequence = [0, 1, -1, 2, -2, 3, -3, 4]
    actual_sequence = [0]

    current_offset = 0
    # Generate the next 7 steps in the chain
    for _ in range(7):
        current_offset = get_next_offset(current_offset, positive_first=True)
        actual_sequence.append(current_offset)

    assert actual_sequence == expected_sequence, (
        f"Expected {expected_sequence}, but got {actual_sequence}"
    )


def test_negative_first_sequence():
    """Verifies the alternation sequence when positive_first=False.

    This explicitly tests the previous infinite loop bug regression.
    Expected: 0 -> -1 -> 1 -> -2 -> 2 -> -3 -> 3 -> -4
    """
    expected_sequence = [0, -1, 1, -2, 2, -3, 3, -4]
    actual_sequence = [0]

    current_offset = 0
    # Generate the next 7 steps in the chain
    for _ in range(7):
        current_offset = get_next_offset(current_offset, positive_first=False)
        actual_sequence.append(current_offset)

    assert actual_sequence == expected_sequence, (
        f"Expected {expected_sequence}, but got {actual_sequence}"
    )


@pytest.mark.parametrize(
    "current, positive_first, expected",
    [
        # Individual edge-case step validations
        (0, True, 1),
        (0, False, -1),
        (10, True, -10),
        (-10, True, 11),
        (10, False, -11),
        (-10, False, 10),
    ],
)
def test_individual_offset_steps(current, positive_first, expected):
    """Parametrized testing for individual stateless boundary jumps."""
    assert get_next_offset(current, positive_first) == expected


def test_invalid_direction_raises_value_error():
    """Ensures that passing None to positive_first properly triggers the

    ValueError guardrail.
    """
    with pytest.raises(ValueError) as exc_info:
        get_next_offset(0, positive_first=None)

    assert "Achieved level without computing" in str(exc_info.value)


def solve_ils_cvxpy(H: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Ground truth Integer Least Squares solver using CVXPY."""
    n = H.shape[1]
    x = cp.Variable(n, integer=True)
    objective = cp.Minimize(cp.sum_squares(H @ x - y))
    problem = cp.Problem(objective)
    problem.solve()

    if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        raise RuntimeError(f"CVXPY failed. Status: {problem.status}")

    return np.round(x.value).astype(np.int64)


@pytest.mark.parametrize("seed", [42, 999, 123, 777, 12412, 214897])
@pytest.mark.parametrize(
    "m, n, noise_std, use_identity",
    [
        (4, 4, 0.0, True),
        (5, 5, 0.0, False),
        (8, 4, 0.5, False),
        (6, 6, 1.5, False),
        (30, 30, 0.3, False),
    ],
)
def test_ils_search_against_cvxpy(seed, m, n, noise_std, use_identity):
    # 1. Initialize a strictly local, reproducible RNG for this specific test
    rng = np.random.default_rng(seed)

    # 2. Generate H and y based on the parametrized specifications
    if use_identity:
        assert m == n, "Identity matrix must be square."
        H = np.eye(n, dtype=np.float64)
        # Random floating point target
        y = rng.uniform(-10.0, 10.0, size=n)
    else:
        H = rng.standard_normal((m, n))
        x_true = rng.integers(-5, 5, size=n)
        # Add noise scaled by noise_std
        noise = rng.standard_normal(m) * noise_std
        y = H @ x_true + noise

    # 3. Run the implementation being tested
    x_est, stats = ils_search(H, y, max_iterations=100 * n**3)
    x_est_c, stats_c = ils_search_c(H, y, max_iterations=100 * n**3)

    # 4. Run the ground truth solver
    x_cvx = solve_ils_cvxpy(H, y)

    # 5. Assertions
    dist_est = np.linalg.norm(H @ x_est - y)
    dist_cvx = np.linalg.norm(H @ x_cvx - y)

    np.testing.assert_allclose(
        dist_est,
        dist_cvx,
        rtol=1e-5,
        atol=1e-8,
        err_msg=f"Objective distances differ. Seed {seed} failed.",
    )

    np.testing.assert_array_equal(
        x_est, x_cvx, err_msg=f"Integer vectors do not match. Seed {seed} failed."
    )
    np.testing.assert_array_equal(
        x_est_c, x_cvx, err_msg=f"Integer vectors do not match. Seed {seed} failed."
    )
