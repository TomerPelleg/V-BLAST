from enum import Enum

import numba as nb
import numpy as np
import numpy.typing as npt
from pydantic import BaseModel


@nb.njit
def get_next_offset(cur_offset: int, positive_first: bool | None) -> int:
    """
    Function to compute next offset
    If positive first should go like 0,1,-1,2,-2,3,-3,...
    If negative first (else) should go like 0,-1,1,-2,2,-3,3,...
    """

    if positive_first is None:
        raise ValueError(
            "Achieved level without computing inital directions - should not have happened"
        )

    if positive_first:
        if cur_offset <= 0:
            return -cur_offset + 1
        else:
            return -cur_offset
    else:
        if cur_offset >= 0:
            return -cur_offset - 1
        else:
            return -cur_offset


class OutcomeStatus(Enum):
    RUNNING = "Running"
    DONE = "Done"
    MAX_ITERATIONS_REACHED = "Max Iterations Reached"


class SearchStats(BaseModel):
    iterations: int
    candidates: int
    outcome: OutcomeStatus

@nb.njit
def _ils_search_core(R: np.ndarray, y_tilde: np.ndarray, max_iterations: int):
    n = R.shape[0]
    best_residue = np.inf
    best_x = np.zeros(n, dtype=np.int64)

    rounded = np.zeros(n, dtype=np.int64)
    offset = np.zeros(n, dtype=np.int64)
    xLevel = np.zeros(n, dtype=np.int64)
    directions = np.zeros(n, dtype=nb.boolean)

    dist = np.zeros(
        n + 1, dtype=np.float64
    )  # To update the distance with O(1) in each level, we save the partial distances
    e = np.zeros(
        n, dtype=np.float64
    )  # e_k is the residual - e_k = y_k - \sum_{i=k+1}^n x_i * R _k,i

    level = n - 1
    e[level] = y_tilde[level]
    xReal = e[level] / R[level, level]
    rounded[level] = int(np.round(xReal))
    directions[level] = rounded[level] < xReal

    iterations = 0
    candidates = 0
    status_code = 1  # 2 = DONE, 3 = MAX_ITERATIONS_REACHED

    while True:
        iterations += 1
        if iterations > max_iterations:
            status_code = 3
            break

        xLevel[level] = rounded[level] + offset[level]

        step_cost = (R[level, level] * xLevel[level] - e[level]) ** 2
        current_dist = dist[level + 1] + step_cost

        if current_dist >= best_residue or level == 0:
            if level == 0 and current_dist < best_residue:
                best_x = xLevel.copy()
                best_residue = current_dist
                candidates += 1

            level += 1
            if level == n:
                status_code = 2
                break

            offset[level] = get_next_offset(offset[level], directions[level])
        else:
            dist[level] = current_dist
            level -= 1

            # Numba trick: an explicit loop here should be faster than np.dot
            # because it avoids array slicing and type-casting overhead.
            dot_product = 0.0
            for j in range(level + 1, n):
                dot_product += R[level, j] * xLevel[j]

            e[level] = y_tilde[level] - dot_product
            xReal = e[level] / R[level, level]
            rounded[level] = int(np.round(xReal))
            offset[level] = 0
            directions[level] = rounded[level] < xReal

    return best_x, iterations, candidates, status_code


def ils_search(
    H: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    max_iterations: int | None = None,
) -> tuple[npt.NDArray[np.int64], SearchStats]:
    """
    Solves an Integer Least Squares (ILS) problem using Schnorr-Euchner enumeration.
    """
    m, n = H.shape

    Q, R = np.linalg.qr(H, mode="reduced")
    y_tilde = Q.T @ y

    if max_iterations is None:
        max_iterations = 25 * n**2

    # Pass everything down to the compiled machine-code function
    best_x, iterations, candidates, status_code = _ils_search_core(
        R, y_tilde, max_iterations
    )

    # Translate the integer status code back to your Python Enum
    outcome = (
        OutcomeStatus.DONE if status_code == 2 else OutcomeStatus.MAX_ITERATIONS_REACHED
    )

    stats = SearchStats(iterations=iterations, candidates=candidates, outcome=outcome)

    return best_x, stats
