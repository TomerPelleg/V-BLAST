import time
from contextlib import nullcontext

import numpy as np
import scipy.linalg as la
from pydantic import BaseModel

from columns_ordering_algorithms.permutation import permute_matrix_columns
from experiments.generate_permutation import (
    PermutationGenerationMethods,
    compute_permutation,
)
from experiments.perf_tracker import PerfTracker
from schnorr_euchner import (
    OutcomeStatus,
    ils_search,
    ils_search_c,
)


class SEMetrics(BaseModel):
    iterations: int
    convergence_probability: bool
    permutation_computation_time_sec: float
    solving_time_sec: float
    total_time_sec: float
    l1_cache_misses: int | None
    l3_cache_misses: int | None


def get_SE_metrics(
    A: np.ndarray,
    b: np.ndarray,
    permutation_method: PermutationGenerationMethods,
    track_cache: bool = False,
) -> SEMetrics:

    if track_cache:
        events = ["L1-dcache-load-misses", "LLC-load-misses"]
        tracker = PerfTracker(events)
    else:
        tracker = nullcontext()

    with tracker:
        start = time.perf_counter()

        permutation = compute_permutation(A=A, method=permutation_method)
        permutation_time = time.perf_counter() - start

        permuted_A = permute_matrix_columns(matrix=A, perm=permutation)

        solving_start = time.perf_counter()
        _, res = ils_search_c(H=permuted_A, y=b)

        end = time.perf_counter()

    if track_cache:
        l1_misses = tracker.measurements.get("L1-dcache-load-misses")
        l3_misses = tracker.measurements.get("LLC-load-misses")
    else:
        l1_misses = None
        l3_misses = None

    return SEMetrics(
        iterations=res.iterations,
        permutation_computation_time_sec=permutation_time,
        solving_time_sec=end - solving_start,
        total_time_sec=end - start,
        l1_cache_misses=l1_misses,
        l3_cache_misses=l3_misses,
        convergence_probability=res.outcome == OutcomeStatus.DONE,
    )
