import numpy as np
import pytest

from columns_ordering_algorithms.approximate_vblast.c_wrapper import (
    repeated_random_sampling_c,
)
from columns_ordering_algorithms.approximate_vblast.repeated_random_sampling import (
    repeated_random_sampling,
)


@pytest.mark.parametrize(
    "m, n, b",
    [
        (10, 10, 3),
        (17, 3, 1),
        (20, 10, 7),
        (50, 50, 16),
        (60, 50, 8),
    ],
)
@pytest.mark.parametrize("seed", [42, 123, 999, 2023, 8675309])
def test_random_sampling_c_equivalence(m: int, n: int, b: int, seed: int):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(m, n)) * 10

    # 1. Pre-generate omegas
    num_blocks = (n + b - 1) // b
    l_dim = b + 6
    omegas = rng.normal(size=(num_blocks, l_dim, n))

    # 2. Inject omegas directly into the unified functions
    perm_py = repeated_random_sampling(A.copy(), b, omegas=omegas)
    perm_c = repeated_random_sampling_c(A.copy(), b, omegas=omegas)

    assert perm_c == perm_py
