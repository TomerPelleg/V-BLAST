import time

import numpy as np

from columns_ordering_algorithms.exact_vblast.c_wrapper import (
    chang_paige_vblast_c,
    householder_vblast_c,
    vblast_by_computing_inverse_c,
)
from columns_ordering_algorithms.exact_vblast.chang_paige import chang_paige_vblast
from columns_ordering_algorithms.exact_vblast.chang_paige_paper import (
    change_paige_paper,
)
from columns_ordering_algorithms.exact_vblast.computing_inverse import (
    vblast_by_computing_inverse,
)
from columns_ordering_algorithms.exact_vblast.housholder import householder_vblast
from columns_ordering_algorithms.exact_vblast.naive import naive_vblast

IMPLEMENTATIONS = [
    {"name": "Naive", "func": naive_vblast},
    {"name": "Chang-Paige", "func": chang_paige_vblast},
    {"name": "Chang-Paige C", "func": chang_paige_vblast_c},
    {"name": "Householder (L Matrix)", "func": householder_vblast},
    {"name": "Householder C", "func": householder_vblast_c},
    {"name": "Computing Inverse", "func": vblast_by_computing_inverse},
    {"name": "Computing Inverse C", "func": vblast_by_computing_inverse_c},
    {"name": "Change-Paige Paper Notation", "func": change_paige_paper},  # noqa: F821
]

if __name__ == "__main__":
    np.set_printoptions(precision=2, suppress=True, linewidth=100)

    num_iterations = 100

    total_times: dict[str, float] = {impl["name"]: 0.0 for impl in IMPLEMENTATIONS}

    print(f"Starting {num_iterations} test iterations across varying matrix sizes...")

    for i in range(num_iterations):
        n = np.random.randint(17, 30)
        m = np.random.randint(n, n + 11)
        A = np.random.rand(m, n)

        baseline_name = IMPLEMENTATIONS[0]["name"]
        baseline_func = IMPLEMENTATIONS[0]["func"]

        start_time = time.perf_counter()
        baseline_order = baseline_func(A)
        total_times[baseline_name] += time.perf_counter() - start_time

        for impl in IMPLEMENTATIONS[1:]:
            name = impl["name"]
            func = impl["func"]

            start_time = time.perf_counter()
            current_order = func(A)
            total_times[name] += time.perf_counter() - start_time

            assert current_order == baseline_order, (
                f"\n{'!' * 60}\n"
                f"Mismatch detected on iteration {i} for shape {A.shape}!\n"
                f"{baseline_name} (Baseline) Order : {baseline_order.elements}\n"
                f"{name} Order                : {current_order.elements}\n"
                f"Matrix A:\n{A}\n"
                f"{'!' * 60}"
            )

    print("\n" + "=" * 60)
    print(f" SUCCESS: All {num_iterations} iterations matched perfectly! ")
    print("=" * 60)
    print(f"{'Algorithm Variant':<35} | {'Total Execution Time':<20}")
    print("-" * 60)

    sorted_times = sorted(total_times.items(), key=lambda item: item[1])
    for name, duration in sorted_times:
        print(f"{name:<35} | {duration:.5f} seconds")

    print("=" * 60)
