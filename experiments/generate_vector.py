from enum import Enum

import numpy as np


class VectorGenerationMethods(Enum):
    BIPOLAR = "BIPOLAR"
    INT_BYTE = "Bytes (uniform integer 0-15)"
    ROUNDED_GAUSSIAN = "Rounded Gaussian"


def generate_schnorr_euchner_target_vector(
    A: np.ndarray, method: VectorGenerationMethods, noise: float
) -> np.ndarray:
    m, n = A.shape
    rng = np.random.default_rng()

    match method:
        case VectorGenerationMethods.BIPOLAR:
            true_b = rng.choice([-1, 1], size=n)
        case VectorGenerationMethods.INT_BYTE:
            true_b = rng.choice([i for i in range(16)], size=n)
        case VectorGenerationMethods.ROUNDED_GAUSSIAN:
            true_b = np.round(rng.standard_normal(size=n))
        case _:
            raise AssertionError(f"Unknown vector generation {method = }")

    lattice_point = A @ true_b
    noise = rng.normal(0, scale=noise, size=n)

    b = lattice_point + noise
    return b
