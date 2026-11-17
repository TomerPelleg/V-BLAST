from enum import Enum

import numpy as np


class MatrixGenerationMethods(Enum):
    ZERO_MEAN_GAUSSIAN = "ZeroMeanGaussian"
    UNIFORM = "Uniform"


def generate_matrix(gen_method: MatrixGenerationMethods, n: int) -> np.ndarray:
    match gen_method:
        case MatrixGenerationMethods.ZERO_MEAN_GAUSSIAN:
            return np.random.normal(loc=0.0, scale=1.0, size=(n, n))
        case MatrixGenerationMethods.UNIFORM:
            return np.random.uniform(low=0.0, high=1.0, size=(n, n))
        case _:
            raise AssertionError("Shouldn't have reached this unreachable code.")
