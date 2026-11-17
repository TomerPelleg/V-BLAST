from typing import Literal

from pydantic import BaseModel, Field, model_validator

from experiments.generate_matrix import MatrixGenerationMethods
from experiments.generate_permutation import (
    PermutationGenerationMethods,
)
from experiments.generate_vector import VectorGenerationMethods

# Define what parameters we are allowed to sweep over in a trend plot
SweepParameter = Literal["matrix_dimension", "noise"]

# Restrict field to valid attribute names of SEMetrics for IDE autocompletion & type safety
# Whenever adding field to SEMetrics add it here
MetricField = Literal[
    "iterations",
    "permutation_computation_time_sec",
    "solving_time_sec",
    "total_time_sec",
    "convergence_probability",
    "l3_cache_misses",
    "l1_cache_misses",
]


class BaseExperimentConfig(BaseModel):
    """Base configuration shared by all experiment types."""

    matrix_generation_method: MatrixGenerationMethods
    vector_generation_method: VectorGenerationMethods

    # Maps a permutation algorithm to how many times it should be repeated per sample
    permutation_methods_and_repetitions: dict[PermutationGenerationMethods, int]

    # Number of different profiles (A, b) to generate
    experiments_number: int = Field(ge=1)

    # Plotting stuff
    metrics_to_plot: list[MetricField] = Field(default=["iterations"])
    plot_log_scale: bool = Field(default=False)


class FixedProfileConfig(BaseExperimentConfig):
    """Configuration for an ECDF plot (all input parameters are fixed)."""

    matrix_dimension: int = Field(ge=1)
    noise: float = Field(ge=0, default=0.0)


class SweepProfileConfig(BaseExperimentConfig):
    """Configuration for a Trend plot (one parameter varies, the rest are fixed)."""

    sweep_parameter: SweepParameter
    sweep_values: list[int | float] = Field(min_length=1)

    # The fixed values - one of these must be set depending on sweep_parameter
    fixed_matrix_dimension: int | None = Field(default=None, ge=1)
    fixed_noise: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_fixed_parameters(self) -> "SweepProfileConfig":
        """Ensures that the non-swept parameters are properly fixed."""
        if self.sweep_parameter == "matrix_dimension":
            if self.fixed_noise is None:
                raise ValueError(
                    "`fixed_noise` must be provided when sweeping `matrix_dimension`."
                )
            # Ensure all sweep values are integers since matrix_dimension must be an int
            if not all(isinstance(v, int) for v in self.sweep_values):
                raise ValueError(
                    "All `sweep_values` must be integers when sweeping `matrix_dimension`."
                )

        elif self.sweep_parameter == "noise":
            if self.fixed_matrix_dimension is None:
                raise ValueError(
                    "`fixed_matrix_dimension` must be provided when sweeping `noise`."
                )

        return self

    def generate_fixed_configs(self) -> dict[float | int, FixedProfileConfig]:
        """
        Helper method that breaks this sweep config down into a dictionary of
        ready-to-run FixedProfileConfigs.
        """
        configs = {}
        for val in self.sweep_values:
            configs[val] = FixedProfileConfig(
                matrix_generation_method=self.matrix_generation_method,
                vector_generation_method=self.vector_generation_method,
                permutation_methods_and_repetitions=self.permutation_methods_and_repetitions,
                experiments_number=self.experiments_number,
                # Resolve the varying vs fixed parameters
                matrix_dimension=val
                if self.sweep_parameter == "matrix_dimension"
                else self.fixed_matrix_dimension,
                noise=val if self.sweep_parameter == "noise" else self.fixed_noise,
                metrics_to_plot=self.metrics_to_plot,
                plot_log_scale=self.plot_log_scale,
            )
        return configs
