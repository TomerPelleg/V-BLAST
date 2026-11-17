import random

from pydantic import BaseModel

from experiments.config import FixedProfileConfig, SweepProfileConfig
from experiments.generate_matrix import generate_matrix
from experiments.generate_permutation import (
    PermutationGenerationMethods,
)
from experiments.generate_vector import generate_schnorr_euchner_target_vector
from experiments.get_full_solving_metrics import SEMetrics, get_SE_metrics


class SingleExperimentResult(BaseModel):
    seed: int
    results: dict[PermutationGenerationMethods, list[SEMetrics]] = []


def produce_one_sample(
    config: FixedProfileConfig, seed: int, track_cache: bool
) -> SingleExperimentResult:
    random.seed(seed)

    n = config.matrix_dimension
    A = generate_matrix(gen_method=config.matrix_generation_method, n=n)
    b = generate_schnorr_euchner_target_vector(
        A=A,
        method=config.vector_generation_method,
        noise=config.noise,
    )

    cur_experiment = SingleExperimentResult(
        seed=seed,
        results={method: [] for method in config.permutation_methods_and_repetitions},
    )

    for (
        perm_method,
        reps,
    ) in config.permutation_methods_and_repetitions.items():
        for _ in range(reps):
            metrics = get_SE_metrics(
                A=A, b=b, permutation_method=perm_method, track_cache=track_cache
            )
            cur_experiment.results[perm_method].append(metrics)

    return cur_experiment


def run_fixed_profile(config: FixedProfileConfig) -> list[SingleExperimentResult]:
    """Runs `produce_one_sample` N times in parallel based on the config."""

    print("Running warmup cycle...")
    produce_one_sample(config, seed=42, track_cache=False)
    
    results: list[SingleExperimentResult] = []

    # We generate seeds in the main process to avoid multiprocessing random state identicals
    seeds = [random.randint(0, 2**32 - 1) for _ in range(config.experiments_number)]

    print(
        f"Starting {config.experiments_number} experiments for matrix_dimension={config.matrix_dimension}, noise={config.noise}..."
    )

    track_cache = ("l1_cache_misses" in config.metrics_to_plot) or (
        "l3_cache_misses" in config.metrics_to_plot
    )

    for i, seed in enumerate(seeds):
        # Optional: Print progress for long runs
        if (i + 1) % 10 == 0 or (i + 1) == config.experiments_number:
            print(f"  Completed {i + 1}/{config.experiments_number}...")

        result = produce_one_sample(config, seed, track_cache=track_cache)
        results.append(result)

    return results


def run_sweep(
    config: SweepProfileConfig,
) -> dict[float | int, list[SingleExperimentResult]]:
    """Runs a full sweep over a varying parameter, returning mapped results."""
    fixed_configs = config.generate_fixed_configs()

    sweep_results: dict[float | int, list[SingleExperimentResult]] = {}

    for sweep_value, fixed_config in fixed_configs.items():
        print(
            f"\n--- Running Sweep Value: {config.sweep_parameter} = {sweep_value} ---"
        )
        sweep_results[sweep_value] = run_fixed_profile(fixed_config)

    return sweep_results
