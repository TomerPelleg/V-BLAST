from datetime import datetime
from pathlib import Path

from experiments.config import (
    BaseExperimentConfig,
    FixedProfileConfig,
    SweepProfileConfig,
)
from experiments.sample import run_fixed_profile, run_sweep
from experiments.save import save_experiment_data
from experiments.visuals.ecdf_plotter import generate_distribution_plots_ecdf
from experiments.visuals.trend_plotter import generate_trend_plot


def execute_and_plot_experiment(
    config: BaseExperimentConfig, base_dir: Path | str = "graphs"
) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H_%M")
    exp_type = "ECDF" if isinstance(config, FixedProfileConfig) else "Trend"

    target_dir = Path(base_dir) / f"{timestamp}_{exp_type}"
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"Directory created: {target_dir}")

    if isinstance(config, FixedProfileConfig):
        results = run_fixed_profile(config)
        save_experiment_data(results=results, settings=config, target_dir=target_dir)

        for field in config.metrics_to_plot:
            generate_distribution_plots_ecdf(
                results=results,
                target_dir=target_dir,
                field=field,
                matrix_dim=config.matrix_dimension,
                log_scale=config.plot_log_scale,
                show_means=False,
            )
            print(f"Generated ECDF plot for {field}")

    elif isinstance(config, SweepProfileConfig):
        sweep_results = run_sweep(config)
        save_experiment_data(
            results=sweep_results, settings=config, target_dir=target_dir
        )

        for field in config.metrics_to_plot:
            generate_trend_plot(
                sweep_results=sweep_results,
                target_dir=target_dir,
                field=field,
                sweep_parameter=config.sweep_parameter,
                log_scale_y=config.plot_log_scale,
            )
            print(f"Generated Trend plot for {field}")

    else:
        raise TypeError("Unknown configuration type provided.")

    return target_dir
