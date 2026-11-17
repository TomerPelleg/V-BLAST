import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pydantic import BaseModel
from scipy.stats import gmean

from experiments.generate_permutation import (
    PermutationGenerationMethods,
    get_permutation_color_index,
    get_permutation_marker,
)
from experiments.sample import SingleExperimentResult
from experiments.visuals.ecdf_plotter import MetricField

plt.rcParams.update(
    {
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 12,
    }
)


class MetricGraphConfig(BaseModel):
    # Only for trend plotter
    metric: MetricField
    needs_title: bool
    y_axis_label: str
    title: str | None
    use_gmean: bool


metric_graph_configs = {
    "iterations": MetricGraphConfig(
        metric="iterations",
        needs_title=False,
        title=None,
        y_axis_label="SE Iterations",
        use_gmean=True,
    ),
    "permutation_computation_time_sec": MetricGraphConfig(
        metric="permutation_computation_time_sec",
        needs_title=False,
        title=None,
        y_axis_label="Permutation Compute Time (s)",
        use_gmean=False,
    ),
    "solving_time_sec": MetricGraphConfig(
        metric="solving_time_sec",
        needs_title=False,
        title=None,
        y_axis_label="Solving Time (s)",
        use_gmean=True,
    ),
    "total_time_sec": MetricGraphConfig(
        metric="total_time_sec",
        needs_title=False,
        title=None,
        y_axis_label="Total Solve Time (s)",
        use_gmean=False,
    ),
    "convergence_probability": MetricGraphConfig(
        metric="convergence_probability",
        needs_title=False,
        title=None,
        y_axis_label="Convergence Probability",
        use_gmean=False,
    ),
    "l3_cache_misses": MetricGraphConfig(
        metric="l3_cache_misses",
        needs_title=False,
        title=None,
        y_axis_label="L3 Cache Misses",
        use_gmean=False,
    ),
    "l1_cache_misses": MetricGraphConfig(
        metric="l1_cache_misses",
        needs_title=False,
        title=None,
        y_axis_label="L1 Cache Misses",
        use_gmean=False,
    ),
}


def generate_trend_plot(
    sweep_results: dict[float | int, list[SingleExperimentResult]],
    target_dir: Path,
    field: MetricField,
    sweep_parameter: str,
    log_scale_y: bool = False,
    log_scale_x: bool = False,
) -> Path:
    """
    Generates a trend graph mapping a sweep parameter (x-axis) to a metric (y-axis).
    Displays the mean and 95% confidence interval across the repeated experiments.
    """
    records: list[dict[str, str | float]] = []

    for sweep_val, exp_list in sweep_results.items():
        for exp in exp_list:
            for method, metrics_list in exp.results.items():
                for metric_obj in metrics_list:
                    records.append(
                        {
                            "Sweep Value": sweep_val,
                            "Permutation Method": method.value,
                            field: getattr(metric_obj, field),
                        }
                    )

    if not records:
        raise ValueError(f"No metric data found for field '{field}'.")

    df = pd.DataFrame(records)
    target_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    unique_methods = df["Permutation Method"].unique()
    color_list = sns.color_palette("colorblind", 10)  # Colorblind-friendly palette

    palette_dict = {}
    marker_dict = {}

    for idx, method in enumerate(unique_methods):
        palette_dict[method] = color_list[
            get_permutation_color_index(PermutationGenerationMethods(method))
        ]
        marker_dict[method] = get_permutation_marker(
            PermutationGenerationMethods(method)
        )

    estimator = gmean if metric_graph_configs[field].use_gmean else "mean"
    sns.lineplot(
        data=df,
        x="Sweep Value",
        y=field,
        hue="Permutation Method",
        style="Permutation Method",  # Tells seaborn to vary styles by method
        palette=palette_dict,
        markers=marker_dict,
        dashes=False,  # Keeps lines solid (don't mix dashed/solid lines)
        linewidth=2,
        ax=ax,
        errorbar=None,
        estimator=estimator,
    )

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels, title=None, loc="lower right")

    if sweep_parameter == "matrix_dimension":
        formatted_sweep = "Matrix Dimension"
    if sweep_parameter == "noise":
        formatted_sweep = "Noise Vector Standard Deviation (σ)"
    else:
        formatted_sweep = sweep_parameter.replace("_", " ").title()

    if metric_graph_configs[field].needs_title:
        ax.set_title(
            f"{metric_graph_configs[field].title} vs {formatted_sweep}",
            fontsize=16,
            pad=15,
        )
    ax.set_xlabel(formatted_sweep)
    ax.set_ylabel(metric_graph_configs[field].y_axis_label)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)

    if log_scale_y:
        ax.set_yscale("log")
    if log_scale_x:
        ax.set_xscale("log")

    output_path = target_dir / f"Trend_{field}_vs_{sweep_parameter}.pdf"
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    return output_path
