from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from experiments.config import MetricField
from experiments.generate_permutation import (
    PermutationGenerationMethods,
    get_permutation_color_index,
)
from experiments.sample import SingleExperimentResult

plt.rcParams.update(
    {
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 12,
    }
)


def generate_distribution_plots_ecdf(
    results: list[SingleExperimentResult],
    target_dir: Path,
    field: MetricField,
    matrix_dim: int,
    log_scale: bool = False,
    show_means: bool = False,
) -> Path:
    """Generates an ECDF plot for a specific metric across different algorithms."""
    records: list[dict[str, str | float]] = []
    # Should have list of experiment, which means in this context perm_method and the wanted metric
    # So an entry should look like:
    # {"Permutation Method": "random", "iterations": 10}

    for exp in results:
        for method, metrics_list in exp.results.items():
            for metric_obj in metrics_list:
                records.append(
                    {
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

    for idx, method in enumerate(unique_methods):
        palette_dict[method] = color_list[
            get_permutation_color_index(PermutationGenerationMethods(method))
        ]

    sns.ecdfplot(
        data=df,
        x=field,
        hue="Permutation Method",
        palette=palette_dict,
        linewidth=2,
        ax=ax,
    )

    if show_means and (leg := ax.get_legend()):
        means = df.groupby("Permutation Method", sort=False)[field].mean()
        for handle, text in zip(leg.legend_handles, leg.get_texts()):
            if (m := text.get_text()) in means:
                val = float(means[m])
                val_str = f"{val:.2e}" if 0 < abs(val) < 0.01 else f"{val:.2f}"
                text.set_text(f"{m} (mean: {val_str})")
                ax.axvline(x=val, color=handle.get_color(), linestyle="--", alpha=0.6)

    formatted_field = field.replace("_", " ").title()
    ax.set_title(f"(n = {matrix_dim})", fontsize=16, pad=15)
    ax.set_xlabel(formatted_field)
    ax.set_ylabel("Probability")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)

    sns.move_legend(ax, loc="best", title=None)

    if log_scale:
        ax.set_xscale("log")

    y_min, y_max = ax.get_ylim()
    buf = (y_max - y_min) * 0.05
    ax.set_ylim(y_min - buf, y_max + buf)

    output_path = target_dir / f"ECDF_{field}.pdf"
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    return output_path
