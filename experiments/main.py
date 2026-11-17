import argparse
import json
from pathlib import Path

from experiments.config import FixedProfileConfig, SweepProfileConfig
from experiments.runner import (
    execute_and_plot_experiment,
)


def load_config(file_path: str):
    raw_dict = json.loads(Path(file_path).read_text())

    if "sweep_parameter" in raw_dict:
        return SweepProfileConfig.model_validate(raw_dict)
    return FixedProfileConfig.model_validate(raw_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config")
    args = parser.parse_args()

    config = load_config(args.config)

    execute_and_plot_experiment(config)
