import gzip
import json
import os
from pathlib import Path

from pydantic import TypeAdapter

from experiments.config import BaseExperimentConfig
from experiments.sample import SingleExperimentResult

ResultData = (
    list[SingleExperimentResult] | dict[float | int, list[SingleExperimentResult]]
)

def save_experiment_data(
    results: ResultData,
    settings: BaseExperimentConfig,
    target_dir: Path,
):

    target_dir.mkdir(parents=True, exist_ok=True)

    results_path = target_dir / "results.json.gz"

    adapter = TypeAdapter(ResultData)
    json_bytes = adapter.dump_json(results)
    with gzip.open(results_path, "wb") as f:
        f.write(json_bytes)

    settings_path = os.path.join(target_dir, "settings.json")

    settings_dict = settings.model_dump(mode="json")

    with open(settings_path, "w", encoding="utf-8") as f:
        json.dump(settings_dict, f, indent=4, sort_keys=True)

    print(f"Experiment saved successfully in: {target_dir}")
