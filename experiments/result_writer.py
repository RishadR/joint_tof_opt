from pathlib import Path
from typing import Any

import yaml


def clear_results(results_path: Path) -> None:
    """
    Clears the content of a YAML file if it exists, or creates a new empty one.

    Args:
        results_path: Path to the YAML file to clear.
    """
    with results_path.open("w") as f:
        yaml.dump({}, f)


def write_results_to_yaml(results: list[dict[str, Any]], results_path: Path, append: bool = False) -> None:
    """
    Writes or appends a list of result dictionaries to a YAML file with indexed keys.

    Args:
        results: List of dictionaries containing experiment results.
        results_path: Path to the output YAML file.
        append: If True, appends to existing results. If False, overwrites.
    """
    existing_results: dict[str, Any] = {}
    if append and results_path.exists():
        with results_path.open() as f:
            existing_results = yaml.safe_load(f) or {}
        if not isinstance(existing_results, dict):
            raise ValueError(f"Expected a mapping in {results_path}, got {type(existing_results).__name__}.")

    next_exp_index = 0
    if existing_results:
        existing_exp_indices = []
        for key in existing_results:
            if key.startswith("exp "):
                try:
                    existing_exp_indices.append(int(key.split()[1]))
                except (IndexError, ValueError):
                    continue
        next_exp_index = max(existing_exp_indices) + 1 if existing_exp_indices else len(existing_results)

    results_dict = {f"exp {i:03d}": res for i, res in enumerate(results, start=next_exp_index)}
    results_to_write = existing_results | results_dict if append else results_dict

    with results_path.open("w") as f:
        yaml.dump(results_to_write, f, default_flow_style=False)
