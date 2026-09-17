"""
Single source of truth for the schema of parameter_mapping.json (the filename -> sweep parameters
mapping). Only the "experiments" key is used; everything else in that file (metadata, base sim/tissue
params) is ignored on load.
"""

import json
from pathlib import Path

from pydantic import BaseModel


class SweepParameterSpec(BaseModel):
    value: int
    object_type: str


class ExperimentEntry(BaseModel):
    filename: str
    index: int
    sweep_parameters: dict[str, SweepParameterSpec]


class ParameterMappingFile(BaseModel):
    experiments: list[ExperimentEntry]


def load_parameter_mapping_entries(path: Path) -> list[ExperimentEntry]:
    """Load the full ExperimentEntry list (including index), e.g. to append more experiments onto it."""
    with open(path) as f:
        raw = json.load(f)  # pyright: ignore[reportAny]
    return ParameterMappingFile.model_validate(raw).experiments


def load_parameter_mapping(path: Path) -> dict[str, dict[str, int]]:
    """
    Maps each ppath filename to its sweep parameters, e.g. {"experiment_0000.npz": {"derm_thickness": 4}}.
    Only derm_thickness is swept today, but each filename maps to a dict (not a single value) to leave
    room for a second sweep parameter later.
    """
    return {
        experiment.filename: {name: spec.value for name, spec in experiment.sweep_parameters.items()}
        for experiment in load_parameter_mapping_entries(path)
    }


def save_parameter_mapping(path: Path, experiments: list[ExperimentEntry]) -> None:
    """Write experiments out to parameter_mapping.json, in ExperimentEntry field order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    mapping_file = ParameterMappingFile(experiments=experiments)
    with open(path, "w") as f:
        json.dump(mapping_file.model_dump(), f, indent=2)
