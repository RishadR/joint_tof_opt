"""
Loads the filename -> sweep parameters mapping out of parameter_mapping.json. Only the
"experiments" key is used; everything else in that file (metadata, base sim/tissue params) is ignored.
"""

import json
from pathlib import Path

from pydantic import BaseModel


class _SweepParameterValue(BaseModel):
    value: int


class _ExperimentEntry(BaseModel):
    filename: str
    sweep_parameters: dict[str, _SweepParameterValue]


class _ParameterMappingFile(BaseModel):
    experiments: list[_ExperimentEntry]


def load_parameter_mapping(path: Path) -> dict[str, dict[str, int]]:
    """
    Maps each ppath filename to its sweep parameters, e.g. {"experiment_0000.npz": {"derm_thickness": 4}}.
    Only derm_thickness is swept today, but each filename maps to a dict (not a single value) to leave
    room for a second sweep parameter later.
    """
    with open(path) as f:
        raw = json.load(f)  # pyright: ignore[reportAny]
    parsed = _ParameterMappingFile.model_validate(raw)
    return {
        experiment.filename: {name: spec.value for name, spec in experiment.sweep_parameters.items()}
        for experiment in parsed.experiments
    }
