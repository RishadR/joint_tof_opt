"""
Miscellaneous utility functions for joint_tof_opt package.
"""

from pathlib import Path
from typing import Any

from joint_tof_opt.core import Evaluator


def noisy_results_path(path: Path, inject_noise: bool) -> Path:
    """
    Prefix a results path with "noisy_" when noise was injected into the evaluation, so noisy and
    noiseless runs never overwrite each other. Returns path unchanged when inject_noise is False.
    """
    return path.with_name(f"noisy_{path.name}") if inject_noise else path


def figure_output_path(path: Path, inject_noise: bool) -> Path:
    """
    Route a figure save path into an "injected_noise" or "no_injected_noise" subfolder of its
    parent directory, so noisy and noiseless figures land in separate folders instead of being
    distinguished by filename prefix. Creates the subfolder if needed.
    """
    subdir = path.parent / ("injected_noise" if inject_noise else "no_injected_noise")
    subdir.mkdir(parents=True, exist_ok=True)
    return subdir / path.name


def evaluate_repeats(
    evaluator: Evaluator, repeats: int
) -> tuple[float | list[float], dict[str, Any] | list[dict[str, Any]]]:
    """
    Call evaluator.evaluate() `repeats` times. Each call re-generates ToFData and re-applies
    evaluator.tof_modifier, so every repeat gets an independent noise draw when tof_modifier is set (see
    EvaluatorSpecs.repeats_if_noisy) - evaluator.tof_modifier is reseeded with the repeat index (0..repeats-1)
    before each call, so repeats are reproducible yet distinct instead of all drawing identical noise.
    Returns (sensitivity, log) directly when repeats == 1, else (list of sensitivities, list of logs) - one
    entry per repeat.
    """
    sensitivities: list[float] = []
    logs: list[dict[str, Any]] = []
    for i in range(repeats):
        if evaluator.tof_modifier is not None:
            evaluator.tof_modifier.reseed(i)
        sensitivities.append(evaluator.evaluate())
        logs.append(evaluator.get_log())
    if repeats == 1:
        return sensitivities[0], logs[0]
    return sensitivities, logs


def format_sensitivity(sensitivity: float | list[float]) -> str:
    """Format a single sensitivity value, or the mean (n=...) of a list of repeats, for logging."""
    if isinstance(sensitivity, list):
        return f"mean={sum(sensitivity) / len(sensitivity):.4e} (n={len(sensitivity)})"
    return f"{sensitivity:.4e}"


def print_evaluator_log(log: dict[str, Any] | list[dict[str, Any]]) -> None:
    """Pretty print one evaluator log, or each log in a list of repeats."""
    print("Log Details:")
    for entry in log if isinstance(log, list) else [log]:
        pretty_print_log(entry)


def pretty_print_log(log_dict: dict[str, Any], float_round: int = 4) -> None:
    """
    Pretty print the log dictionary.

    Parameters
    ----------
    log_dict : dict[str, Any]
        Dictionary containing log metrics and their values.
    float_round : int, optional
        Number of decimal places to round the float values (default is 4).
    """
    formatted_items = []
    for key, value in log_dict.items():
        if isinstance(value, float):
            formatted_items.append(f"{key}: {value:.{float_round}e}")
        else:
            formatted_items.append(f"{key}: {value}")
    print(" | ".join(formatted_items))
