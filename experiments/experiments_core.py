"""
Shared core loop for the noisy train+eval repeat cycle used by every experiment in run_all_experiments.py.

Purpose
-------
Every experiment sweeps some parameter (depth, noise level, SDD index, filter setup, ...) and, for each
point in the sweep, trains an optimizer on a ToF window and evaluates it. When noise is involved, the
whole cycle - training-data noise AND evaluation-time noise - needs to be repeated `repeats_if_noisy`
times with independent-but-reproducible noise draws (see joint_tof_opt.core.ToFModifier.reseed and
joint_tof_opt.misc.evaluate_repeats), or the repeats collapse to identical draws and the mean/std computed
over them downstream is meaningless. Base implementation: experiments/ablation_study.py.
"""

from collections.abc import Callable
from typing import Any

import torch

from joint_tof_opt import AdditiveGaussianToFModifier, Evaluator, OptimizationExperiment, ToFData


def run_noisy_repeats(
    base_tof_data: ToFData,
    build_optimizer: Callable[[ToFData], OptimizationExperiment],
    evaluators_gen: Callable[[torch.Tensor], list[Evaluator]],
    noise_variance: float,
    repeats: int = 1,
) -> tuple[OptimizationExperiment, list[float | list[float]], list[dict[str, Any] | list[dict[str, Any]]]]:
    """
    Repeat the noisy-training + eval cycle `repeats` times (matching EvaluatorSpecs.repeats_if_noisy), so
    both the training-data noise and every evaluator's own eval-time noise get `repeats` independent draws -
    regardless of which side (training vs. eval) the "swept" noise_variance for this experiment lands on.

    Each repeat i (0..repeats-1):
      1. Builds a fresh AdditiveGaussianToFModifier(noise_variance, seed=i) and applies it to base_tof_data.
      2. Calls build_optimizer(noised_tof_data) and runs .optimize() on the result.
      3. Calls evaluators_gen(window) to get the Evaluator(s) to run against this repeat's window, reseeds
         each one's own tof_modifier to the same i (if it has one), then calls .evaluate()/.get_log().

    :param base_tof_data: Deterministic (un-noised) ToFData to draw each repeat's noisy training copy from.
    :param build_optimizer: Given this repeat's noised ToFData, build (but don't optimize) an
        OptimizationExperiment. Bind any extra args (measurand, fetal_f override, filter_hw, ...) via closure.
    :param evaluators_gen: Given this repeat's optimized window, return the list of Evaluator(s) to score it
        with (most experiments use one; overlap_compare*.py use two). Bind ppath_file/gen_config via closure.
    :param noise_variance: Variance for the training-data AdditiveGaussianToFModifier. Pass 0.0 for
        experiments where training noise isn't the point (only evaluation-time noise, gated by
        `inject_noise`, should vary) - a zero-variance draw is a no-op regardless of seed.
    :param repeats: Number of independent noisy repeats to run (1 for a noiseless/non-repeated run).
    :return: (last repeat's OptimizationExperiment, sensitivities, logs) - sensitivities and logs are lists
        with one entry per Evaluator returned by evaluators_gen, each entry being a scalar/dict when
        repeats == 1 or a list of `repeats` values otherwise (matching evaluate_repeats' convention).
    """
    per_evaluator_sensitivities: list[list[float]] = []
    per_evaluator_logs: list[list[dict[str, Any]]] = []
    optimizer_experiment: OptimizationExperiment | None = None

    for i in range(repeats):
        tof_modifier = AdditiveGaussianToFModifier(noise_var=noise_variance, seed=i)
        tof_data = tof_modifier.modify(base_tof_data)

        optimizer_experiment = build_optimizer(tof_data)
        optimizer_experiment.optimize()
        window = optimizer_experiment.window.detach().cpu()

        evaluators = evaluators_gen(window)
        if i == 0:
            per_evaluator_sensitivities = [[] for _ in evaluators]
            per_evaluator_logs = [[] for _ in evaluators]
        for j, evaluator in enumerate(evaluators):
            if evaluator.tof_modifier is not None:
                evaluator.tof_modifier.reseed(i)
            per_evaluator_sensitivities[j].append(evaluator.evaluate())
            per_evaluator_logs[j].append(evaluator.get_log())

    assert optimizer_experiment is not None, "repeats must be >= 1"

    def _collapse(values: list[Any]) -> Any:
        return values[0] if repeats == 1 else values

    sensitivities = [_collapse(v) for v in per_evaluator_sensitivities]
    logs = [_collapse(v) for v in per_evaluator_logs]
    return optimizer_experiment, sensitivities, logs
