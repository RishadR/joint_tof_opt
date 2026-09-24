"""
Ablation study for DIGSSOptimizer for two ablations:
  1. use_window_post_process: (aka Flat-top Projection) - applied post-optimization.
  2. use_snr_left_bound: whether the left fixed region starts at max_snr_index (True) or 0 (False).

Purpose
-------
Sweeps the 4 combinations of the two ablation flags above, at instrument noise standard deviations
0, 100, 1000, 10000 plus the variance from experiments/evaluator_specs.yaml (20 parallel iterations
each), to see how much each trick contributes to optimized-window sensitivity, and how that holds up
under noise.

Runtime
-------
Watch out, might take a while - 5 noise levels x 20 parallel iterations x 4 optimizer configs x every
experiment in data/parameter_mapping.json.

Inputs
------
- experiments/tof_config.yaml
- data/parameter_mapping.json
- data/*.npz (ppath files listed in parameter_mapping.json)

Outputs
-------
- results/ablation_results.yaml
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from joint_tof_opt import (
    CompactStatProcess,
    DIGSSOptimizer,
    Evaluator,
    OptimizationExperiment,
    ToFConfig,
    ToFData,
    WindowSumWithAdditiveGaussianNoiseCalculator,
    build_noise_tof_modifier,
    clear_results,
    format_sensitivity,
    generate_tof,
    get_evaluator_class,
    get_evaluator_filter_hw,
    load_evaluator_specs,
    load_parameter_mapping,
    load_tof_config,
    noisy_results_path,
    print_evaluator_log,
    write_results_to_yaml,
)
from joint_tof_opt.compact_stat_process import get_named_moment_module

from .experiments_core import run_noisy_repeats


def run_ablation(
    evaluator_gen_func: Callable[[Path, torch.Tensor, str, ToFConfig], Evaluator],
    optimizers_to_compare: list[Callable[[ToFData, str | CompactStatProcess], OptimizationExperiment]],
    measurands_to_test: list[str],
    noise_variance: float,
    repeats: int = 1,
    print_log: bool = False,
) -> list[dict[str, Any]]:
    gen_config = load_tof_config(Path("./experiments/tof_config.yaml"))
    assert repeats > 0, "repeats_if_noisy count cannot be non-positive!"
    results = []
    for measurand in measurands_to_test:
        file_sweep_params = load_parameter_mapping(Path("./data/parameter_mapping.json"))
        for ppath_filename, sweep_params in file_sweep_params.items():
            print(f"Running Experiment: {ppath_filename} | Measurand: {measurand}")
            derm_thickness_mm = sweep_params["derm_thickness"]
            ppath_file: Path = Path("./data") / ppath_filename
            base_tof_data = generate_tof(ppath_file, gen_config, True, True)

            for optimizer_func in optimizers_to_compare:
                # Repeat the full noisy-training + eval cycle `repeats` times (matching repeats_if_noisy) so
                # both the training-data noise and the eval-time noise get `repeats` independent draws,
                # regardless of which of the two carries the (currently swept) noise_variance.
                optimizer_experiment, [optimized_sensitivity], [evaluator_log] = run_noisy_repeats(
                    base_tof_data,
                    build_optimizer=lambda tof_data: optimizer_func(tof_data, measurand),
                    evaluators_gen=lambda window: [evaluator_gen_func(ppath_file, window, measurand, gen_config)],
                    noise_variance=noise_variance,
                    repeats=repeats,
                )
                window = optimizer_experiment.window.detach().cpu()
                tof_data = optimizer_experiment.tof_data

                optimizer_name = str(optimizer_experiment)
                loss_history = optimizer_experiment.training_curves
                depth = derm_thickness_mm + 2
                epochs = len(loss_history)
                final_optimizer_loss = loss_history[-1, :].tolist() if epochs > 0 else []

                assert tof_data.meta_data is not None, "ToFData meta_data was not found!"
                bin_edges = tof_data.bin_edges
                measurand_process = get_named_moment_module(measurand, tof_data)
                measurand_time_series = measurand_process.forward(window)

                results.append(
                    {
                        "Measurand": measurand,
                        "Depth_mm": depth,
                        "Optimizer": optimizer_name,
                        "Optimized_Sensitivity": optimized_sensitivity,
                        "Epochs": epochs,
                        "Bin_Edges": bin_edges.tolist(),
                        "Optimized_Window": window.numpy().tolist(),
                        "fetal_hb_series": tof_data.meta_data["fetal_hb_series"].tolist(),
                        "evaluator_log": evaluator_log,
                        "final_optimizer_loss": final_optimizer_loss,
                        "measurand_time_series": measurand_time_series.numpy().tolist(),
                        "noise_variance": noise_variance,
                        "repeat_count": repeats,
                    }
                )
                print(
                    f"Depth: {depth} mm |",
                    f"Optimizer: {optimizer_name} |",
                    f"Sensitivity: {format_sensitivity(optimized_sensitivity)} |",
                    f"Epochs: {epochs} |",
                )
                if print_log:
                    print_evaluator_log(evaluator_log)
    return results


eval_spec = load_evaluator_specs(Path("./experiments/evaluator_specs.yaml"))


def main(inject_noise: bool = eval_spec.inject_noise) -> None:
    repeats = eval_spec.repeats_if_noisy if inject_noise else 1
    evaluator_cls = get_evaluator_class(eval_spec.evaluator_to_use)
    filter_hw = get_evaluator_filter_hw(eval_spec)
    tof_modifier = build_noise_tof_modifier(eval_spec) if inject_noise else None

    results_path = noisy_results_path(Path("./results/ablation_results.yaml"), inject_noise)
    clear_results(results_path)

    # Noise standard deviations to sweep, plus the currently tuned variance from evaluator_specs.yaml.
    # Everything is stored (and passed to the noise calculator) as a variance.
    noise_stds = [0.0, 5.0, 10.0, 15.0]
    noise_variances = [std**2 for std in noise_stds] + [eval_spec.instrument_noise_variance]

    for noise_var in noise_variances:
        print(f"Running noise_var={noise_var}...")
        noise_calc = WindowSumWithAdditiveGaussianNoiseCalculator(noise_var)

        def eval_func(
            ppath: Path, win: torch.Tensor, meas: str, conf: ToFConfig, noise_calc=noise_calc
        ) -> Evaluator:
            return evaluator_cls(ppath, win, meas, conf, noise_calc, filter_hw, tof_modifier)

        base_kwargs: dict[str, Any] = {
            "normalization_scheme": "unit_max",
            "noise_calc": noise_calc,
        }

        optimizer_funcs_to_test: list[Callable[[ToFData, str | CompactStatProcess], OptimizationExperiment]] = [
            # Baseline: both ablations on
            lambda tof_data, measurand: DIGSSOptimizer(
                tof_data,
                measurand,
                **base_kwargs,
                use_window_post_process=True,
                window_smoothening=True,
            ),
            # No post-process
            lambda tof_data, measurand: DIGSSOptimizer(
                tof_data,
                measurand,
                **base_kwargs,
                use_window_post_process=False,
                window_smoothening=True,
            ),
            # No SNR left bound
            lambda tof_data, measurand: DIGSSOptimizer(
                tof_data,
                measurand,
                **base_kwargs,
                use_window_post_process=True,
                window_smoothening=False,
            ),
            # Neither
            lambda tof_data, measurand: DIGSSOptimizer(
                tof_data,
                measurand,
                **base_kwargs,
                use_window_post_process=False,
                window_smoothening=False,
            ),
        ]

        exp_results = run_ablation(
            eval_func, optimizer_funcs_to_test, ["abs"], noise_var, repeats=repeats, print_log=False
        )
        write_results_to_yaml(exp_results, results_path, append=True)


if __name__ == "__main__":
    main()
