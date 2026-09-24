"""
Compare the Sensitivity across different noise variances for DIGSSOptimizer (unit_max).

Purpose
-------
Sweeps instrument noise variance (0, then log-spaced 1 to 1e5, 20 parallel iterations each except the
noiseless case) and re-optimizes a window at each level, to see how sensitivity degrades with noise.

Runtime
-------
Watch out, might take a while - 7 noise levels x up to 20 parallel iterations x every experiment in
data/parameter_mapping.json.

Inputs
------
- experiments/tof_config.yaml
- data/parameter_mapping.json
- data/*.npz (ppath files listed in parameter_mapping.json)

Outputs
-------
- results/noise_sensitivity_comparison_results.yaml
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from joint_tof_opt import (
    CombSeparator,
    CompactStatProcess,
    DIGSSOptimizer,
    Evaluator,
    NoiseCalculator,
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


def run_sensitivity_comparison(
    evaluator_gen_func: Callable[[Path, torch.Tensor, str, ToFConfig], Evaluator],
    optimizers_to_compare: list[Callable[[ToFData, str | CompactStatProcess], OptimizationExperiment]],
    measurands_to_test: list[str],
    noise_variance: float,
    repeats: int = 1,
    print_log: bool = False,
) -> list[dict[str, Any]]:
    """
    Main function to run sensitivity comparison experiments across measurands and depths.
    """
    ## Params
    gen_config = load_tof_config(Path("./experiments/tof_config.yaml"))
    fetal_filter = CombSeparator(
        gen_config.sampling_rate,
        gen_config.fetal_f,
        2 * gen_config.fetal_f,
        0.3,
        gen_config.datapoint_count // 2 + 1,
        True,
    )
    # Initialize results table and windows storage
    results = []
    for measurand in measurands_to_test:
        file_sweep_params = load_parameter_mapping(Path("./data/parameter_mapping.json"))
        for ppath_filename, sweep_params in file_sweep_params.items():
            print(f"Running Experiment: {ppath_filename}| Measurand: {measurand}| Noise Var: {noise_variance}")
            derm_thickness_mm = sweep_params["derm_thickness"]
            ppath_file: Path = Path("./data") / ppath_filename
            base_tof_data = generate_tof(ppath_file, gen_config, True, True)

            # Run Optimizers - repeat the full noisy-training + eval cycle `repeats` times (matching
            # repeats_if_noisy) - see experiments/experiments_core.py.
            for optimizer_func in optimizers_to_compare:
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
                depth = derm_thickness_mm + 2  # Add 2 mm for epidermis
                epochs = len(loss_history)
                if epochs > 0:
                    final_optimizer_loss = loss_history[-1, :].tolist()
                else:
                    final_optimizer_loss = []

                # Compute the unfiltered measurand signal for logging
                assert tof_data.meta_data is not None, "ToFData meta_data was not found!"
                bin_edges = tof_data.bin_edges
                measurand_process = get_named_moment_module(measurand, tof_data)
                measurand_time_series = measurand_process.forward(window)
                filtered_signal = fetal_filter(measurand_time_series.unsqueeze(0).unsqueeze(0)).squeeze()

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
                        "filtered_signal": filtered_signal.numpy().tolist(),
                        "evaluator_log": evaluator_log,
                        "final_optimizer_loss": final_optimizer_loss,
                        "measurand_time_series": measurand_time_series.numpy().tolist(),
                        "noise_variance": noise_variance,
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

    results_path = noisy_results_path(Path("./results/noise_sensitivity_comparison_results.yaml"), inject_noise)
    clear_results(results_path)

    # [0.0, 1.0, 10.0, 100.0, 1000.0, 10000.0]
    noise_vars = cast(list[float], [0] + np.logspace(0, 5, 6).tolist())
    for noise_var in noise_vars:
        print(f"Running noise_var={noise_var}...")
        noise_calc = WindowSumWithAdditiveGaussianNoiseCalculator(noise_var)

        def eval_func(
            ppath: Path, win: torch.Tensor, meas: str, conf: ToFConfig, noise_calc: NoiseCalculator = noise_calc
        ) -> Evaluator:
            return evaluator_cls(ppath, win, meas, conf, noise_calc, filter_hw, tof_modifier)

        optimizer_funcs_to_test: list[Callable[[ToFData, str | CompactStatProcess], OptimizationExperiment]] = [
            lambda tof_data, measurand, noise_calc=noise_calc: DIGSSOptimizer(
                tof_data, measurand, noise_calc=noise_calc
            ),
        ]

        exp_results = run_sensitivity_comparison(
            eval_func, optimizer_funcs_to_test, ["abs"], noise_var, repeats=repeats, print_log=True
        )
        write_results_to_yaml(exp_results, results_path, append=True)


if __name__ == "__main__":
    main()
