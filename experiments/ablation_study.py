"""
Ablation study for DIGSSOptimizer for two ablations:
  1. use_window_post_process: (aka Flat-top Projection) - applied post-optimization.
  2. use_snr_left_bound: whether the left fixed region starts at max_snr_index (True) or 0 (False).

Purpose
-------
Sweeps the 4 combinations of the two ablation flags above, at 5 instrument noise levels (20 parallel
iterations each), to see how much each trick contributes to optimized-window sensitivity.

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

import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import torch

from joint_tof_opt import (
    AdditiveGaussianToFModifier,
    CompactStatProcess,
    Evaluator,
    OptimizationExperiment,
    ToFConfig,
    ToFData,
    WindowSumWithAdditiveGaussianNoiseCalculator,
    clear_results,
    generate_tof,
    load_parameter_mapping,
    load_tof_config,
    pretty_print_log,
    write_results_to_yaml,
)
from joint_tof_opt.compact_stat_process import get_named_moment_module

from .optimize_loop_paper import DIGSSOptimizer
from .sensitivity_compute import AltPaperEvaluator3


def run_ablation(
    evaluator_gen_func: Callable[[Path, torch.Tensor, str, ToFConfig], Evaluator],
    optimizers_to_compare: list[Callable[[ToFData, str | CompactStatProcess], OptimizationExperiment]],
    measurands_to_test: list[str],
    noise_variance: float,
    print_log: bool = False,
) -> list[dict[str, Any]]:
    gen_config = load_tof_config(Path("./experiments/tof_config.yaml"))
    tof_modifier = AdditiveGaussianToFModifier(noise_var=noise_variance)

    results = []
    for measurand in measurands_to_test:
        file_sweep_params = load_parameter_mapping(Path("./data/parameter_mapping.json"))
        for ppath_filename, sweep_params in file_sweep_params.items():
            print(f"Running Experiment: {ppath_filename} | Measurand: {measurand}")
            derm_thickness_mm = sweep_params["derm_thickness"]
            ppath_file: Path = Path("./data") / ppath_filename
            tof_data = generate_tof(ppath_file, gen_config, True, True)
            noisy_tof_file = Path("./data") / f"generated_tof_set_{ppath_file.stem}_t{threading.get_ident()}.npz"
            tof_data = tof_modifier.modify(tof_data)
            tof_data.to_npz(noisy_tof_file)

            for optimizer_func in optimizers_to_compare:
                optimizer_experiment = optimizer_func(tof_data, measurand)
                optimizer_experiment.optimize()
                optimizer_name = str(optimizer_experiment)
                window = optimizer_experiment.window.detach().cpu()
                loss_history = optimizer_experiment.training_curves
                evaluator = evaluator_gen_func(ppath_file, window, measurand, gen_config)
                optimized_sensitivity = evaluator.evaluate()
                depth = derm_thickness_mm + 2
                epochs = len(loss_history)
                final_optimizer_loss = loss_history[-1, :].tolist() if epochs > 0 else []

                tof_data = ToFData.from_npz(noisy_tof_file)
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
                        "evaluator_log": evaluator.get_log(),
                        "final_optimizer_loss": final_optimizer_loss,
                        "measurand_time_series": measurand_time_series.numpy().tolist(),
                        "noise_variance": noise_variance,
                    }
                )
                print(
                    f"Depth: {depth} mm |",
                    f"Optimizer: {optimizer_name} |",
                    f"Sensitivity: {optimized_sensitivity:.4e} |",
                    f"Epochs: {epochs} |",
                )
                if print_log:
                    print("Log Details:")
                    pretty_print_log(evaluator.get_log())
            noisy_tof_file.unlink(missing_ok=True)
    return results


def main(noise_var: float) -> list[dict[str, Any]]:
    filter_hw = 0.01  # Hz
    eval_func = lambda ppath, win, meas, conf: AltPaperEvaluator3(ppath, win, meas, conf, filter_hw, noise_var)
    noise_calc = WindowSumWithAdditiveGaussianNoiseCalculator(noise_var)

    base_kwargs: dict[str, Any] = {
        "normalization_scheme": "unit_max",
        "noise_calc": noise_calc,
        "reg_weight": 0.0,
        "lr": 0.1,
        "window_smoothening": False,
    }

    optimizer_funcs_to_test: list[Callable[[ToFData, str | CompactStatProcess], OptimizationExperiment]] = [
        # Baseline: both ablations on
        lambda tof_data, measurand: DIGSSOptimizer(
            tof_data, measurand, **base_kwargs,
            use_window_post_process=True, use_snr_left_bound=True,
        ),
        # No post-process
        lambda tof_data, measurand: DIGSSOptimizer(
            tof_data, measurand, **base_kwargs,
            use_window_post_process=False, use_snr_left_bound=True,
        ),
        # No SNR left bound
        lambda tof_data, measurand: DIGSSOptimizer(
            tof_data, measurand, **base_kwargs,
            use_window_post_process=True, use_snr_left_bound=False,
        ),
        # Neither
        lambda tof_data, measurand: DIGSSOptimizer(
            tof_data, measurand, **base_kwargs,
            use_window_post_process=False, use_snr_left_bound=False,
        ),
    ]

    return run_ablation(eval_func, optimizer_funcs_to_test, ["abs"], noise_var, print_log=False)


if __name__ == "__main__":
    results_path = Path("./results/ablation_results.yaml")
    clear_results(results_path)
    noise_variances = [0.0, 10.0, 100.0, 1000.0, 10000.0]  # 1000.0 already computed
    iterations = 20
    for noise_var in noise_variances:
        print(f"Running {iterations} iterations in parallel for noise_var={noise_var}...")
        with ThreadPoolExecutor(max_workers=iterations) as executor:
            futures = [executor.submit(main, noise_var) for _ in range(iterations)]
        for i, future in enumerate(futures):
            exp_results = future.result()
            print(f"  Writing results: iteration {i + 1}/{iterations}")
            write_results_to_yaml(exp_results, results_path, append=True)
