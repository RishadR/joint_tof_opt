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
from typing import Any

import numpy as np
import torch

from joint_tof_opt import (
    AdditiveGaussianToFModifier,
    CombSeparator,
    CompactStatProcess,
    Evaluator,
    OptimizationExperiment,
    ToFConfig,
    ToFData,
    UnityTofModifier,
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


def run_sensitivity_comparison(
    evaluator_gen_func: Callable[[Path, torch.Tensor, str, ToFConfig], Evaluator],
    optimizers_to_compare: list[Callable[[ToFData, str | CompactStatProcess], OptimizationExperiment]],
    measurands_to_test: list[str],
    noise_variance: float,
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
    # tof_modifier = AdditiveGaussianToFModifier(noise_var=noise_variance)
    tof_modifier = UnityTofModifier()

    # Initialize results table and windows storage
    results = []
    for measurand in measurands_to_test:
        file_sweep_params = load_parameter_mapping(Path("./data/parameter_mapping.json"))
        for ppath_filename, sweep_params in file_sweep_params.items():
            print(f"Running Experiment: {ppath_filename}| Measurand: {measurand}| Noise Var: {noise_variance}")
            derm_thickness_mm = sweep_params["derm_thickness"]
            ppath_file: Path = Path("./data") / ppath_filename
            tof_data = generate_tof(ppath_file, gen_config, True, True)
            tof_data = tof_modifier.modify(tof_data)

            # Run Optimizers
            for optimizer_func in optimizers_to_compare:
                optimizer_experiment = optimizer_func(tof_data, measurand)
                optimizer_experiment.optimize()
                optimizer_name = str(optimizer_experiment)
                window = optimizer_experiment.window.detach().cpu()
                loss_history = optimizer_experiment.training_curves
                evaluator = evaluator_gen_func(ppath_file, window, measurand, gen_config)
                optimized_sensitivity = evaluator.evaluate()
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
                    log_dict = evaluator.get_log()
                    print("Log Details:")
                    pretty_print_log(log_dict)
    return results


def main(noise_var: float) -> list[dict[str, Any]]:
    filter_hw = 0.01  # Hz

    def eval_func(ppath: Path, win: torch.Tensor, meas: str, conf: ToFConfig) -> Evaluator:
        return AltPaperEvaluator3(ppath, win, meas, conf, filter_hw, noise_var)

    noise_calc = WindowSumWithAdditiveGaussianNoiseCalculator(noise_var)

    optimizer_funcs_to_test: list[Callable[[ToFData, str | CompactStatProcess], OptimizationExperiment]] = [
        lambda tof_data, measurand: DIGSSOptimizer(
            tof_data,
            measurand,
            normalization_scheme="unit_max",
            noise_calc=noise_calc,
            reg_weight=0.0,
            lr=0.1,
            window_smoothening=False,
        ),
        # lambda tof_data, measurand: DIGSSOptimizer(
        #     tof_file,
        #     measurand,
        #     normalization_scheme="unit_sum",
        #     noise_calc=noise_calc,
        #     reg_weight=0.0,
        #     lr=0.1,
        #     window_smoothening=False,
        # ),
    ]

    return run_sensitivity_comparison(eval_func, optimizer_funcs_to_test, ["abs"], noise_var, print_log=True)


def run_full_sweep() -> None:
    """Run main() once for each noise variance level and write all results."""
    results_path = Path("./results/noise_sensitivity_comparison_results.yaml")
    clear_results(results_path)
    # [0.0, 1.0, 10.0, 100.0, 1000.0, 10000.0]
    noise_vars = [0] + np.logspace(0, 5, 6).tolist()
    for noise_var in noise_vars:
        print(f"Running noise_var={noise_var}...")
        exp_results = main(noise_var)
        write_results_to_yaml(exp_results, results_path, append=True)


if __name__ == "__main__":
    run_full_sweep()
