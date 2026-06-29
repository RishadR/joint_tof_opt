"""
Compare the Sensitivity across different noise variances for DIGSSOptimizer (unit_max).
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from joint_tof_opt import (
    AdditiveGaussianToFModifier,
    CombSeparator,
    CompactStatProcess,
    Evaluator,
    OptimizationExperiment,
    ToFData,
    WindowSumWithAdditiveGaussianNoiseCalculator,
    generate_tof,
    pretty_print_log,
)
from joint_tof_opt.compact_stat_process import get_named_moment_module
from optimize_loop_paper import DIGSSOptimizer
from result_writer import clear_results, write_results_to_yaml
from sensitivity_compute import AltPaperEvaluator3


def read_parameter_mapping():
    with open("./data/parameter_mapping.json") as tof_config_file:
        parameter_mapping = yaml.safe_load(tof_config_file)
    return parameter_mapping


def run_sensitivity_comparison(
    evaluator_gen_func: Callable[[Path, torch.Tensor, str, dict], Evaluator],
    optimizers_to_compare: list[Callable[[Path, str | CompactStatProcess], OptimizationExperiment]],
    measurands_to_test: list[str],
    noise_variance: float,
    print_log: bool = False,
) -> list[dict[str, Any]]:
    """
    Main function to run sensitivity comparison experiments across measurands and depths.
    """
    ## Params
    gen_config = yaml.safe_load(open("./experiments/tof_config.yaml"))
    fetal_filter = CombSeparator(
        gen_config["sampling_rate"],
        gen_config["fetal_f"],
        2 * gen_config["fetal_f"],
        0.3,
        gen_config["datapoint_count"] // 2 + 1,
        True,
    )
    tof_modifier = AdditiveGaussianToFModifier(noise_var=noise_variance)

    # Initialize results table and windows storage
    results = []
    for measurand in measurands_to_test:
        ppath_file_mapping = read_parameter_mapping()
        experiments = ppath_file_mapping["experiments"]
        for experiment in experiments:
            print(f"Running Experiment: {experiment['filename']} | Measurand: {measurand} | Noise Var: {noise_variance}")
            ppath_filename = experiment["filename"]
            derm_thickness_mm = experiment["sweep_parameters"]["derm_thickness"]["value"]
            ppath_file: Path = Path("./data") / ppath_filename
            tof_dataset_file = Path("./data") / f"generated_tof_set_{ppath_file.stem}.npz"
            generate_tof(ppath_file, gen_config, tof_dataset_file, True, True)
            tof_data = ToFData.from_npz(tof_dataset_file)
            tof_data = tof_modifier.modify(tof_data)
            tof_data.to_npz(tof_dataset_file)

            # Run Optimizers
            for optimizer_func in optimizers_to_compare:
                optimizer_experiment = optimizer_func(tof_dataset_file, measurand)
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
                tof_data = ToFData.from_npz(tof_dataset_file)
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
            # Clean up generated TOF file
            if tof_dataset_file.exists():
                tof_dataset_file.unlink()
    return results


def main(noise_var: float, append_results: bool = False) -> None:
    results_path = Path("./results/noise_sensitivity_comparison_results.yaml")
    filter_hw = 0.01  # Hz
    # eval_func = lambda ppath, win, meas, conf, noise_calc: PaperEvaluator(ppath, win, meas, conf, filter_hw)
    eval_func = lambda ppath, win, meas, conf: AltPaperEvaluator3(ppath, win, meas, conf, filter_hw, noise_var)
    noise_calc = WindowSumWithAdditiveGaussianNoiseCalculator(noise_var)

    optimizer_funcs_to_test: list[Callable[[Path, str | CompactStatProcess], OptimizationExperiment]] = [
        lambda tof_file, measurand: DIGSSOptimizer(
            tof_file,
            measurand,
            normalization_scheme="unit_max",
            noise_calc=noise_calc,
            reg_weight=0.0,
            lr=0.1,
            window_smoothening=False,
        ),
        lambda tof_file, measurand: DIGSSOptimizer(
            tof_file,
            measurand,
            normalization_scheme="unit_sum",
            noise_calc=noise_calc,
            reg_weight=0.0,
            lr=0.1,
            window_smoothening=False,
        ),
    ]

    exp_results = run_sensitivity_comparison(eval_func, optimizer_funcs_to_test, ["abs"], noise_var, print_log=True)
    write_results_to_yaml(exp_results, results_path, append_results)


if __name__ == "__main__":
    results_path = Path("./results/noise_sensitivity_comparison_results.yaml")
    clear_results(results_path)
    noise_stds = np.logspace(0, 5, 6).tolist() # [1.0, 10.0, 100.0, 1000.0, 10000.0]
    noise_vars = [std**2 for std in noise_stds]
    iterations = 5
    for noise_var in noise_vars:
        for i in range(iterations):
            print(f"Running noise sensitivity comparison: Noise Var={noise_var}, iteration {i+1}/{iterations}...")
            main(noise_var, True)
