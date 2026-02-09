"""
Compare the Sensitivity between optmized vs. non-optimized windows and visualize the results.
"""

from typing import Any, Literal, Callable
from pathlib import Path
import torch
import torch.nn as nn
import yaml
import pandas as pd
import numpy as np
from joint_tof_opt.compact_stat_process import get_named_moment_module
from optimize_loop_paper import main_optimize
from sensitivity_compute import *
from joint_tof_opt import *
from optimize_liu import LiuOptimizer
from optimize_liu_alt import AltLiuOptimizer
from optimize_loop_paper import DIGSSOptimizer
from optimize_dummy import DummyOptimizationExperiment


def read_parameter_mapping():
    with open("./data/parameter_mapping.json", "r") as f:
        parameter_mapping = yaml.safe_load(f)
    return parameter_mapping


def main(
    evaluator_gen_func: Callable[[Path, torch.Tensor, str, dict], Evaluator],
    optimizers_to_compare: list[Callable[[Path, str | CompactStatProcess], OptimizationExperiment]],
    measurands_to_test: list[str],
    print_log: bool = False,
) -> list[dict[str, Any]]:
    """
    Main function to run sensitivity comparison experiments across measurands and depths.

    :param evaluator_gen_func: Function to generate an evaluator for sensitivity computation. The function should take
    (ppath_file: Path, window: torch.Tensor, measurand: nn.Module, noise_func: NoiseCalculator) and return an Evaluator instance.
    :type evaluator_gen_func: Callable[[Path, torch.Tensor, nn.Module], Evaluator]
    :param optimizers_to_compare: List of optimizer functions to compare. Each function should take
    (ppath_file: Path, measurand: CompactStatProcess) and return an OptimizationExperiment instance.
    :type optimizers_to_compare: list[Callable[[Path, CompactStatProcess], OptimizationExperiment]]
    :param measurands_to_test: List of measurand names to test (e.g., ['abs', 'm1', 'V']).
    :type measurands_to_test: list[str]
    :param print_log: Whether to print log messages during execution. (Default: False)
    :type print_log: bool
    :return: List of dictionaries containing results for each experiment.
    :rtype: list[dict[str, Any]]
    """
    ## Params
    gen_config = yaml.safe_load(open("./experiments/tof_config.yaml", "r"))
    fetal_filter = CombSeparator(
        gen_config["sampling_rate"],
        gen_config["fetal_f"],
        2 * gen_config["fetal_f"],
        0.3,
        gen_config["datapoint_count"] // 2 + 1,
        True,
    )

    # Initialize results table and windows storage
    results = []
    for measurand in measurands_to_test:
        ppath_file_mapping = read_parameter_mapping()
        experiments = ppath_file_mapping["experiments"]
        for experiment in experiments:
            print(f"Running Experiment: {experiment['filename']} | Measurand: {measurand}")
            ppath_filename = experiment["filename"]
            derm_thickness_mm = experiment["sweep_parameters"]["derm_thickness"]["value"]
            ppath_file: Path = Path("./data") / ppath_filename
            tof_dataset_file = Path("./data") / f"generated_tof_set_{ppath_file.stem}.npz"
            generate_tof(ppath_file, gen_config, tof_dataset_file, True, True)

            # Run Optimizers
            # measurand_module = get_named_moment_module(measurand, tof_series_tensor, bin_edges_tensor, meta_data)
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


if __name__ == "__main__":
    filter_hw = 0.001  # Hz
    # eval_func = lambda ppath, win, meas, conf, noise_calc: PaperEvaluator(ppath, win, meas, conf, filter_hw)
    eval_func = lambda ppath, win, meas, conf: AltPaperEvaluator2(ppath, win, meas, conf, filter_hw)

    optimizer_funcs_to_test: list[Callable[[Path, str | CompactStatProcess], OptimizationExperiment]] = [
        lambda tof_file, measurand: DIGSSOptimizer(
            tof_file, measurand, normalize_tof=False, patience=100, filter_hw=filter_hw, lr=0.01
        ),
        lambda tof_file, measurand: LiuOptimizer(tof_file, measurand, None, "mean", filter_hw, 2, 1.0),
        lambda tof_file, measurand: AltLiuOptimizer(tof_file, measurand, None, None, "mean", filter_hw, 2, 1.0),
        lambda tof_file, measurand: DummyOptimizationExperiment(tof_file, measurand, 1.0),
    ]

    exp_results = main(eval_func, optimizer_funcs_to_test, ["abs"], print_log=False)
    results_dict = {f"exp {i:03d}": res for i, res in enumerate(exp_results)}
    with open("./results/sensitivity_comparison_results.yaml", "w") as f:
        yaml.dump(results_dict, f, default_flow_style=False)
