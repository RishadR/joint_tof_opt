"""
Compare the sensitivity of our metric between different detector indices
"""

from typing import Any, Literal, Callable
from pathlib import Path
import torch
import torch.nn as nn
import yaml
import pandas as pd
import numpy as np
from optimize_loop_paper import main_optimize
from sensitivity_compute import *
from joint_tof_opt import (
    OptimizationExperiment,
    Evaluator,
    OptimizationExperiment,
    CompactStatProcess,
    generate_tof,
    pretty_print_log,
    get_noise_calculator,
    NoiseCalculator,
)
from optimize_liu import LiuOptimizer
from optimize_loop_paper import DIGSSOptimizer
from optimize_dummy import DummyOptimizationExperiment


def read_parameter_mapping():
    with open("./data/parameter_mapping.json", "r") as f:
        parameter_mapping = yaml.safe_load(f)
    return parameter_mapping


def main(
    evaluator_gen_func: Callable[[Path, torch.Tensor, str, NoiseCalculator], Evaluator],
    optimizers_to_compare: list[Callable[[Path, str | CompactStatProcess], OptimizationExperiment]],
    sdd_indices_to_test: list[int],
    print_log: bool = False,
) -> list[dict[str, Any]]:
    """
    Main function to run sensitivity comparison experiments across measurands and depths.

    :param evaluator_gen_func: Function to generate an evaluator for sensitivity computation. The function should take
    (ppath_file: Path, window: torch.Tensor, measurand: nn.Module, noise_func: NoiseCalculator) and return an Evaluator instance.
    :type evaluator_gen_func: Callable[[Path, torch.Tensor, nn.Module, NoiseCalculator], Evaluator]
    :param optimizers_to_compare: List of optimizer functions to compare. Each function should take
    (ppath_file: Path, measurand: CompactStatProcess) and return an OptimizationExperiment instance.
    :type optimizers_to_compare: list[Callable[[Path, CompactStatProcess], OptimizationExperiment]]
    :param sdd_indices_to_test: List of SDD indices to test (e.g., [0, 1, 2]).
    :type sdd_indices_to_test: list[int]
    :param print_log: Whether to print log messages during execution. (Default: False)
    :type print_log: bool
    :return: List of dictionaries containing results for each experiment.
    :rtype: list[dict[str, Any]]
    """
    ## Params
    lr_list = {"abs": 0.01, "m1": 0.01, "V": 0.01}  # Learning rates for different measurands

    # Initialize results table and windows storage
    results = []
    measurand = "abs"  # Fixed measurand for this experiment
    for sdd_index in sdd_indices_to_test:
        gen_config = yaml.safe_load(open("./experiments/tof_config.yaml", "r"))
        gen_config["selected_sdd_index"] = sdd_index
        lr = lr_list.get(measurand, 0.01)
        # Get the noise function for the measurand

        ## Run experiments
        print(f"Starting sensitivity comparison for measurand: {measurand}")
        ppath_file_mapping = read_parameter_mapping()
        experiments = ppath_file_mapping["experiments"]
        for experiment in experiments:
            ppath_filename = experiment["filename"]
            derm_thickness_mm = experiment["sweep_parameters"]["derm_thickness"]["value"]
            ppath_file: Path = Path("./data") / ppath_filename
            tof_dataset_file = Path("./data") / f"generated_tof_set_{ppath_file.stem}.npz"
            generate_tof(ppath_file, gen_config, tof_dataset_file)

            # Get TOF data tensors
            tof_data = np.load(tof_dataset_file)
            meta_data = dict(tof_data)
            tof_series = tof_data["tof_dataset"]
            bin_edges = tof_data["bin_edges"]
            tof_series_tensor = torch.tensor(tof_series, dtype=torch.float32)
            bin_edges_tensor = torch.tensor(bin_edges, dtype=torch.float32)

            # Run Optimizers
            # measurand_module = get_named_moment_module(measurand, tof_series_tensor, bin_edges_tensor, meta_data)
            for optimizer_func in optimizers_to_compare:
                optimizer_experiment = optimizer_func(tof_dataset_file, measurand)
                optimizer_experiment.lr = lr
                optimizer_experiment.optimize()
                optimizer_name = str(optimizer_experiment)
                window = optimizer_experiment.window
                loss_history = optimizer_experiment.training_curves
                noise_calculator = get_noise_calculator(measurand)
                evaluator = evaluator_gen_func(ppath_file, window, measurand, noise_calculator)
                optimized_sensitivity = evaluator.evaluate()
                depth = derm_thickness_mm + 2  # Add 2 mm for epidermis
                epochs = len(loss_history)
                results.append(
                    {
                        "Measurand": measurand,
                        "SDD_Index": sdd_index,
                        "Depth_mm": depth,
                        "Optimizer": optimizer_name,
                        "Optimized_Sensitivity": optimized_sensitivity,
                        "Epochs": epochs,
                        # Not exactly needed right now - maybe useul later
                        # "Bin_Edges": bin_edges.tolist(),
                        # "Optimized_Window": window.detach().cpu().numpy().tolist(),
                        # "fetal_hb_series": meta_data["fetal_hb_series"].tolist(),
                        # "filtered_signal": optimizer_experiment.final_signal.numpy().tolist(),
                        "evaluator_log": evaluator.get_log(),
                    }
                )
                print(
                    f"Depth: {depth} mm |",
                    f"Optimizer: {optimizer_name} |",
                    f"Sensitivity: {optimized_sensitivity:.4f} |",
                    f"Epochs: {epochs} |",
                )
                if print_log:
                    log_dict = evaluator.get_log()
                    print("Log Details:")
                    pretty_print_log(log_dict)
    return results


if __name__ == "__main__":
    eval_func = lambda ppath, win, meas, noise_calc: PaperEvaluator(ppath, win, meas)

    optimizer_funcs_to_test: list[Callable[[Path, str | CompactStatProcess], OptimizationExperiment]] = [
        lambda tof_file, measurand: DIGSSOptimizer(tof_file, measurand, normalize_tof=True, patience=100),
        lambda tof_file, measurand: LiuOptimizer(tof_file, measurand, "mean", 0.3, 2, 1.0),
        lambda tof_file, measurand: DummyOptimizationExperiment(tof_file, measurand, 1.0),
    ]

    exp_results = main(eval_func, optimizer_funcs_to_test, [2], print_log=False)
    results_dict = {f"exp {i:03d}": res for i, res in enumerate(exp_results)}
    with open("./results/detector_comparison_results.yaml", "w") as f:
        yaml.dump(results_dict, f, default_flow_style=False)
