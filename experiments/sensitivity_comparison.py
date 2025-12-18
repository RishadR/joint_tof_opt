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
from generate_tof_set import generate_tof
from optimize_loop_paper import main_optimize
from compute_sensitivity import (
    FetalSensitivityEvaluator,
    FetalSensitivityNoInterferenceEvaluator,
    CorrelationEvaluator,
    CorrelationxSNREvaluator,
)
from joint_tof_opt import (
    named_moment_types,
    OptimizationExperiment,
    Evaluator,
    get_named_moment_module,
    OptimizationExperiment,
    CompactStatProcess,
    noise_func_table,
)
from optimize_liu import LiuOptimizer
from optimize_loop_paper import DIGSSOptimizer


def read_parameter_mapping():
    with open("./data/parameter_mapping.json", "r") as f:
        parameter_mapping = yaml.safe_load(f)
    return parameter_mapping


def main(
    evaluator_gen_func: Callable[[Path, torch.Tensor, nn.Module, Callable], Evaluator],
    optimizers_to_compare: list[Callable[[Path, str | CompactStatProcess], OptimizationExperiment]],
    save: bool = True,
) -> list[dict[str, Any]]:
    """
    Main function to run sensitivity comparison experiments across measurands and depths.

    :param evaluator_gen_func: Function to generate an evaluator for sensitivity computation. The function should take
    (ppath_file: Path, window: torch.Tensor, measurand: nn.Module, noise_func: Callable) and return an Evaluator instance.
    :type evaluator_gen_func: Callable[[Path, torch.Tensor, nn.Module, Callable], Evaluator]
    :param optimizers_to_compare: List of optimizer functions to compare. Each function should take
    (ppath_file: Path, measurand: CompactStatProcess) and return an OptimizationExperiment instance.
    :type optimizers_to_compare: list[Callable[[Path, CompactStatProcess], OptimizationExperiment]]
    :param save: Whether to save the results.
    :type save: bool
    """
    ## Params
    lr_list = {"abs": 0.05, "m1": 0.01, "V": 0.01}  # Learning rates for different measurands
    gen_config = yaml.safe_load(open("./experiments/tof_config.yaml", "r"))

    # Initialize results table and windows storage
    results = []
    for measurand in ["abs"]:
        # for measurand in named_moment_types:
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
            measurand_module = get_named_moment_module(measurand, tof_series_tensor, bin_edges_tensor, meta_data)
            for optimizer_func in optimizers_to_compare:
                optimizer_experiment = optimizer_func(tof_dataset_file, measurand)
                optimizer_experiment.lr = lr
                optimizer_experiment.optimize()
                optimizer_name = str(optimizer_experiment)
                window = optimizer_experiment.window
                loss_history = optimizer_experiment.training_curves
                evaluator = evaluator_gen_func(ppath_file, window, measurand_module, noise_func_table[measurand])
                optimized_sensitivity = evaluator.evaluate()
                depth = derm_thickness_mm + 2  # Add 2 mm for epidermis
                epochs = len(loss_history)
                results.append(
                    {
                        "Measurand": measurand,
                        "Depth_mm": depth,
                        "Optimizer": optimizer_name,
                        "Optimized_Sensitivity": optimized_sensitivity,
                        "Epochs": epochs,
                        "Bin_Edges": bin_edges,
                        "Optimized_Window": window.detach().cpu().numpy(),
                    }
                )
                print(
                    f"Depth: {depth} mm |",
                    f"Optimizer: {optimizer_name} |",
                    f"Sensitivity: {optimized_sensitivity:.4f} |",
                    f"Epochs: {epochs} |",
                )
    return results


if __name__ == "__main__":
    # eval_func = lambda ppath, win, meas, noise: FetalSensitivityEvaluator(ppath, win, meas, 0.3, "fetal")
    # eval_func = lambda ppath, win, meas: CorrelationEvaluator(ppath, win, meas, 0.1)
    eval_func = lambda ppath, win, meas, noise: CorrelationxSNREvaluator(ppath, win, meas, noise)
    optimizer_funcs_to_test = [
        lambda tof_file, measurand: DIGSSOptimizer(tof_file, measurand),
        lambda tof_file, measurand: LiuOptimizer(tof_file, measurand, "median", normalize_window=True),
    ]
    exp_results = main(eval_func, optimizer_funcs_to_test, save=False)
    for result in exp_results:
        print(f"\nMeasurand: {result['Measurand']}, Depth: {result['Depth_mm']} mm, Optimizer: {result['Optimizer']}")
        print(f"Window: {result['Optimized_Window']}")
