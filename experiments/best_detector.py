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
from optimize_loop_paper import main_optimize
from sensitivity_compute import (
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
    generate_tof
)
from optimize_liu import LiuOptimizer
from optimize_loop_paper import DIGSSOptimizer


def read_parameter_mapping():
    with open("./data/parameter_mapping.json", "r") as f:
        parameter_mapping = yaml.safe_load(f)
    return parameter_mapping


def main(
    evaluator_gen_func: Callable[[Path, torch.Tensor, nn.Module, Callable], Evaluator],
    sdd_indcies_to_test: list[int],
    save: bool = True,
) -> list[dict[str, Any]]:
    """
    Main function to run sensitivity comparison experiments across measurands and depths for different SDDs.
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
        noise_func = noise_func_table[measurand]
        
        ## Run experiments
        print(f"Starting sensitivity comparison for measurand: {measurand}")
        ppath_file_mapping = read_parameter_mapping()
        experiments = ppath_file_mapping["experiments"]
        for experiment in experiments:
            ppath_filename = experiment["filename"]
            derm_thickness_mm = experiment["sweep_parameters"]["derm_thickness"]["value"]
            ppath_file: Path = Path("./data") / ppath_filename

        # Run Optimizers
            for sdd_index in sdd_indcies_to_test:
                # Generate TOF DATA
                tof_dataset_file = Path("./data") / f"generated_tof_set_{ppath_file.stem}.npz"
                gen_config["selected_sdd_index"] = sdd_index
                generate_tof(ppath_file, gen_config, tof_dataset_file)
                tof_data = np.load(tof_dataset_file)
                meta_data = dict(tof_data)
                tof_series = tof_data["tof_dataset"]
                bin_edges = tof_data["bin_edges"]
                tof_series_tensor = torch.tensor(tof_series, dtype=torch.float32)
                bin_edges_tensor = torch.tensor(bin_edges, dtype=torch.float32)
                
                # Run Optimization            
                measurand_module = get_named_moment_module(measurand, tof_series_tensor, bin_edges_tensor, meta_data)
                # optimizer_experiment = DIGSSOptimizer(tof_dataset_file, measurand_module, noise_func, lr = lr)
                optimizer_experiment = LiuOptimizer(tof_dataset_file, measurand_module, "median")
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
                        "SDD_Index": sdd_index,
                    }
                )
                print(
                    f"Depth: {depth} mm |",
                    f"SDD Index: {sdd_index} |",
                    f"Sensitivity: {optimized_sensitivity:.4f} |",
                    f"Epochs: {epochs} |",
                )
    return results


if __name__ == "__main__":
    # eval_func = lambda ppath, win, meas, noise: FetalSensitivityEvaluator(ppath, win, meas, 0.3, "fetal")
    eval_func = lambda ppath, win, meas, noise: CorrelationEvaluator(ppath, win, meas, 0.1)
    # eval_func = lambda ppath, win, meas, noise: CorrelationxSNREvaluator(ppath, win, meas, noise)
    sdd_indcies_to_test = [1, 3]  # 1-indexing because that's how MCX does it
    
    exp_results = main(eval_func, sdd_indcies_to_test, save=False)
    # for result in exp_results:
    #     print(f"\nMeasurand: {result['Measurand']}, Depth: {result['Depth_mm']} mm, Optimizer: {result['Optimizer']}")
    #     print(f"Window: {result['Optimized_Window']}")
