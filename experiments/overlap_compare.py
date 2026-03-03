"""
Compare the performance of all 4 methods for different levels of overlap(separation) between fetal and the second
harmonic of maternal for the deepest model (The last one!)
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
from sensitivity_compute import AltPaperEvaluator
from sensitivity_compute import AltPaperEvaluator2


def read_parameter_mapping():
    with open("./data/parameter_mapping.json", "r") as f:
        parameter_mapping = yaml.safe_load(f)
    return parameter_mapping


def main(
    evaluator_gen_func1: Callable[[Path, torch.Tensor, str, dict], Evaluator],
    evaluator_gen_func2: Callable[[Path, torch.Tensor, str, dict], Evaluator],
    optimizers_to_compare: list[Callable[[Path, str | CompactStatProcess], OptimizationExperiment]],
    fetal_f_separations: list[float],
    print_log: bool = False,
) -> list[dict[str, Any]]:
    ## Params
    # Initialize results table and windows storage
    results = []
    measurand = "abs"  # Fixed measurand for this experiment
    for separation in fetal_f_separations:
        ## Run experiments
        print(f"Starting sensitivity comparison for measurand: abs with fetal f separation: {separation} Hz")
        gen_config = yaml.safe_load(open("./experiments/tof_config.yaml", "r"))

        # Switch fetal f based on separation
        maternal_f = gen_config["maternal_f"]
        fetal_f_no_shift = 2 * maternal_f
        new_fetal_f = fetal_f_no_shift + separation
        gen_config["fetal_f"] = new_fetal_f

        fetal_filter = CombSeparator(
            gen_config["sampling_rate"],
            gen_config["fetal_f"],
            2 * gen_config["fetal_f"],
            0.3,
            gen_config["datapoint_count"] // 2 + 1,
            True,
        )
        ppath_file_mapping = read_parameter_mapping()
        experiments = ppath_file_mapping["experiments"]
        for experiment in [experiments[-1]]:
            ppath_filename = experiment["filename"]
            derm_thickness_mm = experiment["sweep_parameters"]["derm_thickness"]["value"]
            ppath_file: Path = Path("./data") / ppath_filename
            tof_dataset_file = Path("./data") / f"generated_tof_set_{ppath_file.stem}.npz"
            generate_tof(ppath_file, gen_config, tof_dataset_file)
            # Run Optimizers
            # measurand_module = get_named_moment_module(measurand, tof_series_tensor, bin_edges_tensor, meta_data)
            for optimizer_func in optimizers_to_compare:
                optimizer_experiment = optimizer_func(tof_dataset_file, measurand)
                optimizer_experiment.optimize()
                optimizer_name = str(optimizer_experiment)
                window = optimizer_experiment.window.detach().cpu()
                loss_history = optimizer_experiment.training_curves
                evaluator1 = evaluator_gen_func1(ppath_file, window, measurand, gen_config)
                evaluator2 = evaluator_gen_func2(ppath_file, window, measurand, gen_config)
                optimized_sensitivity1 = evaluator1.evaluate()
                optimized_sensitivity2 = evaluator2.evaluate()
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
                        "Separation_Hz": separation,
                        "Maternal_2nd_Harmonic_F_Hz": fetal_f_no_shift,
                        "Fetal_F_Hz": new_fetal_f,
                        "Optimizer": optimizer_name,
                        "Optimized_Sensitivity1": optimized_sensitivity1,
                        "Optimized_Sensitivity2": optimized_sensitivity2,
                        "Epochs": epochs,
                        "Bin_Edges": bin_edges.tolist(),
                        "Optimized_Window": window.numpy().tolist(),
                        "fetal_hb_series": tof_data.meta_data["fetal_hb_series"].tolist(),
                        "filtered_signal": filtered_signal.numpy().tolist(),
                        "evaluator_log1": evaluator1.get_log(),
                        "evaluator_log2": evaluator2.get_log(),
                        "final_optimizer_loss": final_optimizer_loss,
                        "measurand_time_series": measurand_time_series.numpy().tolist(),
                    }
                )
                print(
                    f"Depth: {depth} mm |",
                    f"Optimizer: {optimizer_name} |",
                    f"Separation: {separation} Hz |",
                    f"Sensitivity1: {optimized_sensitivity1:.4e} |",
                    f"Sensitivity2: {optimized_sensitivity2:.4e} |",
                    f"Epochs: {epochs} |",
                )
                if print_log:
                    log_dict1 = evaluator1.get_log()
                    log_dict2 = evaluator2.get_log()
                    print("Log Details for Evaluator 1:")
                    pretty_print_log(log_dict1)
                    print("Log Details for Evaluator 2:")
                    pretty_print_log(log_dict2)
    return results


if __name__ == "__main__":
    filter_hw = 0.1  # Hz
    eval_func1 = lambda ppath, win, meas, conf: AltPaperEvaluator2(ppath, win, meas, conf, filter_hw)
    eval_func2 = lambda ppath, win, meas, conf: PaperEvaluator(ppath, win, meas, conf, filter_hw)
    # eval_func = lambda ppath, win, meas, conf: PaperEvaluator(ppath, win, meas, conf, filter_hw)

    optimizer_funcs_to_test: list[Callable[[Path, str | CompactStatProcess], OptimizationExperiment]] = [
        lambda tof_file, measurand: DIGSSOptimizer(
            tof_file, measurand, normalize_tof=False, patience=100, l2_reg=0.0001, filter_hw=filter_hw, lr=0.01, filter_type="comb"
        ),
    ]
    separations_to_test = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    exp_results = main(eval_func1, eval_func2, optimizer_funcs_to_test, separations_to_test, print_log=False)
    results_dict = {f"exp {i:03d}": res for i, res in enumerate(exp_results)}
    with open("./results/overlap_comparison_results.yaml", "w") as f:
        yaml.dump(results_dict, f, default_flow_style=False)
