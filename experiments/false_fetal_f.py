"""
Comparing our optimizers performance when the Fetal F is off by some margin
"""

from typing import Any, Literal, Callable
from pathlib import Path
import torch
import yaml
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
import numpy as np

def read_parameter_mapping():
    with open("./data/parameter_mapping.json", "r") as f:
        parameter_mapping = yaml.safe_load(f)
    return parameter_mapping


def main(
    evaluator_gen_func: Callable[[Path, torch.Tensor, str, dict], Evaluator],
    optimizers_to_compare: list[Callable[[Path, str | CompactStatProcess, float], DIGSSOptimizer]],
    error_hzs: list[float],
    print_log: bool = False,
) -> list[dict[str, Any]]:
    """
    Main function to run sensitivity comparison experiments across measurands and depths.

    :param evaluator_gen_func: Function to generate an evaluator for sensitivity computation. The function should take
    (ppath_file: Path, window: torch.Tensor, measurand: nn.Module) and return an Evaluator instance.
    :type evaluator_gen_func: Callable[[Path, torch.Tensor, nn.Module], Evaluator]
    :param optimizers_to_compare: List of optimizer functions to compare. Each function should take
    (ppath_file: Path, measurand: CompactStatProcess) and return a DIGSSOptimizer instance.
    :type optimizers_to_compare: list[Callable[[Path, CompactStatProcess, float], DIGSSOptimizer]]
    :param error_hzs: List of fetal frequency errors to test (e.g., [0.1, 0.2, 0.3]).
    :type error_hzs: list[float]
    :param print_log: Whether to print log messages during execution. (Default: False)
    :type print_log: bool
    :return: List of dictionaries containing results for each experiment.
    :rtype: list[dict[str, Any]]
    """
    # Initialize results table and windows storage
    results = []
    measurand = "abs"  # Fixed measurand for this experiment
    for error_hz in error_hzs:
        print(f"Running experiments for fetal frequency error: {error_hz*100:.1f}%")
        gen_config_true = yaml.safe_load(open("./experiments/tof_config.yaml", "r"))
        true_fetal_f: float = gen_config_true["fetal_f"]
        new_fetal_f = true_fetal_f + error_hz
        # Get the noise function for the measurand
        ppath_file_mapping = read_parameter_mapping()
        experiments = ppath_file_mapping["experiments"]
        for experiment in experiments[:2]:
            ppath_filename = experiment["filename"]
            derm_thickness_mm = experiment["sweep_parameters"]["derm_thickness"]["value"]
            ppath_file: Path = Path("./data") / ppath_filename
            tof_dataset_file = Path("./data") / f"generated_tof_set_{ppath_file.stem}.npz"
            generate_tof(ppath_file, gen_config_true, tof_dataset_file)
            # Run Optimizers

            for optimizer_func in optimizers_to_compare:
                # Optimize with the new (errored) fetal F as the BPF Center Freq
                optimizer_experiment = optimizer_func(tof_dataset_file, measurand, new_fetal_f)
                optimizer_experiment.optimize()
                optimizer_name = str(optimizer_experiment)
                window = optimizer_experiment.window.detach().cpu()
                loss_history = optimizer_experiment.training_curves
                evaluator = evaluator_gen_func(ppath_file, window, measurand, gen_config_true)
                optimized_sensitivity = evaluator.evaluate()
                fetal_energy = optimizer_experiment.training_curves_extra[-1, 0] 
                maternal_energy = optimizer_experiment.training_curves_extra[-1, 1]
                noise_std = optimizer_experiment.training_curves_extra[-1, 2]
                depth = derm_thickness_mm + 2  # Add 2 mm for epidermis
                epochs = len(loss_history)
                results.append(
                    {
                        "Measurand": measurand,
                        "Error": error_hz,
                        "True_Fetal_F_Hz": true_fetal_f,
                        "Errored_Fetal_F_Hz": new_fetal_f,
                        "Depth_mm": depth,
                        "Optimizer": optimizer_name,
                        "Optimized_Sensitivity": optimized_sensitivity,
                        "Epochs": epochs,
                        "Optimized_Window": window.numpy().tolist(),
                        "evaluator_log": evaluator.get_log(),
                        "Optimizer(Fetal Energy)": float(fetal_energy),
                        "Optimizer(Maternal Energy)": float(maternal_energy),
                        "Optimizer(Noise Std)": float(noise_std),
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
    filter_hw = 0.01  # Hz
    # eval_func = lambda ppath, win, meas, conf: PaperEvaluator(ppath, win, meas, conf)
    eval_func = lambda ppath, win, meas, conf: AltPaperEvaluator3(ppath, win, meas, conf, filter_hw)
    optimizer_funcs_to_test: list[Callable[[Path, str | CompactStatProcess, float], DIGSSOptimizer]] = [
        lambda tof_file, measurand, new_fetal_f: DIGSSOptimizer(tof_file, measurand, fetal_f=new_fetal_f, patience=100, filter_hw=0.01, lr=0.01, filter_type="comb",),
        lambda tof_file, measurand, new_fetal_f: DIGSSOptimizer(tof_file, measurand, fetal_f=new_fetal_f, patience=100, filter_hw=0.1, lr=0.01, filter_type="comb",),
        lambda tof_file, measurand, new_fetal_f: DIGSSOptimizer(tof_file, measurand, fetal_f=new_fetal_f, patience=100, filter_hw=0.1, lr=0.01, filter_type="comb", normalize_reward=False)
    ]
    # error_rates = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]  # 5%, 10%, 15%, 20% error in fetal F
    error_rates_np = np.arange(0.0, 0.61, 0.05)
    error_rates = [float(x) for x in error_rates_np]
    exp_results = main(eval_func, optimizer_funcs_to_test, error_rates, print_log=False)
    results_dict = {f"exp {i:03d}": res for i, res in enumerate(exp_results)}
    with open("./results/false_fetal_f_results.yaml", "w") as f:
        yaml.dump(results_dict, f, default_flow_style=False)
