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


def read_parameter_mapping():
    with open("./data/parameter_mapping.json", "r") as f:
        parameter_mapping = yaml.safe_load(f)
    return parameter_mapping


def main(
    evaluator_gen_func: Callable[[Path, torch.Tensor, str, dict, NoiseCalculator], Evaluator],
    optimizers_to_compare: list[Callable[[Path, str | CompactStatProcess, float], OptimizationExperiment]],
    percent_errors: list[float],
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
    :param percent_errors: List of percentage errors to test (e.g., [0.1, 0.2, 0.3]).
    :type percent_errors: list[float]
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
    for percent_error in percent_errors:
        gen_config_true = yaml.safe_load(open("./experiments/tof_config.yaml", "r"))
        true_fetal_f: float = gen_config_true["fetal_f"]
        error_value = true_fetal_f * percent_error
        new_fetal_f = true_fetal_f + error_value
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
            generate_tof(ppath_file, gen_config_true, tof_dataset_file)
            # Run Optimizers

            for optimizer_func in optimizers_to_compare:
                # Optimize with the new (errored) fetal F as the BPF Center Freq
                optimizer_experiment = optimizer_func(tof_dataset_file, measurand, new_fetal_f)
                optimizer_experiment.lr = lr
                optimizer_experiment.optimize()
                optimizer_name = str(optimizer_experiment)
                window = optimizer_experiment.window
                loss_history = optimizer_experiment.training_curves
                noise_calculator = get_noise_calculator(measurand)
                evaluator = evaluator_gen_func(ppath_file, window, measurand, gen_config_true, noise_calculator)
                optimized_sensitivity = evaluator.evaluate()
                depth = derm_thickness_mm + 2  # Add 2 mm for epidermis
                epochs = len(loss_history)
                results.append(
                    {
                        "Measurand": measurand,
                        "Percent_Error": percent_error,
                        "True_Fetal_F_Hz": true_fetal_f,
                        "Errored_Fetal_F_Hz": new_fetal_f,
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
    filter_hw = 0.01  # Hz
    eval_func = lambda ppath, win, meas, conf, noise_calc: PaperEvaluator(ppath, win, meas, conf)
    optimizer_funcs_to_test: list[Callable[[Path, str | CompactStatProcess, float], OptimizationExperiment]] = [
        lambda tof_file, measurand, new_fetal_f: DIGSSOptimizer(
            tof_file, measurand, fetal_f=new_fetal_f, normalize_tof=False, patience=100, filter_hw=filter_hw
        ),
        # lambda tof_file, measurand, new_fetal_f: LiuOptimizer(
        #     tof_file, measurand, fetal_f=new_fetal_f, dtof_to_find_max_on="mean", fhr_hw=0.3, harmonic_count=2, norm=1.0
        # ),
        # lambda tof_file, measurand, new_fetal_f: DummyOptimizationExperiment(tof_file, measurand, 1.0),
    ]
    error_rates = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.4, 0.45, 0.5]  # 5%, 10%, 15%, 20% error in fetal F
    exp_results = main(eval_func, optimizer_funcs_to_test, error_rates, print_log=False)
    results_dict = {f"exp {i:03d}": res for i, res in enumerate(exp_results)}
    with open("./results/false_fetal_f_results.yaml", "w") as f:
        yaml.dump(results_dict, f, default_flow_style=False)
