"""
Compare the sensitivity of our metric between different detector indices

Purpose
-------
Sweeps SDD (source-detector distance) indices 1-7 and, for each, optimizes a window and evaluates
sensitivity, to find which detector position gives the best signal.

Runtime
-------
Slow - 7 SDD indices x every experiment in data/parameter_mapping.json, single-threaded.

Inputs
------
- experiments/tof_config.yaml
- data/parameter_mapping.json
- data/*.npz (ppath files listed in parameter_mapping.json)

Outputs
-------
- results/detector_comparison_results.yaml
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from joint_tof_opt import (
    CompactStatProcess,
    Evaluator,
    ToFConfig,
    ToFData,
    clear_results,
    generate_tof,
    load_parameter_mapping,
    load_tof_config,
    pretty_print_log,
    write_results_to_yaml,
)

from .optimize_loop_paper import DIGSSOptimizer
from .sensitivity_compute import AltPaperEvaluator3


def run_detector_comparison(
    evaluator_gen_func: Callable[[Path, torch.Tensor, str, ToFConfig], Evaluator],
    optimizers_to_compare: list[Callable[[ToFData, str | CompactStatProcess], DIGSSOptimizer]],
    sdd_indices_to_test: list[int],
    print_log: bool = False,
) -> list[dict[str, Any]]:
    """
    Main function to run sensitivity comparison experiments across measurands and depths.

    :param evaluator_gen_func: Function to generate an evaluator for sensitivity computation. The function should take
    (ppath_file: Path, window: torch.Tensor, measurand: nn.Module) and return an Evaluator instance.
    :type evaluator_gen_func: Callable[[Path, torch.Tensor, nn.Module], Evaluator]
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

    # Initialize results table and windows storage
    results = []
    measurand = "abs"  # Fixed measurand for this experiment
    base_gen_config = load_tof_config(Path("./experiments/tof_config.yaml"))
    for sdd_index in sdd_indices_to_test:
        gen_config = base_gen_config.model_copy(update={"selected_sdd_index": sdd_index})
        # Get the noise function for the measurand

        ## Run experiments
        file_sweep_params = load_parameter_mapping(Path("./data/parameter_mapping.json"))
        for ppath_filename, sweep_params in file_sweep_params.items():
            print(f"Running experiment: {ppath_filename} with SDD index: {sdd_index}")
            derm_thickness_mm = sweep_params["derm_thickness"]
            ppath_file: Path = Path("./data") / ppath_filename
            tof_data = generate_tof(ppath_file, gen_config)
            # Run Optimizers
            # measurand_module = get_named_moment_module(measurand, tof_series_tensor, bin_edges_tensor, meta_data)
            for optimizer_func in optimizers_to_compare:
                optimizer_experiment = optimizer_func(tof_data, measurand)
                optimizer_experiment.optimize()
                optimizer_name = str(optimizer_experiment)
                window = optimizer_experiment.window
                loss_history = optimizer_experiment.training_curves
                evaluator = evaluator_gen_func(ppath_file, window, measurand, gen_config)
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
                        # Not exactly needed right now - maybe useful later
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
                    f"Sensitivity: {optimized_sensitivity:.4e} |",
                    f"Epochs: {epochs} |",
                )
                if print_log:
                    log_dict = evaluator.get_log()
                    print("Log Details:")
                    pretty_print_log(log_dict)
    return results


def main() -> None:
    eval_func = lambda ppath, win, meas, conf: AltPaperEvaluator3(ppath, win, meas, conf)

    optimizer_funcs_to_test: list[Callable[[ToFData, str | CompactStatProcess], DIGSSOptimizer]] = [
        lambda tof_data, measurand: DIGSSOptimizer(tof_data, measurand, normalization_scheme="unit_max")
    ]
    # run_detector_comparison(eval_func, optimizer_funcs_to_test, [5, 6], print_log=False)
    exp_results = run_detector_comparison(eval_func, optimizer_funcs_to_test, [1, 2, 3, 4, 5, 6, 7], print_log=False)
    result_path = Path(__file__).parent.parent / "results" / "detector_comparison_results.yaml"
    clear_results(result_path)
    write_results_to_yaml(exp_results, result_path)


if __name__ == "__main__":
    main()
