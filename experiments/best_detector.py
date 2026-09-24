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
    DIGSSOptimizer,
    Evaluator,
    ToFConfig,
    ToFData,
    WindowSumWithAdditiveGaussianNoiseCalculator,
    build_noise_tof_modifier,
    clear_results,
    format_sensitivity,
    generate_tof,
    get_evaluator_class,
    get_evaluator_filter_hw,
    load_evaluator_specs,
    load_parameter_mapping,
    load_tof_config,
    noisy_results_path,
    print_evaluator_log,
    write_results_to_yaml,
)

from .experiments_core import run_noisy_repeats


def run_detector_comparison(
    evaluator_gen_func: Callable[[Path, torch.Tensor, str, ToFConfig], Evaluator],
    optimizers_to_compare: list[Callable[[ToFData, str | CompactStatProcess], DIGSSOptimizer]],
    sdd_indices_to_test: list[int],
    noise_variance: float = 0.0,
    repeats: int = 1,
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

        ## Run experiments
        file_sweep_params = load_parameter_mapping(Path("./data/parameter_mapping.json"))
        for ppath_filename, sweep_params in file_sweep_params.items():
            print(f"Running experiment: {ppath_filename} with SDD index: {sdd_index}")
            derm_thickness_mm = sweep_params["derm_thickness"]
            ppath_file: Path = Path("./data") / ppath_filename
            base_tof_data = generate_tof(ppath_file, gen_config)
            # Run Optimizers - repeat the full noisy-training + eval cycle `repeats` times (matching
            # repeats_if_noisy) - see experiments/experiments_core.py.
            for optimizer_func in optimizers_to_compare:
                optimizer_experiment, [optimized_sensitivity], [evaluator_log] = run_noisy_repeats(
                    base_tof_data,
                    build_optimizer=lambda tof_data: optimizer_func(tof_data, measurand),
                    evaluators_gen=lambda window: [evaluator_gen_func(ppath_file, window, measurand, gen_config)],
                    noise_variance=noise_variance,
                    repeats=repeats,
                )
                loss_history = optimizer_experiment.training_curves

                optimizer_name = str(optimizer_experiment)
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
                        "evaluator_log": evaluator_log,
                    }
                )
                print(
                    f"Depth: {depth} mm |",
                    f"Optimizer: {optimizer_name} |",
                    f"Sensitivity: {format_sensitivity(optimized_sensitivity)} |",
                    f"Epochs: {epochs} |",
                )
                if print_log:
                    print_evaluator_log(evaluator_log)
    return results


eval_spec = load_evaluator_specs(Path("./experiments/evaluator_specs.yaml"))


def main(inject_noise: bool = eval_spec.inject_noise) -> None:
    repeats = eval_spec.repeats_if_noisy if inject_noise else 1
    evaluator_cls = get_evaluator_class(eval_spec.evaluator_to_use)
    filter_hw = get_evaluator_filter_hw(eval_spec)
    tof_modifier = build_noise_tof_modifier(eval_spec) if inject_noise else None
    noise_calc = WindowSumWithAdditiveGaussianNoiseCalculator(eval_spec.instrument_noise_variance)

    def eval_func(ppath: Path, win: torch.Tensor, meas: str, conf: ToFConfig) -> Evaluator:
        return evaluator_cls(ppath, win, meas, conf, noise_calc, filter_hw, tof_modifier)

    optimizer_funcs_to_test: list[Callable[[ToFData, str | CompactStatProcess], DIGSSOptimizer]] = [
        lambda tof_data, measurand: DIGSSOptimizer(tof_data, measurand)
    ]
    # Training-data noise is gated on inject_noise - SDD index is the swept variable here, not noise level.
    train_noise_variance = eval_spec.instrument_noise_variance if inject_noise else 0.0
    exp_results = run_detector_comparison(
        eval_func,
        optimizer_funcs_to_test,
        [1, 2, 3, 4, 5],
        noise_variance=train_noise_variance,
        repeats=repeats,
        print_log=False,
    )
    result_path = noisy_results_path(
        Path(__file__).parent.parent / "results" / "detector_comparison_results.yaml", inject_noise
    )
    clear_results(result_path)
    write_results_to_yaml(exp_results, result_path)


if __name__ == "__main__":
    main()
