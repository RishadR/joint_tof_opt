"""
Compare figure of merit across different optimizers

Purpose
-------
Runs BoxCarOptimizer (other optimizers commented out) across every experiment file in
data/parameter_mapping.json for the "abs" measurand, 20 parallel iterations, to compare optimized vs.
vanilla window sensitivity.

Runtime
-------
Watch out, might take a while - 20 parallel iterations x every experiment in data/parameter_mapping.json.

Inputs
------
- experiments/tof_config.yaml
- data/parameter_mapping.json
- data/*.npz (ppath files listed in parameter_mapping.json)

Outputs
-------
- results/sensitivity_comparison_results.yaml
"""

import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import torch

from joint_tof_opt import (
    AdditiveGaussianToFModifier,
    CombSeparator,
    CompactStatProcess,
    Evaluator,
    OptimizationExperiment,
    ToFConfig,
    ToFData,
    WindowSumWithAdditiveGaussianNoiseCalculator,
    clear_results,
    generate_tof,
    load_parameter_mapping,
    load_tof_config,
    pretty_print_log,
    write_results_to_yaml,
)
from joint_tof_opt.compact_stat_process import get_named_moment_module

from .optimize_dummy import DummyOptimizationExperiment
from .optimize_liu import LiuOptimizer
from .optimize_liu_alt import AltLiuOptimizer
from .optimize_loop_boxcar import BoxCarOptimizer
from .optimize_loop_paper import DIGSSOptimizer
from .sensitivity_compute import AltPaperEvaluator3


def run_sensitivity_comparison(
    evaluator_gen_func: Callable[[Path, torch.Tensor, str, ToFConfig], Evaluator],
    optimizers_to_compare: list[Callable[[ToFData, str | CompactStatProcess], OptimizationExperiment]],
    measurands_to_test: list[str],
    noise_variance: float,
    print_log: bool = False,
) -> list[dict[str, Any]]:
    """
    Main function to run sensitivity comparison experiments across measurands and depths.

    :param evaluator_gen_func: Function to generate an evaluator for sensitivity computation. The function should take
    (ppath_file: Path, window: torch.Tensor, measurand: nn.Module, noise_func: NoiseCalculator) and return an
        Evaluator instance.
    :type evaluator_gen_func: Callable[[Path, torch.Tensor, nn.Module], Evaluator]
    :param optimizers_to_compare: List of optimizer functions to compare. Each function should take
    (tof_data: ToFData, measurand: CompactStatProcess) and return an OptimizationExperiment instance.
    :type optimizers_to_compare: list[Callable[[ToFData, CompactStatProcess], OptimizationExperiment]]
    :param measurands_to_test: List of measurand names to test (e.g., ['abs', 'm1', 'V']).
    :type measurands_to_test: list[str]
    :param print_log: Whether to print log messages during execution. (Default: False)
    :type print_log: bool
    :return: List of dictionaries containing results for each experiment.
    :rtype: list[dict[str, Any]]
    """
    ## Params
    gen_config = load_tof_config(Path("./experiments/tof_config.yaml"))
    tof_modifier = AdditiveGaussianToFModifier(noise_var=noise_variance)

    # Initialize results table and windows storage
    results = []
    for measurand in measurands_to_test:
        file_sweep_params = load_parameter_mapping(Path("./data/parameter_mapping.json"))
        for ppath_filename, sweep_params in file_sweep_params.items():
            print(f"Running Experiment: {ppath_filename} | Measurand: {measurand}")
            derm_thickness_mm = sweep_params["derm_thickness"]
            ppath_file: Path = Path("./data") / ppath_filename
            tof_data = generate_tof(ppath_file, gen_config, True, True)
            noisy_tof_file = Path("./data") / f"generated_tof_set_{ppath_file.stem}_t{threading.get_ident()}.npz"
            tof_data = tof_modifier.modify(tof_data)
            tof_data.to_npz(noisy_tof_file)

            # Run Optimizers
            # measurand_module = get_named_moment_module(measurand, tof_series_tensor, bin_edges_tensor, meta_data)
            for optimizer_func in optimizers_to_compare:
                optimizer_experiment = optimizer_func(tof_data, measurand)
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
                tof_data = ToFData.from_npz(noisy_tof_file)
                assert tof_data.meta_data is not None, "ToFData meta_data was not found!"
                bin_edges = tof_data.bin_edges
                measurand_process = get_named_moment_module(measurand, tof_data)
                measurand_time_series = measurand_process.forward(window)

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
            noisy_tof_file.unlink(missing_ok=True)
    return results


def main() -> list[dict[str, Any]]:
    filter_hw = 0.01  # Hz
    noise_var = 100.0
    eval_func = lambda ppath, win, meas, conf: AltPaperEvaluator3(ppath, win, meas, conf, filter_hw, noise_var)
    noise_calc = WindowSumWithAdditiveGaussianNoiseCalculator(noise_var)

    optimizer_funcs_to_test: list[Callable[[ToFData, str | CompactStatProcess], OptimizationExperiment]] = [
        # lambda tof_data, measurand: DIGSSOptimizer(
        #     tof_data,
        #     measurand,
        #     normalization_scheme="unit_max",
        #     noise_calc=noise_calc,
        #     reg_weight=0.0,
        #     lr=0.1,
        #     window_smoothening=False,
        # ),
        lambda tof_data, measurand: BoxCarOptimizer(
            tof_data,
            measurand,
            normalization_scheme="unit_max",
            noise_calc=noise_calc,
            reg_weight=0.0,
            lr=0.1,
            window_smoothening=False,
        ),
        # lambda tof_data, measurand: LiuOptimizer(tof_data, measurand, None, "mean", filter_hw, 2, None),
        # lambda tof_data, measurand: AltLiuOptimizer(tof_data, measurand, None, None, "mean", filter_hw, 2, None),
        # lambda tof_data, measurand: DummyOptimizationExperiment(tof_data, measurand, None),
    ]

    return run_sensitivity_comparison(eval_func, optimizer_funcs_to_test, ["abs"], noise_var, print_log=True)


if __name__ == "__main__":
    results_path = Path("./results/sensitivity_comparison_results.yaml")
    # clear_results(results_path)   # Clears older results - otherwise appends to the existing results file
    iterations = 20
    print(f"Running {iterations} iterations in parallel...")
    with ThreadPoolExecutor(max_workers=iterations) as executor:
        futures = [executor.submit(main) for _ in range(iterations)]
    for i, future in enumerate(futures):
        exp_results = future.result()
        print(f"Writing results: iteration {i + 1}/{iterations}")
        write_results_to_yaml(exp_results, results_path, append=True)
