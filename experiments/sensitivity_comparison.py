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

from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from joint_tof_opt import (
    AltLiuOptimizer,
    BoxCarOptimizer,
    CompactStatProcess,
    DIGSSOptimizer,
    DummyOptimizationExperiment,
    Evaluator,
    LiuOptimizer,
    OptimizationExperiment,
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
from joint_tof_opt.compact_stat_process import get_named_moment_module

from .experiments_core import run_noisy_repeats


def run_sensitivity_comparison(
    evaluator_gen_func: Callable[[Path, torch.Tensor, str, ToFConfig], Evaluator],
    optimizers_to_compare: list[Callable[[ToFData, str | CompactStatProcess], OptimizationExperiment]],
    measurands_to_test: list[str],
    noise_variance: float,
    repeats: int = 1,
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

    # Initialize results table and windows storage
    results = []
    for measurand in measurands_to_test:
        file_sweep_params = load_parameter_mapping(Path("./data/parameter_mapping.json"))
        for ppath_filename, sweep_params in file_sweep_params.items():
            print(f"Running Experiment: {ppath_filename} | Measurand: {measurand}")
            derm_thickness_mm = sweep_params["derm_thickness"]
            ppath_file: Path = Path("./data") / ppath_filename
            base_tof_data = generate_tof(ppath_file, gen_config, True, True)

            # Run Optimizers
            for optimizer_func in optimizers_to_compare:
                # Repeat the full noisy-training + eval cycle `repeats` times (matching repeats_if_noisy) - see
                # experiments/experiments_core.py.
                optimizer_experiment, [optimized_sensitivity], [evaluator_log] = run_noisy_repeats(
                    base_tof_data,
                    build_optimizer=lambda tof_data: optimizer_func(tof_data, measurand),
                    evaluators_gen=lambda window: [evaluator_gen_func(ppath_file, window, measurand, gen_config)],
                    noise_variance=noise_variance,
                    repeats=repeats,
                )
                window = optimizer_experiment.window.detach().cpu()
                tof_data = optimizer_experiment.tof_data

                optimizer_name = str(optimizer_experiment)
                loss_history = optimizer_experiment.training_curves
                depth = derm_thickness_mm + 2  # Add 2 mm for epidermis
                epochs = len(loss_history)
                if epochs > 0:
                    final_optimizer_loss = loss_history[-1, :].tolist()
                else:
                    final_optimizer_loss = []

                # Compute the unfiltered measurand signal for logging
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
                        "evaluator_log": evaluator_log,
                        "final_optimizer_loss": final_optimizer_loss,
                        "measurand_time_series": measurand_time_series.numpy().tolist(),
                        "noise_variance": noise_variance,
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


def main(inject_noise: bool = eval_spec.inject_noise):
    repeats = eval_spec.repeats_if_noisy if inject_noise else 1
    evaluator_cls = get_evaluator_class(eval_spec.evaluator_to_use)
    filter_hw = get_evaluator_filter_hw(eval_spec)
    tof_modifier = build_noise_tof_modifier(eval_spec) if inject_noise else None
    instrument_noise_var = eval_spec.instrument_noise_variance
    noise_calc = WindowSumWithAdditiveGaussianNoiseCalculator(instrument_noise_var)

    def eval_func(ppath: Path, win: torch.Tensor, meas: str, conf: ToFConfig) -> Evaluator:
        return evaluator_cls(ppath, win, meas, conf, noise_calc, filter_hw, tof_modifier)

    optimizer_funcs_to_test: list[Callable[[ToFData, str | CompactStatProcess], OptimizationExperiment]] = [
        lambda tof_data, measurand: DIGSSOptimizer(tof_data, measurand, noise_calc=noise_calc),
        lambda tof_data, measurand: BoxCarOptimizer(tof_data, measurand, noise_calc=noise_calc),
        lambda tof_data, measurand: LiuOptimizer(tof_data, measurand),
        lambda tof_data, measurand: AltLiuOptimizer(tof_data, measurand),
        lambda tof_data, measurand: DummyOptimizationExperiment(tof_data, measurand),
    ]

    # Training-data noise is gated on inject_noise (unlike ablation_study.py/noise_sensitivity_comparison.py,
    # noise level isn't the swept variable here - it's either on at the spec's variance, or off).
    train_noise_variance = instrument_noise_var if inject_noise else 0.0
    exp_results = run_sensitivity_comparison(
        eval_func, optimizer_funcs_to_test, ["abs"], train_noise_variance, repeats=repeats, print_log=True
    )

    # Store results
    results_path = noisy_results_path(Path("./results/sensitivity_comparison_results.yaml"), inject_noise)
    clear_results(results_path)  # Clears older results - comment this to appends
    write_results_to_yaml(exp_results, results_path, append=True)


if __name__ == "__main__":
    main()
