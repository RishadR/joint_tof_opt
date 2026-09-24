"""
Comparing our optimizers performance when the Fetal F is off by some margin

Purpose
-------
Sweeps error in the assumed fetal heartbeat frequency (0% to 100% of the true rate, in 5% steps) and
re-optimizes with the errored frequency as the BPF center, to see how sensitive the optimizer is to
getting fetal_f wrong. Tests 3 DIGSSOptimizer configs (comb filter, varying filter_hw and normalize_reward).

Runtime
-------
Slow - 21 error levels x 2 experiment files x 3 optimizer configs = 126 full DIGSS optimizations,
single-threaded.

Inputs
------
- experiments/tof_config.yaml
- data/parameter_mapping.json
- data/*.npz (first 2 ppath files listed in parameter_mapping.json)

Outputs
-------
- results/false_fetal_f_results2.yaml
"""

from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from joint_tof_opt import (
    CompactStatProcess,
    DIGSSOptimizer,
    Evaluator,
    ToFConfig,
    ToFData,
    WindowSumWithAdditiveGaussianNoiseCalculator,
    build_noise_tof_modifier,
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


def run_false_fetal_frequency_experiment(
    evaluator_gen_func: Callable[[Path, torch.Tensor, str, ToFConfig], Evaluator],
    optimizers_to_compare: list[Callable[[ToFData, str | CompactStatProcess, float], DIGSSOptimizer]],
    error_hzs: list[float],
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
    (tof_data: ToFData, measurand: CompactStatProcess) and return a DIGSSOptimizer instance.
    :type optimizers_to_compare: list[Callable[[ToFData, CompactStatProcess, float], DIGSSOptimizer]]
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
    gen_config_true = load_tof_config(Path("./experiments/tof_config.yaml"))
    for error_hz in error_hzs:
        print(f"Running experiments for fetal frequency error: {error_hz * 100:.1f}%")
        true_fetal_f: float = gen_config_true.fetal_f
        new_fetal_f = true_fetal_f - error_hz
        # Get the noise function for the measurand
        file_sweep_params = load_parameter_mapping(Path("./data/parameter_mapping.json"))
        for ppath_filename, sweep_params in list(file_sweep_params.items())[:2]:
            derm_thickness_mm = sweep_params["derm_thickness"]
            ppath_file: Path = Path("./data") / ppath_filename
            base_tof_data = generate_tof(ppath_file, gen_config_true)
            # Run Optimizers - repeat the full noisy-training + eval cycle `repeats` times (matching
            # repeats_if_noisy) - see experiments/experiments_core.py.

            for optimizer_func in optimizers_to_compare:
                # Optimize with the new (errored) fetal F as the BPF Center Freq
                optimizer_experiment, [optimized_sensitivity], [evaluator_log] = run_noisy_repeats(
                    base_tof_data,
                    build_optimizer=lambda tof_data: optimizer_func(tof_data, measurand, new_fetal_f),
                    evaluators_gen=lambda window: [
                        evaluator_gen_func(ppath_file, window, measurand, gen_config_true)
                    ],
                    noise_variance=noise_variance,
                    repeats=repeats,
                )
                window = optimizer_experiment.window.detach().cpu()
                loss_history = optimizer_experiment.training_curves

                optimizer_name = str(optimizer_experiment)
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
                        "evaluator_log": evaluator_log,
                        "Optimizer(Fetal Energy)": float(fetal_energy),
                        "Optimizer(Maternal Energy)": float(maternal_energy),
                        "Optimizer(Noise Std)": float(noise_std),
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
    results_path = noisy_results_path(Path("./results/false_fetal_f_results2.yaml"), inject_noise)
    evaluator_cls = get_evaluator_class(eval_spec.evaluator_to_use)
    filter_hw = get_evaluator_filter_hw(eval_spec)
    tof_modifier = build_noise_tof_modifier(eval_spec) if inject_noise else None
    noise_calc = WindowSumWithAdditiveGaussianNoiseCalculator(eval_spec.instrument_noise_variance)

    def eval_func(ppath: Path, win: torch.Tensor, meas: str, conf: ToFConfig) -> Evaluator:
        return evaluator_cls(ppath, win, meas, conf, noise_calc, filter_hw, tof_modifier)

    optimizer_funcs_to_test: list[Callable[[ToFData, str | CompactStatProcess, float], DIGSSOptimizer]] = [
        lambda tof_data, measurand, new_fetal_f: DIGSSOptimizer(
            tof_data, measurand, fetal_f=new_fetal_f, filter_hw=0.01, filter_type="comb"
        ),
        lambda tof_data, measurand, new_fetal_f: DIGSSOptimizer(
            tof_data, measurand, fetal_f=new_fetal_f, filter_hw=0.1, filter_type="comb"
        ),
        lambda tof_data, measurand, new_fetal_f: DIGSSOptimizer(
            tof_data, measurand, fetal_f=new_fetal_f, filter_hw=0.1, filter_type="comb", normalize_reward=False
        ),
    ]
    # error_rates = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]  # 5%, 10%, 15%, 20% error in fetal F
    error_rates_np = np.arange(0.0, 1.01, 0.05)
    error_rates = [float(x) for x in error_rates_np]
    # Training-data noise is gated on inject_noise - fetal_f error is the swept variable here, not noise level.
    train_noise_variance = eval_spec.instrument_noise_variance if inject_noise else 0.0
    exp_results = run_false_fetal_frequency_experiment(
        eval_func,
        optimizer_funcs_to_test,
        error_rates,
        noise_variance=train_noise_variance,
        repeats=repeats,
        print_log=False,
    )
    write_results_to_yaml(exp_results, results_path, append=False)


if __name__ == "__main__":
    main()
