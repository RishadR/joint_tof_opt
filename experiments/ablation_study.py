"""
Ablation study for DIGSSOptimizer for two ablations:
  1. use_window_post_process: (aka Flat-top Projection) - applied post-optimization.
  2. use_snr_left_bound: whether the left fixed region starts at max_snr_index (True) or 0 (False).

Purpose
-------
Sweeps the 4 combinations of the two ablation flags above, at the instrument noise level from
experiments/evaluator_specs.yaml (20 parallel iterations), to see how much each trick contributes to
optimized-window sensitivity.

Runtime
-------
Watch out, might take a while - 20 parallel iterations x 4 optimizer configs x every experiment in
data/parameter_mapping.json.

Inputs
------
- experiments/tof_config.yaml
- data/parameter_mapping.json
- data/*.npz (ppath files listed in parameter_mapping.json)

Outputs
-------
- results/ablation_results.yaml
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from joint_tof_opt import (
    CompactStatProcess,
    DIGSSOptimizer,
    Evaluator,
    OptimizationExperiment,
    ToFConfig,
    ToFData,
    UnityTofModifier,
    WindowSumWithAdditiveGaussianNoiseCalculator,
    build_noise_tof_modifier,
    clear_results,
    evaluate_repeats,
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


def run_ablation(
    evaluator_gen_func: Callable[[Path, torch.Tensor, str, ToFConfig], Evaluator],
    optimizers_to_compare: list[Callable[[ToFData, str | CompactStatProcess], OptimizationExperiment]],
    measurands_to_test: list[str],
    noise_variance: float,
    repeats: int = 1,
    print_log: bool = False,
) -> list[dict[str, Any]]:
    gen_config = load_tof_config(Path("./experiments/tof_config.yaml"))
    # tof_modifier = AdditiveGaussianToFModifier(noise_var=noise_variance)
    tof_modifier = UnityTofModifier()

    results = []
    for measurand in measurands_to_test:
        file_sweep_params = load_parameter_mapping(Path("./data/parameter_mapping.json"))
        for ppath_filename, sweep_params in file_sweep_params.items():
            print(f"Running Experiment: {ppath_filename} | Measurand: {measurand}")
            derm_thickness_mm = sweep_params["derm_thickness"]
            ppath_file: Path = Path("./data") / ppath_filename
            tof_data = generate_tof(ppath_file, gen_config, True, True)
            tof_data = tof_modifier.modify(tof_data)

            for optimizer_func in optimizers_to_compare:
                optimizer_experiment = optimizer_func(tof_data, measurand)
                optimizer_experiment.optimize()
                optimizer_name = str(optimizer_experiment)
                window = optimizer_experiment.window.detach().cpu()
                loss_history = optimizer_experiment.training_curves
                evaluator = evaluator_gen_func(ppath_file, window, measurand, gen_config)
                optimized_sensitivity, evaluator_log = evaluate_repeats(evaluator, repeats)
                depth = derm_thickness_mm + 2
                epochs = len(loss_history)
                final_optimizer_loss = loss_history[-1, :].tolist() if epochs > 0 else []

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


def main(inject_noise: bool | None = None) -> None:
    eval_spec = load_evaluator_specs(Path("./experiments/evaluator_specs.yaml"))
    if inject_noise is None:
        inject_noise = eval_spec.inject_noise
    repeats = eval_spec.repeats_if_noisy if inject_noise else 1
    evaluator_cls = get_evaluator_class(eval_spec.evaluator_to_use)
    filter_hw = get_evaluator_filter_hw(eval_spec)
    tof_modifier = build_noise_tof_modifier(eval_spec) if inject_noise else None
    noise_var = eval_spec.instrument_noise_variance
    noise_calc = WindowSumWithAdditiveGaussianNoiseCalculator(noise_var)

    def eval_func(ppath: Path, win: torch.Tensor, meas: str, conf: ToFConfig) -> Evaluator:
        return evaluator_cls(ppath, win, meas, conf, noise_calc, filter_hw, tof_modifier)

    base_kwargs: dict[str, Any] = {
        "normalization_scheme": "unit_max",
        "noise_calc": noise_calc,
        "reg_weight": 0.0,
        "lr": 0.1,
        "window_smoothening": False,
    }

    optimizer_funcs_to_test: list[Callable[[ToFData, str | CompactStatProcess], OptimizationExperiment]] = [
        # Baseline: both ablations on
        lambda tof_data, measurand: DIGSSOptimizer(
            tof_data,
            measurand,
            **base_kwargs,
            use_window_post_process=True,
            use_snr_left_bound=True,
        ),
        # No post-process
        lambda tof_data, measurand: DIGSSOptimizer(
            tof_data,
            measurand,
            **base_kwargs,
            use_window_post_process=False,
            use_snr_left_bound=True,
        ),
        # No SNR left bound
        lambda tof_data, measurand: DIGSSOptimizer(
            tof_data,
            measurand,
            **base_kwargs,
            use_window_post_process=True,
            use_snr_left_bound=False,
        ),
        # Neither
        lambda tof_data, measurand: DIGSSOptimizer(
            tof_data,
            measurand,
            **base_kwargs,
            use_window_post_process=False,
            use_snr_left_bound=False,
        ),
    ]

    exp_results = run_ablation(eval_func, optimizer_funcs_to_test, ["abs"], noise_var, repeats=repeats, print_log=False)

    results_path = noisy_results_path(Path("./results/ablation_results.yaml"), inject_noise)
    clear_results(results_path)
    write_results_to_yaml(exp_results, results_path, append=True)


if __name__ == "__main__":
    main()
