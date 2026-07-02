"""
Ablation study for DIGSSOptimizer.
Tests two ablations:
  1. use_window_post_process: whether the bridging post-process is applied after optimization.
  2. use_snr_left_bound: whether the left fixed region starts at max_snr_index (True) or 0 (False).
"""

import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import torch
import yaml

from joint_tof_opt import (
    AdditiveGaussianToFModifier,
    CompactStatProcess,
    Evaluator,
    OptimizationExperiment,
    ToFData,
    WindowSumWithAdditiveGaussianNoiseCalculator,
    generate_tof,
    pretty_print_log,
)
from joint_tof_opt.compact_stat_process import get_named_moment_module
from optimize_loop_paper import DIGSSOptimizer
from result_writer import clear_results, write_results_to_yaml
from sensitivity_compute import AltPaperEvaluator3

_tof_gen_locks: dict[Path, threading.Lock] = {}
_tof_gen_locks_mutex = threading.Lock()


def _tof_lock(path: Path) -> threading.Lock:
    with _tof_gen_locks_mutex:
        if path not in _tof_gen_locks:
            _tof_gen_locks[path] = threading.Lock()
        return _tof_gen_locks[path]


def read_parameter_mapping():
    with open("./data/parameter_mapping.json") as tof_config_file:
        parameter_mapping = yaml.safe_load(tof_config_file)
    return parameter_mapping


def run_ablation(
    evaluator_gen_func: Callable[[Path, torch.Tensor, str, dict], Evaluator],
    optimizers_to_compare: list[Callable[[ToFData, str | CompactStatProcess], OptimizationExperiment]],
    measurands_to_test: list[str],
    noise_variance: float,
    print_log: bool = False,
) -> tuple[list[dict[str, Any]], set[Path]]:
    gen_config = yaml.safe_load(open("./experiments/tof_config.yaml"))
    tof_modifier = AdditiveGaussianToFModifier(noise_var=noise_variance)

    results = []
    tof_files: set[Path] = set()
    for measurand in measurands_to_test:
        ppath_file_mapping = read_parameter_mapping()
        experiments = ppath_file_mapping["experiments"]
        for experiment in experiments:
            print(f"Running Experiment: {experiment['filename']} | Measurand: {measurand}")
            ppath_filename = experiment["filename"]
            derm_thickness_mm = experiment["sweep_parameters"]["derm_thickness"]["value"]
            ppath_file: Path = Path("./data") / ppath_filename
            tof_dataset_file = Path("./data") / f"generated_tof_set_{ppath_file.stem}.npz"
            with _tof_lock(tof_dataset_file):
                if not tof_dataset_file.exists():
                    generate_tof(ppath_file, gen_config, tof_dataset_file, True, True)
                tof_files.add(tof_dataset_file)
            noisy_tof_file = tof_dataset_file.with_stem(f"{tof_dataset_file.stem}_t{threading.get_ident()}")
            tof_data = ToFData.from_npz(tof_dataset_file)
            tof_data = tof_modifier.modify(tof_data)
            tof_data.to_npz(noisy_tof_file)

            for optimizer_func in optimizers_to_compare:
                optimizer_experiment = optimizer_func(tof_data, measurand)
                optimizer_experiment.optimize()
                optimizer_name = str(optimizer_experiment)
                window = optimizer_experiment.window.detach().cpu()
                loss_history = optimizer_experiment.training_curves
                evaluator = evaluator_gen_func(ppath_file, window, measurand, gen_config)
                optimized_sensitivity = evaluator.evaluate()
                depth = derm_thickness_mm + 2
                epochs = len(loss_history)
                final_optimizer_loss = loss_history[-1, :].tolist() if epochs > 0 else []

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
                    print("Log Details:")
                    pretty_print_log(evaluator.get_log())
            noisy_tof_file.unlink(missing_ok=True)
    return results, tof_files


def main(noise_var: float) -> tuple[list[dict[str, Any]], set[Path]]:
    filter_hw = 0.01  # Hz
    eval_func = lambda ppath, win, meas, conf: AltPaperEvaluator3(ppath, win, meas, conf, filter_hw, noise_var)
    noise_calc = WindowSumWithAdditiveGaussianNoiseCalculator(noise_var)

    base_kwargs = {
        "normalization_scheme": "unit_max",
        "noise_calc": noise_calc,
        "reg_weight": 0.0,
        "lr": 0.1,
        "window_smoothening": False,
    }

    optimizer_funcs_to_test: list[Callable[[ToFData, str | CompactStatProcess], OptimizationExperiment]] = [
        # Baseline: both ablations on
        lambda tof_data, measurand: DIGSSOptimizer(
            tof_data, measurand, **base_kwargs,
            use_window_post_process=True, use_snr_left_bound=True,
        ),
        # No post-process
        lambda tof_data, measurand: DIGSSOptimizer(
            tof_data, measurand, **base_kwargs,
            use_window_post_process=False, use_snr_left_bound=True,
        ),
        # No SNR left bound
        lambda tof_data, measurand: DIGSSOptimizer(
            tof_data, measurand, **base_kwargs,
            use_window_post_process=True, use_snr_left_bound=False,
        ),
        # Neither
        lambda tof_data, measurand: DIGSSOptimizer(
            tof_data, measurand, **base_kwargs,
            use_window_post_process=False, use_snr_left_bound=False,
        ),
    ]

    return run_ablation(eval_func, optimizer_funcs_to_test, ["abs"], noise_var, print_log=False)


if __name__ == "__main__":
    results_path = Path("./results/ablation_results.yaml")
    clear_results(results_path)
    noise_variances = [0.0, 10.0, 100.0, 1000.0, 10000.0]  # 1000.0 already computed
    iterations = 20
    all_tof_files: set[Path] = set()
    for noise_var in noise_variances:
        print(f"Running {iterations} iterations in parallel for noise_var={noise_var}...")
        with ThreadPoolExecutor(max_workers=iterations) as executor:
            futures = [executor.submit(main, noise_var) for _ in range(iterations)]
        for i, future in enumerate(futures):
            exp_results, tof_files = future.result()
            all_tof_files |= tof_files
            print(f"  Writing results: iteration {i + 1}/{iterations}")
            write_results_to_yaml(exp_results, results_path, append=True)
    for f in all_tof_files:
        f.unlink(missing_ok=True)
