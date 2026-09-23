"""
Compare optimizer and evaluator performance across data lengths and experiment files.

Purpose
-------
Sweeps time-series datapoint counts (5-30 heartbeat periods) across 8 experiment files, re-optimizing
a window each time, to see how optimizer/evaluator performance depends on how much data is available.

Runtime
-------
Slow - 8 experiment files x 6 datapoint counts = 48 full DIGSS optimizations, single-threaded.

Inputs
------
- experiments/tof_config.yaml
- data/experiment_0000.npz .. data/experiment_0007.npz

Outputs
-------
- results/datalength_compare_results.yaml
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from joint_tof_opt import (
    AltPaperEvaluator2,
    DIGSSOptimizer,
    PaperEvaluator,
    WindowSumWithAdditiveGaussianNoiseCalculator,
    build_noise_tof_modifier,
    evaluate_repeats,
    format_sensitivity,
    generate_tof,
    load_evaluator_specs,
    load_tof_config,
    noisy_results_path,
)


def _to_builtin(obj: Any) -> Any:
    """Convert numpy/torch scalars and containers to YAML-safe Python types."""
    if isinstance(obj, dict):
        return {str(k): _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


eval_spec = load_evaluator_specs(Path("./experiments/evaluator_specs.yaml"))


def main(inject_noise: bool = eval_spec.inject_noise) -> None:
    file_indices = list(range(8))
    datapoint_counts = [5 * 15 + 1, 10 * 15 + 1, 15 * 15 + 1, 20 * 15 + 1, 25 * 15 + 1, 30 * 15 + 1]
    measurand = "abs"
    output_yaml = noisy_results_path(Path("./results/datalength_compare_results.yaml"), inject_noise)

    results: dict[str, Any] = {}
    exp_idx = 0
    base_gen_config = load_tof_config(Path("./experiments/tof_config.yaml"))
    repeats = eval_spec.repeats_if_noisy if inject_noise else 1
    tof_modifier = build_noise_tof_modifier(eval_spec) if inject_noise else None
    noise_calc = WindowSumWithAdditiveGaussianNoiseCalculator(eval_spec.instrument_noise_variance)

    for file_idx in file_indices:
        ppath_file = Path(f"./data/experiment_{file_idx:04d}.npz")

        for datapoint_count in datapoint_counts:
            sampling_rate = float(base_gen_config.sampling_rate)
            end_sec = (int(datapoint_count) - 1) / sampling_rate
            gen_config = base_gen_config.model_copy(
                update={"datapoint_count": int(datapoint_count), "end_sec": end_sec}
            )

            tof_data = generate_tof(ppath_file, deepcopy(gen_config), True, True)

            experiment = DIGSSOptimizer(tof_data=tof_data, measurand=measurand)
            experiment.optimize()

            training_curves = experiment.training_curves
            best_final_metric = float(training_curves[-1, 2])
            best_selectivity = float(training_curves[-1, 0])
            best_snr = float(training_curves[-1, 1])
            epochs = int(training_curves.shape[0])

            evaluator1 = AltPaperEvaluator2(
                ppath_file,
                experiment.window,
                measurand,
                gen_config,
                noise_calc,
                eval_spec.alt_paper2.filter_hw,
                tof_modifier,
            )
            eval_results1, _ = evaluate_repeats(evaluator1, repeats)

            evaluator2 = PaperEvaluator(
                ppath_file,
                experiment.window,
                measurand,
                gen_config,
                noise_calc,
                eval_spec.paper.filter_hw,
                tof_modifier,
            )
            eval_results2, _ = evaluate_repeats(evaluator2, repeats)

            exp_key = f"exp {exp_idx:03d}"
            results[exp_key] = {
                "File_Idx": int(file_idx),
                "Datapoint_Count": int(datapoint_count),
                "End_Sec": float(end_sec),
                "Sampling_Rate_Hz": sampling_rate,
                "Filter_Type": str(experiment.filter_type),
                "Filter_HW": float(experiment.filter_hw),
                "Epochs": epochs,
                "Optimizer Best Metric": best_final_metric,
                "Optimizer Best Selectivity": best_selectivity,
                "Optimizer Best SNR": best_snr,
                "Sensitivity1": eval_results1,
                "Sensitivity2": eval_results2,
                "Experiment": str(experiment),
                "Optimized_Window": experiment.window.detach().cpu().tolist(),
            }

            print(
                f"[{exp_key}] file_idx={file_idx:04d} | datapoints={datapoint_count} | "
                f"end_sec={end_sec:.6g} | epochs={epochs} | best_metric={best_final_metric:.6g} | "
                f"eval_results1={format_sensitivity(eval_results1)} | eval_results2={format_sensitivity(eval_results2)}"
            )
            exp_idx += 1

    output_yaml.parent.mkdir(parents=True, exist_ok=True)
    with open(output_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(_to_builtin(results), f, sort_keys=False, default_flow_style=False)

    print(f"Saved data-length comparison results to: {output_yaml}")


if __name__ == "__main__":
    main()
