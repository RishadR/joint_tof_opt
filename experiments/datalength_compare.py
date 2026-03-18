"""
Compare optimizer and evaluator performance across data lengths and experiment files.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from joint_tof_opt import generate_tof
from optimize_loop_paper import DIGSSOptimizer
from sensitivity_compute import AltPaperEvaluator2, PaperEvaluator


def _to_builtin(obj: Any) -> Any:
    """Convert numpy/torch scalars and containers to YAML-safe Python types."""
    if isinstance(obj, dict):
        return {str(k): _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def run_datalength_sweep(
    file_indices: list[int],
    datapoint_counts: list[int],
    measurand: str = "abs",
    lr: float = 0.1,
    filter_hw: float = 0.3,
    patience: int = 50,
    reg_type: str = "l1",
    reg_weight: float = 0.1,
    filter_type: str = "comb",
    output_yaml: Path = Path("./results/datalength_compare_results.yaml"),
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    exp_idx = 0

    for file_idx in file_indices:
        ppath_file = Path(f"./data/experiment_{file_idx:04d}.npz")

        for datapoint_count in datapoint_counts:
            with open("./experiments/tof_config.yaml", "r", encoding="utf-8") as f:
                gen_config: dict[str, Any] = yaml.safe_load(f)

            sampling_rate = float(gen_config["sampling_rate"])
            end_sec = (int(datapoint_count) - 1) / sampling_rate
            gen_config["datapoint_count"] = int(datapoint_count)
            gen_config["end_sec"] = end_sec

            datapoint_tag = f"{int(datapoint_count):04d}"
            tof_dataset_path = (
                Path("./data")
                / f"generated_tof_set_{ppath_file.stem}_datapoints_{datapoint_tag}.npz"
            )

            generate_tof(ppath_file, deepcopy(gen_config), tof_dataset_path, True, True)

            experiment = DIGSSOptimizer(
                tof_dataset_path=tof_dataset_path,
                measurand=measurand,
                lr=lr,
                filter_hw=float(filter_hw),
                patience=patience,
                reg_type=reg_type,  # type: ignore
                reg_weight=reg_weight,
                filter_type=filter_type,  # type: ignore
            )
            experiment.optimize()

            training_curves = experiment.training_curves
            best_final_metric = float(training_curves[-1, 2])
            best_selectivity = float(training_curves[-1, 0])
            best_snr = float(training_curves[-1, 1])
            epochs = int(training_curves.shape[0])

            evaluator1 = AltPaperEvaluator2(ppath_file, experiment.window, measurand, gen_config, 0.01)
            evaluator1.evaluate()
            eval_log1 = evaluator1.get_log()
            eval_results1 = float(eval_log1["final_metric"])

            evaluator2 = PaperEvaluator(ppath_file, experiment.window, measurand, gen_config, 0.01)
            evaluator2.evaluate()
            eval_log2 = evaluator2.get_log()
            eval_results2 = float(eval_log2["final_metric"])

            exp_key = f"exp {exp_idx:03d}"
            results[exp_key] = {
                "File_Idx": int(file_idx),
                "Datapoint_Count": int(datapoint_count),
                "End_Sec": float(end_sec),
                "Sampling_Rate_Hz": sampling_rate,
                "Filter_Type": str(filter_type),
                "Filter_HW": float(filter_hw),
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
                f"eval_results1={eval_results1:.6g} | eval_results2={eval_results2:.6g}"
            )
            exp_idx += 1
            tof_dataset_path.unlink()

    output_yaml.parent.mkdir(parents=True, exist_ok=True)
    with open(output_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(_to_builtin(results), f, sort_keys=False, default_flow_style=False)

    print(f"Saved data-length comparison results to: {output_yaml}")
    return results


if __name__ == "__main__":
    file_indices = list(range(8))
    datapoint_counts = [5 * 15 + 1, 10 * 15 + 1, 15 * 15 + 1, 20 * 15 + 1, 25 * 15 + 1, 30 * 15 + 1]

    _ = run_datalength_sweep(
        file_indices=file_indices,
        datapoint_counts=datapoint_counts,
        measurand="abs",
        filter_hw=0.3,
        filter_type="psafe_same_width",
        output_yaml=Path("./results/datalength_compare_results.yaml"),
        reg_weight=0.0001,
        reg_type="l1",
    )