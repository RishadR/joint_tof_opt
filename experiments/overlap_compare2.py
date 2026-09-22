"""
Compare the performance across different tissue depths (file_idx sweep) when there is overlap
between maternal and fetal.

Purpose
-------
Companion to overlap_compare.py: instead of sweeping separation, fixes fetal-maternal separation at
0.5 Hz and sweeps tissue depth across all 8 experiment files x 3 filter setups (comb hw=0.10, comb
hw=0.30, psafe_same_width), to see how filter choice performs at different depths.

Runtime
-------
Slow - 8 depths x 3 filter setups = 24 full DIGSS optimizations, single-threaded.

Inputs
------
- experiments/tof_config.yaml
- data/parameter_mapping.json
- data/experiment_0000.npz .. data/experiment_0007.npz

Outputs
-------
- results/overlap_results2.yaml
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
    FilterType,
    PaperEvaluator,
    WindowSumNoiseCalculator,
    generate_tof,
    load_parameter_mapping,
    load_tof_config,
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


def _get_depth_mm(file_idx: int, param_mapping_path: Path) -> float:
    """Extract depth_mm from parameter_mapping.json for given file_idx."""
    file_sweep_params = load_parameter_mapping(param_mapping_path)
    filename = f"experiment_{file_idx:04d}.npz"
    if filename not in file_sweep_params:
        raise ValueError(f"file_idx {file_idx} not found in parameter_mapping.json")
    derm_thickness = file_sweep_params[filename]["derm_thickness"]
    return float(derm_thickness + 2)


def run_depth_sweep(
    file_idx_list: list[int],
    separation_hz: float,
    filter_setups: list[tuple[FilterType, float]],  # (filter_type, filter_hw)
    measurand: str = "abs",
    param_mapping_path: Path = Path("./data/parameter_mapping.json"),
    output_yaml: Path = Path("./results/overlap_results2.yaml"),
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    exp_idx = 0
    base_gen_config = load_tof_config(Path("./experiments/tof_config.yaml"))

    for file_idx in file_idx_list:
        ppath_file = Path(f"./data/experiment_{file_idx:04d}.npz")
        depth_mm = _get_depth_mm(file_idx, param_mapping_path)

        for filter_type, filter_hw in filter_setups:
            # Modify fetal_f for this run
            maternal_f = float(base_gen_config.maternal_f)
            fetal_f = 2 * maternal_f + float(separation_hz)
            gen_config = base_gen_config.model_copy(update={"fetal_f": fetal_f})

            tof_data = generate_tof(ppath_file, deepcopy(gen_config), True, True)

            experiment = DIGSSOptimizer(
                tof_data=tof_data, measurand=measurand, filter_hw=float(filter_hw), filter_type=filter_type
            )
            experiment.optimize()

            training_curves = experiment.training_curves
            best_final_metric = float(training_curves[-1, 2])
            best_selectivity = float(training_curves[-1, 0])
            best_snr = float(training_curves[-1, 1])
            epochs = int(training_curves.shape[0])

            noise_calc = WindowSumNoiseCalculator()
            evaluator1 = AltPaperEvaluator2(ppath_file, experiment.window, measurand, gen_config, noise_calc, 0.01)
            evaluator1.evaluate()
            eval_log1 = evaluator1.get_log()
            eval_results1 = float(eval_log1["final_metric"])
            # eval_results1 = float(eval_log1["fetal_ac_energy"] / eval_log1["maternal_ac_energy"])
            evaluator2 = PaperEvaluator(ppath_file, experiment.window, measurand, gen_config, noise_calc, 0.01)
            evaluator2.evaluate()
            eval_log2 = evaluator2.get_log()
            # eval_results2 = float(eval_log2["fetal_ac_energy"] / eval_log2["maternal_ac_amp"] ** 2)
            eval_results2 = float(eval_log2["final_metric"])

            exp_key = f"exp {exp_idx:03d}"
            results[exp_key] = {
                "File_Idx": file_idx,
                "Depth_mm": depth_mm,
                "Separation_Hz": float(separation_hz),
                "Filter_Type": str(filter_type),
                "Filter_HW": float(filter_hw),
                "Maternal_F_Hz": maternal_f,
                "Fetal_F_Hz": fetal_f,
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
                f"[{exp_key}] file_idx={file_idx} | depth={depth_mm:.1f} mm | type={filter_type} | "
                f"hw={filter_hw:.3f} | fetal={fetal_f:.3f} Hz | epochs={epochs} | "
                f"best_metric={best_final_metric:.6g} | eval_results1={eval_results1:.6g} | "
                f"eval_results2={eval_results2:.6g}"
            )
            exp_idx += 1

    output_yaml.parent.mkdir(parents=True, exist_ok=True)
    with open(output_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(results, f, sort_keys=False, default_flow_style=False)

    print(f"Saved depth sweep results to: {output_yaml}")
    return results


def main() -> None:
    file_indices = list(range(8))  # 0 to 7
    separation = 0.5  # Hz - fixed separation
    filter_combos: list[tuple[FilterType, float]] = [
        ("comb", 0.10),
        ("comb", 0.30),
        # ("comb", 0.50),
        ("psafe_same_width", 0.0),  # filter_hw not used for this filter type
    ]

    run_depth_sweep(
        file_idx_list=file_indices,
        separation_hz=separation,
        measurand="abs",
        filter_setups=filter_combos,
    )


if __name__ == "__main__":
    main()
