"""
Compare the performance across different tissue depths (file_idx sweep).
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any
import json

import numpy as np
import yaml

from joint_tof_opt import generate_tof
from sensitivity_compute import AltPaperEvaluator2, PaperEvaluator
from optimize_loop_paper import DIGSSOptimizer


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
    with open(param_mapping_path, "r", encoding="utf-8") as f:
        param_data = json.load(f)
    
    for exp in param_data["experiments"]:
        if exp["index"] == file_idx:
            derm_thickness = exp["sweep_parameters"]["derm_thickness"]["value"]
            return float(derm_thickness + 2)
    
    raise ValueError(f"file_idx {file_idx} not found in parameter_mapping.json")


def run_depth_sweep(
    file_idx_list: list[int],
    separation_hz: float,
    filter_setups: list[tuple[str, float]],  # (filter_type, filter_hw)
    measurand: str = "abs",
    lr: float = 0.1,
    patience: int = 50,
    reg_type: str = "l1",
    reg_weight: float = 0.1,
    param_mapping_path: Path = Path("./data/parameter_mapping.json"),
    output_yaml: Path = Path("./results/overlap_results2.yaml"),
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    exp_idx = 0

    for file_idx in file_idx_list:
        ppath_file = Path(f"./data/experiment_{file_idx:04d}.npz")
        depth_mm = _get_depth_mm(file_idx, param_mapping_path)
        
        for filter_type, filter_hw in filter_setups:
            # Read config each run, then modify fetal_f
            with open("./experiments/tof_config.yaml", "r", encoding="utf-8") as f:
                gen_config: dict[str, Any] = yaml.safe_load(f)

            maternal_f = float(gen_config["maternal_f"])
            fetal_f = 2 * maternal_f + float(separation_hz)
            gen_config["fetal_f"] = fetal_f

            sep_tag = f"{separation_hz:.3f}".replace(".", "p")
            hw_tag = f"{float(filter_hw):.3f}".replace(".", "p")
            type_tag = str(filter_type).replace(" ", "_")
            tof_dataset_path = (
                Path("./data")
                / f"generated_tof_set_{ppath_file.stem}_sep_{sep_tag}_{type_tag}_hw_{hw_tag}.npz"
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
            # eval_results1 = float(eval_log1["fetal_ac_energy"] / eval_log1["maternal_ac_energy"])
            evaluator2 = PaperEvaluator(ppath_file, experiment.window, measurand, gen_config, 0.01)
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


if __name__ == "__main__":
    file_indices = list(range(8))  # 0 to 7
    separation = 0.5  # Hz - fixed separation
    filter_combos = [
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
        reg_weight=0.001,
        reg_type='l2'
    )
