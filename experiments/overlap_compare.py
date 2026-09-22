"""
Compare the performance when there is overlap between maternal and fetal frequencies.

Purpose
-------
Sweeps fetal-maternal frequency separation (0.01-0.7 Hz, fetal_f = 2*maternal_f + separation) and 3
filter setups (comb hw=0.10, comb hw=0.30, psafe_same_width), re-optimizing each time, to see how
filter choice handles closely-spaced maternal/fetal harmonics. Uses a single fixed experiment file
(file_idx=3).

Runtime
-------
Slow - 8 separations x 3 filter setups = 24 full DIGSS optimizations, single-threaded.

Inputs
------
- experiments/tof_config.yaml
- data/experiment_0003.npz

Outputs
-------
- results/overlap_results.yaml
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


def run_overlap_sweep(
    file_idx: int,
    separations_hz: list[float],
    filter_setups: list[tuple[FilterType, float]],  # (filter_type, filter_hw)
    measurand: str = "abs",
    output_yaml: Path = Path("./results/overlap_results.yaml"),
) -> dict[str, Any]:
    ppath_file = Path(f"./data/experiment_{file_idx:04d}.npz")
    results: dict[str, Any] = {}
    exp_idx = 0
    base_gen_config = load_tof_config(Path("./experiments/tof_config.yaml"))

    for separation in separations_hz:
        for filter_type, filter_hw in filter_setups:
            # Modify fetal_f for this run
            maternal_f = float(base_gen_config.maternal_f)
            fetal_f = 2 * maternal_f + float(separation)
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
                "Separation_Hz": float(separation),
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
                f"[{exp_key}] sep={separation:.3f} Hz | type={filter_type} | hw={filter_hw:.3f} | "
                f"fetal={fetal_f:.3f} Hz | epochs={epochs} | best_metric={best_final_metric:.6g} | "
                f"eval_results1={eval_results1:.6g} | eval_results2={eval_results2:.6g}"
            )
            exp_idx += 1

    output_yaml.parent.mkdir(parents=True, exist_ok=True)
    with open(output_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(results, f, sort_keys=False, default_flow_style=False)

    print(f"Saved overlap comparison results to: {output_yaml}")
    return results


def main() -> None:
    # separations = [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]  # Hz
    separations = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]  # Hz
    filter_combos: list[tuple[FilterType, float]] = [
        ("comb", 0.10),
        ("comb", 0.30),
        # ("comb", 0.50),
        ("psafe_same_width", 0.0),  # filter_hw not used for this filter type
    ]

    _ = run_overlap_sweep(
        file_idx=3,
        measurand="abs",
        separations_hz=separations,
        filter_setups=filter_combos,
        output_yaml=Path("./results/overlap_results.yaml"),
    )


if __name__ == "__main__":
    main()
