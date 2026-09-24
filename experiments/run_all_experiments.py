"""Run every experiment entry point in sequence.

Purpose
-------
Orchestrator - calls the main entry point of every experiment script back-to-back: ablation_study,
best_detector, datalength_compare, false_fetal_f, noise_sensitivity_comparison, overlap_compare,
overlap_compare2, sensitivity_comparison. Convenience for running a batch overnight rather than
launching each script by hand.

Runtime
-------
Watch out, might take a while - runs every experiment in sequence; see each experiment's own
docstring for its individual runtime.

Inputs
------
See the Inputs section of each experiment listed in Purpose above.

Outputs
-------
See the Outputs section of each experiment listed in Purpose above.
"""

from collections.abc import Callable

from .ablation_study import main as ablation_study_main
from .best_detector import main as best_detector_main
from .false_fetal_f import main as false_fetal_f_main
from .noise_sensitivity_comparison import main as noise_sensitivity_comparison_main
from .overlap_compare import main as overlap_compare_main
from .overlap_compare2 import main as overlap_compare2_main
from .sensitivity_comparison import main as sensitivity_comparison_main


def run_all_experiments() -> None:
    experiment_mains: list[tuple[str, Callable[[], None]]] = [
        ("ablation_study", ablation_study_main),
        ("best_detector", best_detector_main),
        ("false_fetal_f", false_fetal_f_main),
        ("noise_sensitivity_comparison", noise_sensitivity_comparison_main),
        ("overlap_compare", overlap_compare_main),
        ("overlap_compare2", overlap_compare2_main),
        ("sensitivity_comparison", sensitivity_comparison_main),
    ]

    for experiment_name, experiment_main in experiment_mains:
        print(f"\n=== Running {experiment_name} ===")
        experiment_main()
        print(f"=== Completed {experiment_name} ===")


def main() -> None:
    run_all_experiments()


if __name__ == "__main__":
    main()
