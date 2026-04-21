"""Run selected experiment entry points in sequence."""

from __future__ import annotations

from typing import Callable

from best_detector import main as best_detector_main
from datalength_compare import main as datalength_compare_main
from false_fetal_f import main as false_fetal_f_main
from overlap_compare import main as overlap_compare_main
from overlap_compare2 import main as overlap_compare2_main
from sensitivity_comparison import main as sensitivity_comparison_main


def run_all_experiments() -> None:
    experiment_mains: list[tuple[str, Callable[[], None]]] = [
        ("best_detector", best_detector_main),
        # ("datalength_compare", datalength_compare_main),
        ("false_fetal_f", false_fetal_f_main),
        ("overlap_compare", overlap_compare_main),
        ("overlap_compare2", overlap_compare2_main),
        ("sensitivity_comparison", sensitivity_comparison_main),
    ]

    for experiment_name, experiment_main in experiment_mains:
        print(f"\\n=== Running {experiment_name}.main() ===")
        experiment_main()
        print(f"=== Completed {experiment_name}.main() ===")


def main() -> None:
    run_all_experiments()


if __name__ == "__main__":
    main()
