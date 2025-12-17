"""
Choose the best detector for a given meauserand.
"""

from pathlib import Path
import yaml
from compute_sensitivity import compute_sensitivity
from joint_tof_opt import named_moment_types
from optimize_loop_paper import main_optimize
from generate_tof_set import generate_tof


def find_best_detector(
    ppath_dataset_path: Path, sdd_indices_to_try: list[int], measurand: str
) -> tuple[int, dict[int, float]]:
    """
    Find the best detector (SDD index) for a given measurand based on sensitivity.

    :param ppath_dataset_path: Path to the ToF dataset (.npz file).
    :type ppath_dataset_path: Path
    :param sdd_indices_to_try: List of SDD indices to evaluate.
    :type sdd_indices_to_try: list[int]
    :param measurand: The measurand to compute sensitivity for ("abs", "m1", "V").
    :type measurand: str
    :return: A tuple containing the best SDD index and a dictionary of sensitivities for each SDD index.
    :rtype: tuple[int, dict[int, float]]
    """
    gen_config = yaml.safe_load(open("./experiments/tof_config.yaml", "r"))
    temp_tofdataset_path = Path("./data/temp_generated_tof_set.npz")

    sensitivities = {}
    best_sensitivity = -float("inf")
    best_sdd_index = -1
    for sdd_index in sdd_indices_to_try:
        print(f"Evaluating SDD index: {sdd_index} for measurand: {measurand}")
        gen_config["selected_sdd_index"] = sdd_index
        generate_tof(ppath_dataset_path, gen_config, temp_tofdataset_path)
        window, losses = main_optimize(temp_tofdataset_path, measurand, lr=0.01)
        sensitivity, _ = compute_sensitivity(temp_tofdataset_path, window, measurand)
        
        # Debug code
        sensitivity = losses[-1, 2]
        
        sensitivities[sdd_index] = sensitivity
        if sensitivity > best_sensitivity:
            best_sensitivity = sensitivity
            best_sdd_index = sdd_index
    return best_sdd_index, sensitivities

if __name__ == "__main__":
    ppath_data = Path("./data/experiment_0004.npz")
    sdd_indices = [1, 2, 3, 4, 5]   # This is 1-indexed, higher index = larger SDD
    measurand_to_test = "V"
    best_sdd, all_sensitivities = find_best_detector(ppath_data, sdd_indices, measurand_to_test)
    print(f"Best SDD index for measurand '{measurand_to_test}': {best_sdd}")
    print(f"All sensitivities: {all_sensitivities}")
