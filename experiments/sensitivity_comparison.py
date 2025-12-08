"""
Compare the Sensitivity between optmized vs. non-optimized windows and visualize the results.
"""

import yaml
import pandas as pd
import numpy as np
from generate_tof_set import generate_tof
from optimize_loop_paper import main_optimize
from compute_sensitivity import compute_sensitivity
from pathlib import Path
from joint_tof_opt import named_moment_types
import torch


def read_parameter_mapping():
    with open("./data/parameter_mapping.json", "r") as f:
        parameter_mapping = yaml.safe_load(f)
    return parameter_mapping


def main():
    ## Params
    filter_hw = 0.3  # Comb filter half-width in Hz
    lr_list = {"abs": 0.05, "m1": 0.01, "V": 0.01}  # Learning rates for different measurands

    # Initialize results table and windows storage
    results = []
    windows_data = {}  # Dictionary to store windows: {(measurand, depth): window_array}
    loss_history_data = {}  # Dictionary to store loss histories: {(measurand, depth): loss_array}
    bin_edges_data = {}  # Dictionary to store timebin edges: {(measurand, depth): edges_array}

    for measurand in named_moment_types:
    # for measurand in ['m1']:
        lr = lr_list.get(measurand, 0.01)

        ## Run experiments
        print(f"Starting sensitivity comparison for measurand: {measurand}")
        ppath_file_mapping = read_parameter_mapping()
        experiments = ppath_file_mapping["experiments"]
        for experiment in experiments:
            ppath_filename = experiment["filename"]
            derm_thickness_mm = experiment["sweep_parameters"]["derm_thickness"]["value"]
            ppath_file = Path("./data") / ppath_filename
            tof_dataset_file = Path("./data") / f"generated_tof_set_{ppath_file.stem}.npz"
            generate_tof(ppath_file, tof_dataset_file)
            window, loss_history = main_optimize(tof_dataset_file, measurand, filter_hw=filter_hw, lr=lr)
            vanilla_window = torch.ones_like(window)
            optimized_sensitivity, _ = compute_sensitivity(tof_dataset_file, window, measurand, filter_hw=filter_hw)
            vanilla_sensitivity, _ = compute_sensitivity(
                tof_dataset_file, vanilla_window, measurand, filter_hw=filter_hw
            )

            depth = derm_thickness_mm + 2  # Add 2 mm for epidermis
            improvement = (optimized_sensitivity - vanilla_sensitivity) / vanilla_sensitivity * 100
            epochs = len(loss_history)

            # Add row to results
            results.append(
                {
                    "Measurand": measurand,
                    "Depth": depth,
                    "Optimized Sensitivity": optimized_sensitivity,
                    "Vanilla Sensitivity": vanilla_sensitivity,
                    "Improvement": improvement,
                    "Epochs": epochs,
                }
            )

            # More logging - since I am a lumberjack apparently
            windows_data[(measurand, depth)] = window.detach().cpu().numpy()
            loss_history_data[(measurand, depth)] = loss_history
            bin_edges_data[(measurand, depth)] = np.load(tof_dataset_file)["bin_edges"]

            print(
                f"Depth: {depth} mm |",
                f"Optimized Sensitivity: {optimized_sensitivity:.3e} |",
                f"Vanilla Sensitivity: {vanilla_sensitivity:.3e}",
                f"Epochs: {epochs} |",
                f"Improvement: {improvement:.2f}%",
            )

    # Create DataFrame and save
    results_df = pd.DataFrame(results)
    results_df.to_csv("./results/sensitivity_comparison_results.csv", index=False)
    print("\nResults saved to ./results/sensitivity_comparison_results.csv")

    # Save windows as .npz file
    windows_dict = {}
    loss_history_dict = {}
    bin_edges_dict = {}
    for measurand, depth in windows_data.keys():
        key = f"{measurand}_depth_{depth}"
        windows_dict[key] = windows_data[(measurand, depth)]
        loss_history_dict[key] = np.array(loss_history_data[(measurand, depth)])
        bin_edges_dict[key] = bin_edges_data[(measurand, depth)]

    np.savez("./results/optimized_windows.npz", **windows_dict)
    np.savez("./results/loss_histories.npz", **loss_history_dict)
    np.savez("./results/timebin_edges.npz", **bin_edges_dict)
    print("Windows saved to ./results/optimized_windows.npz")
    print("Loss histories saved to ./results/loss_histories.npz")
    print("Timebin edges saved to ./results/timebin_edges.npz")


if __name__ == "__main__":
    main()
