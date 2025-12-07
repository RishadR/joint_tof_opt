"""
Code to compute the sensitivity of some measurand w.r.t. some window

Sensitivity is defined as the change in measurand per unit change in mu_a (in dL/M)
You can compare sensitivities across different windows to see which window gives better

For our case, the measurand is a mix between fetal and maternal hemoglobin changes - so sensitivities are computed
via a bandpass comb filter and an energy ratio metric. Which is further square rooted to get sensitivity in terms
of amplitude.

Process flow:
1. Read the Config file (tof_config.yaml) and load the generated ToF dataset
2. Use the given window to compute the measurand time series
3. Get the fetal_hb_series and maternal_hb_series from the dataset
4. Extract the fetal and maternal componets of the measurand using comb filters @FHR and MHR
5. Remove the DC from fetal_hb_series and maternal_hb_series
6. Compute Fetal Sensitivity as sqrt(energy ratio between measurand fetal component and fetal_hb_series)
7. Compute Maternal Sensitivity as sqrt(energy ratio between measurand maternal component and maternal_hb_series)
8. Return the two sensitivities
"""

from typing import Literal
import torch
import yaml
import numpy as np
from joint_tof_opt.compact_stat_process import NthOrderMoment, NthOrderCenteredMoment, WindowedSum
from joint_tof_opt.signal_process import CombSeparator
from joint_tof_opt.metric_process import EnergyRatioMetric


def compute_sensitivity(window: torch.Tensor, measurand: Literal["abs", "m1", "V"]) -> tuple[float, float]:
    """
    Compute the fetal and maternal sensitivities for a given window.

    :param window: The window to be applied to the ToF dataset. (Will be energy normalized internally)
    :type window: torch.Tensor
    :return: tuple: (fetal_sensitivity, maternal_sensitivity)
    :rtype: tuple[float, float]
    """
    # Step 1
    # Load Config
    with open("./experiments/tof_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    save_path = config["save_path"]

    ## Load Dataset
    dataset = np.load(f"./data/{save_path}")
    tof_dataset = dataset["tof_dataset"]
    bin_edges = dataset["bin_edges"]
    time_axis = dataset["time_axis"]
    maternal_f = dataset["maternal_f"]
    fetal_f = dataset["fetal_f"]
    sampling_rate = dataset["sampling_rate"]
    num_timepoints, num_bins = tof_dataset.shape
    assert len(window) == num_bins, "Window size must match the number of ToF bins."
    tof_series_tensor = torch.tensor(tof_dataset, dtype=torch.float32)
    bin_edges_tensor = torch.tensor(bin_edges, dtype=torch.float32)

    # Step 2
    ## Initialize moment calculator based on measurand
    if measurand not in ["abs", "m1", "V"]:
        raise ValueError(f"Invalid measurand: {measurand}")
    moment_calculator_table = {
        "abs": WindowedSum(tof_series_tensor, bin_edges_tensor),
        "m1": NthOrderMoment(tof_series_tensor, bin_edges_tensor, order=1),
        "V": NthOrderCenteredMoment(tof_series_tensor, bin_edges_tensor, order=2),
    }
    moment_calculator = moment_calculator_table[measurand]
    window = window.reshape(1, -1) / (torch.norm(window) + 1e-20)  # Unit norm
    compact_stats = moment_calculator(window)  # Shape: (num_timepoints,)
    compact_stats_reshaped = compact_stats.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, num_timepoints)

    # Step 3
    maternal_hb_series = dataset["maternal_hb_series"]
    fetal_hb_series = dataset["fetal_hb_series"]

    # Step 4
    ## Initialize comb filters and metrics
    fetal_comb_filter = CombSeparator(
        fs=sampling_rate,
        f0=fetal_f,
        f1=2 * fetal_f,
        half_width=0.3,
        filter_length=len(time_axis) // 2 + 1,
    )
    maternal_comb_filter = CombSeparator(
        fs=sampling_rate,
        f0=maternal_f,
        f1=2 * maternal_f,
        half_width=0.3,
        filter_length=len(time_axis) // 2 + 1,
    )
    fetal_filtered_signal = fetal_comb_filter(compact_stats_reshaped)
    maternal_filtered_signal = maternal_comb_filter(compact_stats_reshaped)

    # Step 5
    maternal_hb_series = maternal_hb_series - np.mean(maternal_hb_series)
    fetal_hb_series = fetal_hb_series - np.mean(fetal_hb_series)
    maternal_hb_series_tensor = torch.tensor(maternal_hb_series, dtype=torch.float32)
    fetal_hb_series_tensor = torch.tensor(fetal_hb_series, dtype=torch.float32)

    # Step 6 & 7
    energy_ratio_metric = EnergyRatioMetric()
    fetal_sensitivity = torch.sqrt(energy_ratio_metric(fetal_filtered_signal, fetal_hb_series_tensor)).item()
    maternal_sensitivity = torch.sqrt(energy_ratio_metric(maternal_filtered_signal, maternal_hb_series_tensor)).item()
    return fetal_sensitivity, maternal_sensitivity


if __name__ == "__main__":
    # Example usage
    window_size = 20  # Has to be the same size as TOF Bin count
    windows = {
        "abs": torch.ones(window_size, dtype=torch.float32),
        "V": torch.ones(window_size, dtype=torch.float32),
        "m1": torch.ones(window_size, dtype=torch.float32),
    }
    win = torch.tensor(
        [
            0.02256192,
            0.11523294,
            0.2057083,
            0.32318896,
            0.3580648,
            0.42087367,
            0.38735074,
            0.33061472,
            0.29306746,
            0.25933307,
            0.22063763,
            0.15431361,
            0.15209313,
            0.11616065,
            0.07943211,
            0.05635443,
            0.03971517,
            0.02958288,
            0.01846677,
            0.01460959,
        ],
        dtype=torch.float32,
    )
    windows["V"] = win

    print("Computing Sensitvities for all measurands with example window...")
    for measurand in ["abs", "m1", "V"]:
        fetal_sens, maternal_sens = compute_sensitivity(windows[measurand], measurand)  # type: ignore[arg-type]
        print(f"{measurand} | Fetal Sensitivity: {fetal_sens:.3e} | Maternal Sensitivity: {maternal_sens:.3e}")
