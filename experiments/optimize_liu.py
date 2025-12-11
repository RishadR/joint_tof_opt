"""
Discrete ToF Implementation of Shing-Jiuan's Optimization (https://doi.org/10.1364/BOE.500898)
Extended for any Measurand rather than just N_{tot} as described in the paper.
Also, improved SNR computation via MAD rather than simple std-dev.

For Implementation Details:
Implemented Using the Supplementary Material provided with the paper + Talking with her directly.
(Section 4: Segmentation of the time-of-flight (TOF) curve)
(https://opticapublishing.figshare.com/articles/journal_contribution/Supplementary_document_for_Recovering_fetal_signals_in_transabdominal_fetal_pulse_oximetry_through_interferometric_near-infrared_spectroscopy_iNIRS_-_6645474_pdf/24306511?file=52777772)

Steps:
1. Obtain the bin with the maximum count (bmax) over the average of all DTOFs
2. Obtain b0 and bf:
    a. b0 = first bin to the right of bmax with ~50% of bmax count (assuming falling edge)
    b. bf = first bin to the right of bmax with ~10% of bmax count (assuming falling edge)
3. Run a nested loop of all pairs of (b2, b3) where bf >= b3 > b2 >= b0
    a. For each (b2, b3) compute Measurand time series using a rectangular window between [b2, b3]
    b. Compute the FFT of the Measurand signal
    c. Get the FFT component at known FHR frequency
    d. Compute noise-floor via MAD while keeping the harmonics of FHR and MHR included (MAD Should be robust enough)
    e. Choose the pair that maxmizes SNR = Signal_at_FHR / Noise_Floor
4. Return the window corresponding to the best (b2, b3) pair but also maintain unit energy constraint

Notes:
1. I do both MAD and remove harmonics of FHR and MHR - I think only MAD might be sufficient but I kept both to be safe.
2. The original paper does not do MAD
3. Not sure which DTOF the original paper chooses to compute bmax and b0
4. Also, mine is a discrete DTOF implementation rather than continuous - that optimization will take forver to run
5. I do not consider her t_end.
"""

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Callable, Literal
from pathlib import Path
import matplotlib.pyplot as plt
from joint_tof_opt import get_named_moment_module


def liu_optimize(
    tof_dataset_path: Path,
    measurand: str | nn.Module,
    dtof_to_find_max_on: Literal["mean", "median", "first"] = "mean",
    fhr_hw: float = 0.3,
    harmonic_count: int = 2,
    normalize_window: bool = True,
) -> tuple[torch.Tensor, np.ndarray]:
    """
    The optimization loop implementation used in the paper.

    :param tof_dataset_path: Path to the ToF dataset (.npz file).
    :type tof_dataset_path: Path
    :param measurand: The measurand to optimize for ("abs", "m1", "V") or a custom moment module. If a custom module is
    provided, noise_func must also be provided.
    Predefined options:
        - "abs": Windowed Sum
        - "m1": First Order Moment
        - "V": Second Order Centered Moment (Variance)
    :type measurand: str | nn.Module
    :param dtof_to_find_max_on: Which DTOF to use to find bmax and b0. Options are "mean", "median", and "first".
    :type dtof_to_find_max_on: Literal["mean", "median", "first"]
    :param fhr_hw: Helps in computing SNR. How many Hz around FHR to consider when computing signal in SNR
    :type fhr_hw: float
    :param harmonic_count: Number of harmonics of FHR and MHR to exclude from noise calculation.
    :type harmonic_count: int
    :param normalize_window: Whether to normalize the output window to have unit energy. (Not used during optimization)
    :type normalize_window: bool
    :return: Tuple containing the optimized window tensor and an array of training curves (empty in this implementation)
    This implementation does not return training curves as the optimization is not iterative.
    :rtype: tuple[torch.Tensor, np.ndarray]
    """
    ## Data Loading
    data = np.load(tof_dataset_path)

    tof_series = data["tof_dataset"]  # Shape: (num_timepoints, num_bins)
    bin_edges = data["bin_edges"]  # Shape: (num_bins + 1,)
    sampling_rate = data["sampling_rate"]  # Sampling rate in Hz
    fetal_f = data["fetal_f"]  # Fetal heartbeat frequency in Hz
    maternal_f = data["maternal_f"]  # Maternal heartbeat frequency in Hz
    time_axis = data["time_axis"]  # Time axis
    tof_series_tensor = torch.tensor(tof_series, dtype=torch.float32)
    bin_edges_tensor = torch.tensor(bin_edges, dtype=torch.float32)
    num_timepoints, num_bins = tof_series_tensor.shape
    if isinstance(measurand, str):
        moment_calculator = get_named_moment_module(measurand, tof_series_tensor, bin_edges_tensor)
    else:
        moment_calculator = measurand

    fetal_bin = int(fetal_f / (sampling_rate / num_timepoints))
    maternal_bin = int(maternal_f / (sampling_rate / num_timepoints))
    fetal_bins = []
    for h in range(1, harmonic_count + 1):
        width_int_in_bins = int(fhr_hw / (sampling_rate / num_timepoints))
        left_edge = max(h * fetal_bin - width_int_in_bins, 0)
        right_edge = min(h * fetal_bin + width_int_in_bins, num_timepoints // 2 + 1)
        fetal_bins.extend(list(range(left_edge, right_edge + 1)))

    # Step 1: Find bmax
    if dtof_to_find_max_on == "mean":
        representative_dtof = torch.mean(tof_series_tensor, dim=0)
    elif dtof_to_find_max_on == "median":
        representative_dtof = torch.median(tof_series_tensor, dim=0).values
    elif dtof_to_find_max_on == "first":
        representative_dtof = tof_series_tensor[0, :]
    else:
        raise ValueError(f"Invalid dtof_to_find_max_on value: {dtof_to_find_max_on}")
    bmax = int(torch.argmax(representative_dtof).item())

    # Step 2: Find b0 (first bin to the right of bmax with ~50% of bmax count assuming a falling edge)
    half_max_value = representative_dtof[bmax] * 0.5
    b0 = bmax  # 50% point bin
    for b in range(bmax + 1, num_bins):
        if representative_dtof[b] <= half_max_value:
            b0 = b
            break
    bf = num_bins - 1 # bin where counts fall to 10% of max
    for b in range(num_bins - 1, bmax, -1):
        if representative_dtof[b] <= half_max_value * 0.1:
            bf = b
            break

    # Step 3: Nested loop over (b2, b3)
    best_snr = 0.0
    best_window = None
    for b2 in range(b0, bf):
        for b3 in range(b2 + 1, bf):    # Inclusive
            # Create rectangular window
            window = torch.zeros(num_bins, dtype=torch.float32)
            window[b2 : b3 + 1] = 1.0
            measurand_series = moment_calculator(window)
            measurand_series = measurand_series - torch.mean(measurand_series)  # Detrend
            measurand_fft = torch.fft.rfft(measurand_series)
            fetal_fft_component = float(measurand_fft[fetal_bins].abs().sum().item())
            # Compute noise floor using MAD
            # measurand_fft_without_excluded = measurand_fft[bins_to_include]
            measurand_fft_without_excluded = measurand_fft
            median_fft = torch.median(measurand_fft_without_excluded.abs()).item()
            mad_fft = torch.median(torch.abs(measurand_fft_without_excluded.abs() - median_fft)).item()
            noise_floor = mad_fft * 1.4826  # Convert MAD to std dev
            if noise_floor == 0:
                continue
            snr = fetal_fft_component / noise_floor
            if snr > best_snr:
                best_snr = snr
                best_window = window.clone()  # Clone this so we don't point to the changing tensor

    # Prepare Output
    if best_window is not None:
        if normalize_window:
            best_window = best_window / torch.norm(best_window)
    else:
        raise ValueError("Something went wrong with Liu optimization; no valid window that improves SNR above 0.")
    return best_window, np.array([])  # No training curves to return in this method since it's not iterative


if __name__ == "__main__":
    tof_dataset_path = Path("./data/generated_tof_set_experiment_0000.npz")
    optimized_window, training_curves = liu_optimize(tof_dataset_path=tof_dataset_path, measurand="abs")
    print("Optimized Window:", optimized_window.numpy())
