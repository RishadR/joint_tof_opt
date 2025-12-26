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
from joint_tof_opt import get_named_moment_module, named_moment_types, OptimizationExperiment, CompactStatProcess


class LiuOptimizer(OptimizationExperiment):
    """
    Optimization experiment implementing the optimization loop from Liu et al.
    (https://doi.org/10.1364/BOE.500898)

    Extended for any measurand rather than just N_tot as described in the paper.
    Also uses improved SNR computation via MAD (Median Absolute Deviation) rather than simple std-dev.

    This implementation uses discrete ToF and finds the optimal rectangular window by:
    1. Finding the bin with maximum count (bmax) in a representative DTOF
    2. Determining boundary bins b0 and bf based on count thresholds
    3. Exhaustively searching all (b2, b3) window pairs to maximize SNR
    4. SNR is computed as Signal_at_FHR / Noise_Floor (using MAD)
    """

    def __init__(
        self,
        tof_dataset_path: Path,
        measurand: str | CompactStatProcess,
        dtof_to_find_max_on: Literal["mean", "median", "first"] = "mean",
        fhr_hw: float = 0.3,
        harmonic_count: int = 2,
        normalize_window: bool = True,
    ):
        """
        Initialize the LiuOptimizer.

        :param tof_dataset_path: Path to the ToF dataset (.npz file).
        :param measurand: The measurand to optimize for ("abs", "m1", "V") or custom module.
        :param dtof_to_find_max_on: Which DTOF to use to find bmax and b0 ("mean", "median", or "first").
        :param fhr_hw: Frequency half-width around FHR for signal extraction (in Hz).
        :param harmonic_count: Number of harmonics of FHR and MHR to exclude from noise calculation.
        :param normalize_window: Whether to normalize output window to unit energy.
        """
        if isinstance(measurand, str):
            tof_series_tensor = torch.tensor(np.load(tof_dataset_path)["tof_dataset"], dtype=torch.float32)
            bin_edges_tensor = torch.tensor(np.load(tof_dataset_path)["bin_edges"], dtype=torch.float32)
            measurand = get_named_moment_module(measurand, tof_series_tensor, bin_edges_tensor)
        super().__init__(tof_dataset_path, measurand)

        self.dtof_to_find_max_on = dtof_to_find_max_on
        self.fhr_hw = fhr_hw
        self.harmonic_count = harmonic_count
        self.normalize_window = normalize_window

        # Extract metadata
        self.sampling_rate = self.tof_data["sampling_rate"]
        self.fetal_f = self.tof_data["fetal_f"]
        self.maternal_f = self.tof_data["maternal_f"]

        # Pre-compute fetal bins for SNR calculation
        num_timepoints = self.tof_series.shape[0]
        self.fetal_bins = self._compute_fetal_bins(num_timepoints)

        # Set training curve labels (empty for this non-iterative method)
        self.training_curve_labels = []

    def _compute_fetal_bins(self, num_timepoints: int) -> list[int]:
        """
        Compute the FFT bin indices that correspond to FHR harmonics.

        :param num_timepoints: Number of time points in the signal.
        :return: List of bin indices to include in fetal signal extraction.
        """
        fetal_bin = int(self.fetal_f / (self.sampling_rate / num_timepoints))
        fetal_bins = []
        for h in range(1, self.harmonic_count + 1):
            width_int_in_bins = int(self.fhr_hw / (self.sampling_rate / num_timepoints))
            left_edge = max(h * fetal_bin - width_int_in_bins, 0)
            right_edge = min(h * fetal_bin + width_int_in_bins, num_timepoints // 2 + 1)
            fetal_bins.extend(list(range(left_edge, right_edge + 1)))
        return fetal_bins

    def __str__(self) -> str:
        return (
            f"LiuOptimizer(measurand={self.moment_module.__class__.__name__}, "
            f"dtof_to_find_max_on={self.dtof_to_find_max_on}, fhr_hw={self.fhr_hw})"
        )

    def components(self) -> dict[str, nn.Module]:
        """Return the internal components/modules used in optimization."""
        return {
            "moment_module": self.moment_module,
        }

    def optimize(self):
        """
        Perform the window optimization using Liu et al. approach.

        Exhaustively searches all rectangular window pairs (b2, b3) to find the one
        that maximizes SNR between fetal signal and noise floor.
        """
        num_timepoints, num_bins = self.tof_series.shape

        # Step 1: Find bmax (bin with maximum count)
        if self.dtof_to_find_max_on == "mean":
            representative_dtof = torch.mean(self.tof_series, dim=0)
        elif self.dtof_to_find_max_on == "median":
            representative_dtof = torch.median(self.tof_series, dim=0).values
        elif self.dtof_to_find_max_on == "first":
            representative_dtof = self.tof_series[0, :]
        else:
            raise ValueError(f"Invalid dtof_to_find_max_on value: {self.dtof_to_find_max_on}")

        self.bmax = int(torch.argmax(representative_dtof).item())

        # Step 2: Find b0 (50% of bmax) and bf (10% of bmax)
        half_max_value = representative_dtof[self.bmax] * 0.5
        self.b0 = self.bmax
        for b in range(self.bmax + 1, num_bins):
            if representative_dtof[b] <= half_max_value:
                self.b0 = b
                break

        self.bf = num_bins - 1
        for b in range(num_bins - 1, self.bmax, -1):
            if representative_dtof[b] <= half_max_value * 0.1:
                self.bf = b
                break

        # Step 3: Exhaustive search over all (b2, b3) window pairs
        best_snr = 0.0
        best_window = None

        for b2 in range(self.b0, self.bf):
            for b3 in range(b2 + 1, self.bf):
                # Create rectangular window
                window = torch.zeros(num_bins, dtype=torch.float32)
                window[b2 : b3 + 1] = 1.0

                # Compute measurand signal
                measurand_series = self.moment_module(window)
                measurand_series = measurand_series - torch.mean(measurand_series)  # Detrend

                # Compute FFT
                measurand_fft = torch.fft.rfft(measurand_series)
                fetal_fft_component = float(measurand_fft[self.fetal_bins].abs().sum().item())

                # Compute noise floor using MAD
                median_fft = torch.median(measurand_fft.abs()).item()
                mad_fft = torch.median(torch.abs(measurand_fft.abs() - median_fft)).item()
                noise_floor = mad_fft * 1.4826  # Convert MAD to std dev

                if noise_floor == 0:
                    continue

                # Compute SNR
                snr = fetal_fft_component / noise_floor
                if snr >= best_snr:  # Bias towards later windows
                    best_snr = snr
                    best_window = window.clone()
                    self.final_signal = measurand_series.detach().cpu()

        # Store results
        if best_window is not None:
            if self.normalize_window:
                self.window = best_window / torch.norm(best_window)
            else:
                self.window = best_window
        else:
            raise ValueError("Something went wrong; no valid window that improves SNR above 0.")

        # No training curves for this non-iterative method
        self.training_curves = np.array([])


def liu_optimize(
    tof_dataset_path: Path,
    measurand: str | CompactStatProcess,
    dtof_to_find_max_on: Literal["mean", "median", "first"] = "mean",
    fhr_hw: float = 0.3,
    harmonic_count: int = 2,
    normalize_window: bool = True,
) -> tuple[torch.Tensor, np.ndarray]:
    """
    The optimization loop implementation based on Liu et al. (https://doi.org/10.1364/BOE.500898).

    :param tof_dataset_path: Path to the ToF dataset (.npz file).
    :type tof_dataset_path: Path
    :param measurand: The measurand to optimize for ("abs", "m1", "V") or a custom moment module.
    Predefined options:
        - "abs": Windowed Sum
        - "m1": First Order Moment
        - "V": Second Order Centered Moment (Variance)
    :type measurand: str | nn.Module
    :param dtof_to_find_max_on: Which DTOF to use to find bmax and b0. Options are "mean", "medi    an", and "first".
    :type dtof_to_find_max_on: Literal["mean", "median", "first"]
    :param fhr_hw: Helps in computing SNR. How many Hz around FHR to consider when computing signal in SNR
    :type fhr_hw: float
    :param harmonic_count: Number of harmonics of FHR and MHR to exclude from noise calculation.
    :type harmonic_count: int
    :param normalize_window: Whether to normalize the output window to have unit energy.
    :type normalize_window: bool
    :return: Tuple containing the optimized window tensor and an array of training curves (empty in this implementation).
    This implementation does not return training curves as the optimization is not iterative.
    :rtype: tuple[torch.Tensor, np.ndarray]
    """
    optimizer = LiuOptimizer(
        tof_dataset_path=tof_dataset_path,
        measurand=measurand,
        dtof_to_find_max_on=dtof_to_find_max_on,
        fhr_hw=fhr_hw,
        harmonic_count=harmonic_count,
        normalize_window=normalize_window,
    )
    optimizer.optimize()
    return optimizer.window, optimizer.training_curves


if __name__ == "__main__":
    tof_dataset_path = Path("./data/generated_tof_set_experiment_0000.npz")
    optimized_window, training_curves = liu_optimize(tof_dataset_path=tof_dataset_path, measurand="abs")
    print("Optimized Window:", optimized_window.numpy())
