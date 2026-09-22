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

import numpy as np
import torch
import torch.nn as nn

from joint_tof_opt.compact_stat_process import get_named_moment_module
from joint_tof_opt.core import CompactStatProcess, OptimizationExperiment, ToFData
from joint_tof_opt.optimizers.specs import DEFAULT_SPECS_PATH, DtofSelection, load_optimizer_specs

_LIU_SPEC = load_optimizer_specs(DEFAULT_SPECS_PATH).liu


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
        tof_data: ToFData,
        measurand: str | CompactStatProcess,
        fetal_f: float | None = None,
        dtof_to_find_max_on: DtofSelection = _LIU_SPEC.dtof_to_find_max_on,
        half_width: float = _LIU_SPEC.half_width,
        harmonic_count: int = _LIU_SPEC.harmonic_count,
        norm: None | float = _LIU_SPEC.norm,
    ):
        """
        Initialize the LiuOptimizer.

        :param tof_data: ToFData instance to optimize on.
        :param measurand: The measurand to optimize for ("abs", "m1", "V") or custom module.
        :param fetal_f: Central frequency of fetal comb filter (in Hz). If None, extracted from dataset metadata.
        :param dtof_to_find_max_on: Which DTOF to use to find bmax and b0 ("mean", "median", or "first").
        :param half_width: Frequency half-width around FHR for signal extraction (in Hz).
        :param harmonic_count: Number of harmonics of FHR and MHR to exclude from noise calculation.
        :param norm: If specified, normalizes the window to have this p-norm. Ex: norm=1 for L1 norm, norm=2 for L2 norm.
        """
        if isinstance(measurand, str):
            measurand = get_named_moment_module(measurand, tof_data)
        super().__init__(tof_data, measurand)

        self.dtof_to_find_max_on = dtof_to_find_max_on
        self.half_width = half_width
        self.harmonic_count = harmonic_count
        self.norm = norm

        # Extract metadata
        assert self.tof_data.meta_data is not None, "ToFData meta_data cannot be None"
        self.sampling_rate = self.tof_data.meta_data["sampling_rate"]
        self.fetal_f = fetal_f if fetal_f is not None else self.tof_data.meta_data["fetal_f"]
        self.maternal_f = self.tof_data.meta_data["maternal_f"]

        # Pre-compute fetal bins for SNR calculation
        num_timepoints = self.tof_data.tof_series.shape[0]
        self.fetal_bins = self._compute_fetal_bins(num_timepoints)

        # Set training curve labels (empty for this non-iterative method)
        self.training_curve_labels = ["Left Bin Index", "Right Bin Index", "SNR"]

    def _compute_fetal_bins(self, num_timepoints: int) -> list[int]:
        """
        Compute the FFT bin indices that correspond to FHR harmonics.

        :param num_timepoints: Number of time points in the signal.
        :return: List of bin indices to include in fetal signal extraction.
        """
        fetal_bin = int(self.fetal_f / (self.sampling_rate / num_timepoints))
        fetal_bins = []
        for h in range(1, self.harmonic_count + 1):
            width_int_in_bins = int(self.half_width / (self.sampling_rate / num_timepoints))
            left_edge = max(h * fetal_bin - width_int_in_bins, 0)
            right_edge = min(h * fetal_bin + width_int_in_bins, num_timepoints // 2 + 1)
            fetal_bins.extend(list(range(left_edge, right_edge + 1)))
        return fetal_bins

    def __str__(self) -> str:
        return (
            f"LiuOptimizer(measurand={self.moment_module.__class__.__name__}, "
            f"dtof_to_find_max_on={self.dtof_to_find_max_on}, half_width={self.half_width},"
            f"harmonics={self.harmonic_count}, norm={self.norm})"
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
        results = []
        num_timepoints, num_bins = self.tof_data.tof_series.shape

        # Step 1: Find bmax (bin with maximum count)
        if self.dtof_to_find_max_on == "mean":
            representative_dtof = torch.mean(self.tof_data.tof_series, dim=0)
        elif self.dtof_to_find_max_on == "median":
            representative_dtof = torch.median(self.tof_data.tof_series, dim=0).values
        elif self.dtof_to_find_max_on == "first":
            representative_dtof = self.tof_data.tof_series[0, :]
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
        best_window = torch.zeros(num_bins, dtype=torch.float32)
        best_window[self.b0] = 1.0  # Default to single bin window at b0 if no better window is found

        for b2 in range(self.b0, self.bf):
            for b3 in range(b2 + 1, self.bf):
                # Create rectangular window
                window = torch.zeros(num_bins, dtype=torch.float32)
                window[b2 : b3 + 1] = 1.0

                # Compute measurand signal
                measurand_series = self.moment_module.forward(window)
                measurand_series = measurand_series - torch.mean(measurand_series)  # Detrend

                # Compute FFT
                measurand_fft = torch.fft.rfft(measurand_series)  # pylint: disable=not-callable
                fetal_fft_component = float(measurand_fft[self.fetal_bins].abs().sum().item())

                # Compute noise floor using MAD
                median_fft = torch.median(measurand_fft.abs()).item()
                mad_fft = torch.median(torch.abs(measurand_fft.abs() - median_fft)).item()
                noise_floor = mad_fft * 1.4826  # Convert MAD to std dev

                if noise_floor == 0:
                    continue

                # Compute SNR
                snr = fetal_fft_component / noise_floor

                # Log training curve data
                results.append([b2, b3, snr])

                if snr >= best_snr:  # Bias towards later windows
                    best_snr = snr
                    best_window = window.clone()
                    self.final_signal = measurand_series.detach().cpu()

        # Store results
        if self.norm is not None:
            self.window = best_window / torch.norm(best_window, p=self.norm)
        else:
            self.window = best_window

        # No training curves for this non-iterative method
        self.training_curves = np.array(results)
