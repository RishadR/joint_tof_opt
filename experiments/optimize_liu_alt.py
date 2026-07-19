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

import logging
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml

from joint_tof_opt import (
    AdditiveGaussianToFModifier,
    CompactStatProcess,
    OptimizationExperiment,
    ToFData,
    generate_tof,
    get_named_moment_module,
)
from sensitivity_compute import AltPaperEvaluator3

logger = logging.getLogger(__name__)

class AltLiuOptimizer(OptimizationExperiment):
    """
    An Alternate Version of the Optimization experiment implementing the optimization loop from Liu et al. that uses
    the Spectral Component Ratio between FHR and MHR to find the optimal window
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
        maternal_f: float | None = None,
        dtof_to_find_max_on: Literal["mean", "median", "first"] = "mean",
        half_width: float = 0.3,
        harmonic_count: int = 2,
        norm: None | float = None,
    ):
        """
        Initialize the AltLiuOptimizer.

        :param tof_data: ToFData instance to optimize on.
        :param measurand: The measurand to optimize for ("abs", "m1", "V") or custom module.
        :param fetal_f: Central frequency of fetal comb filter (in Hz). If None, extracted from dataset metadata.
        :param dtof_to_find_max_on: Which DTOF to use to find bmax and b0 ("mean", "median", or "first").
        :param half_width: Frequency half-width around FHR and MHR for signal extraction (in Hz).
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
        self.maternal_f = maternal_f if maternal_f is not None else self.tof_data.meta_data["maternal_f"]

        # Pre-compute fetal bins for SNR calculation
        num_timepoints = self.tof_data.tof_series.shape[0]
        self.fetal_bins = self._compute_f_bins(self.fetal_f, num_timepoints, self.half_width)
        self.maternal_bins = self._compute_f_bins(self.maternal_f, num_timepoints, self.half_width)

        # Set training curve labels (empty for this non-iterative method)
        self.training_curve_labels = ["Left Bin Index", "Right Bin Index", "Selectivity"]
        self.training_curves = np.array([])

    def _compute_f_bins(self, f: float, num_timepoints: int, hw: float) -> list[int]:
        """
        Compute the FFT bin indices that correspond to FHR harmonics.

        :param num_timepoints: Number of time points in the signal.
        :return: List of bin indices to include in fetal signal extraction.
        """
        fetal_bin = int(f / (self.sampling_rate / num_timepoints))
        fetal_bins = []
        for h in range(1, self.harmonic_count + 1):
            width_int_in_bins = int(hw / (self.sampling_rate / num_timepoints))
            left_edge = max(h * fetal_bin - width_int_in_bins, 0)
            right_edge = min(h * fetal_bin + width_int_in_bins, num_timepoints // 2 + 1)
            fetal_bins.extend(list(range(left_edge, right_edge + 1)))
        return fetal_bins

    def __str__(self) -> str:
        return (
            f"AltLiuOptimizer(measurand={self.moment_module.__class__.__name__}, "
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
            representative_dtof = torch.mean(self.tof_data.tof_series, dim=0)   # type: ignore
        elif self.dtof_to_find_max_on == "median":
            representative_dtof = torch.median(self.tof_data.tof_series, dim=0).values  # type: ignore
        elif self.dtof_to_find_max_on == "first":
            representative_dtof = self.tof_data.tof_series[0, :]
        else:
            raise ValueError(f"Invalid dtof_to_find_max_on value: {self.dtof_to_find_max_on}")

        self.bmax = int(torch.argmax(representative_dtof).item())   # type: ignore

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

        # Step 3: Exhaustive search over all (b2, b3) window pairs (Both inclusive)
        best_selectivity = 0.0
        best_window = torch.zeros(num_bins, dtype=torch.float32)
        best_window[self.b0] = 1.0  # Default to single bin window at b0 if no better window is found

        ##################################
        for b2 in range(self.b0, self.bf):
            for b3 in range(b2 + 1, self.bf):
                # Create rectangular window
                window = torch.zeros(num_bins, dtype=torch.float32) # type: ignore
                window[b2 : b3 + 1] = 1.0

                # Compute measurand signal
                measurand_series = self.moment_module(window)
                measurand_series = measurand_series - torch.mean(measurand_series)  # type: ignore

                # Compute FFT
                measurand_fft = torch.fft.rfft(measurand_series)    # pylint: disable=not-callable
                fetal_fft_component = float(measurand_fft[self.fetal_bins].abs().sum().item())
                maternal_fft_component = float(measurand_fft[self.maternal_bins].abs().sum().item())

                # Compute noise floor using MAD
                median_fft = torch.median(measurand_fft.abs()).item()   # type: ignore
                mad_fft = torch.median(torch.abs(measurand_fft.abs() - median_fft)).item()  # type: ignore
                noise_floor = mad_fft * 1.4826  # Convert MAD to std dev

                if noise_floor == 0:
                    continue

                # Compute Selectivity (SNR)
                selectivity = (fetal_fft_component - noise_floor) / (maternal_fft_component - noise_floor)

                # Log training curve
                results.append([b2, b3, selectivity])

                if selectivity >= best_selectivity:  # Bias towards later windows
                    best_selectivity = selectivity
                    best_window = window.clone()
                    self.final_signal = measurand_series.detach().cpu()

        # Store results
        if best_window is not None:
            if self.norm is not None:
                self.window = best_window / torch.norm(best_window, p=self.norm)    # type: ignore
            else:
                self.window = best_window
        else:
            raise ValueError("Something went wrong; no valid window that improves SNR above 0.")

        # No training curves for this non-iterative method
        self.training_curves = np.array(results)


def plot_training_curves_and_window(
    training_curves: np.ndarray,
    curve_column_labels: list[str],
    optimized_window: torch.Tensor,
    bin_edges: np.ndarray,
    fig_size: tuple[int, int] = (10, 6),
    normalize_curves: bool = False,
    filename: str = "liu_alt_optimization_result",
) -> None:
    """
    Plot the training curves and the optimized window.

    :param training_curves: Numpy array of training curves.
    :param curve_column_labels: Labels for each column in training_curves.
    :param optimizer_window: The optimized window tensor.
    :param bin_edges: The edges of the ToF bins for plotting the window.
    :param fig_size: Size of the figure.
    :param normalize_curves: Whether to normalize training curves for plotting.
    :param filename: Base file name to save the plots.
    """
    ## Validity Checks
    assert training_curves.shape[1] == len(curve_column_labels), "Mismatch between training curves and labels"

    # Bin Centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_centers_ns = bin_centers * 1e9  # Convert to ns for plotting
    bin_centers_ns = np.round(bin_centers_ns, 2)

    ## Load config for plotting if available
    config_path = Path("./plotting_codes/plot_config.yaml")
    if config_path.exists():
        with open(config_path) as f:
            plot_config = yaml.safe_load(f)
            plt.rcParams.update(plot_config)

    plt.subplots(1, 2, figsize=fig_size)

    # Plot Training Curves
    win_length = len(optimized_window)
    selectivity_grid = np.zeros((win_length, win_length))
    for b2, b3, selectivity in training_curves:
        selectivity_grid[int(b2), int(b3)] = selectivity
    selectivity_max = np.max(selectivity_grid)
    if normalize_curves:
        selectivity_grid /= selectivity_max

    plt.subplot(1, 2, 1)
    plt.imshow(
        selectivity_grid.T, origin="lower", cmap="viridis", aspect="auto"
    )
    plt.colorbar(label="Selectivity (Normalized)" if normalize_curves else "Selectivity")
    plt.xlabel("Left Bin Index (b2)")
    plt.ylabel("Right Bin Index (b3)")
    plt.title("Selectivity Heatmap")

    # Plot Optimized Window
    plt.subplot(1, 2, 2)
    plt.plot(bin_centers_ns, optimized_window.detach().cpu().numpy(), marker="o")
    plt.xlabel("Bin Center (ns)")
    plt.ylabel("Window Value")
    plt.title("Optimized Window")
    plt.tight_layout()

    plt.savefig(f"./figures/{filename}.svg")
    plt.savefig(f"./figures/{filename}.pdf")


def main(noise_var: float = 100.0) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    # file_idx = 7
    for file_idx in range(2, 3):
        measurand = "abs"
        ppath_file = Path(f"./data/experiment_{file_idx:04d}.npz")
        logger.info("Running optimization loop for file: %04d.npz | Measurand: %s", file_idx, measurand)
        tof_dataset_path = Path("./data") / f"generated_tof_set_{ppath_file.stem}.npz"
        gen_config: dict = yaml.safe_load(open("./experiments/tof_config.yaml"))
        filter_hw = 0.01
        generate_tof(ppath_file, gen_config, tof_dataset_path, True, True)
        tof_data = ToFData.from_npz(tof_dataset_path)
        modifier = AdditiveGaussianToFModifier(noise_var)
        tof_data = modifier.modify(tof_data)
        experiment = AltLiuOptimizer(
            tof_data=tof_data,
            measurand=measurand,
            dtof_to_find_max_on='mean',
            half_width = 0.1,
            harmonic_count=2,
            norm=None
        )
        experiment.optimize()
        optimized_window = experiment.window  # type: ignore
        logger.info("Optimized Window: %s", optimized_window.numpy())

        # Evaluate using an Evaluator and print log
        evaluator = AltPaperEvaluator3(ppath_file, optimized_window, measurand, gen_config, filter_hw)
        eval_results = evaluator.evaluate()
        logger.info("Evaluation Results: %s", eval_results)
        logger.info("Evaluator log: %s", evaluator.get_log())

        # Clean up
        tof_dataset_path.unlink()  # Remove the generated ToF dataset to save space


if __name__ == "__main__":
    main()
