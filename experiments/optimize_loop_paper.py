"""
Implementation of the optimization loop used for the paper.

This has an main_optimize function that loads a data, applies the optimization and outputs the optmized window as
well as the training curves.

Process Flow:
1. Load DTOF dataset (Each row is a histogram/DTOF, each column is a timebin), the dataset also contains other info
2. Extract all the info from the dataset including fetal and maternal frequencies
3. Initialize the Window vector as a learnable parameter
4. Optimization Loop Starts: Compute the compact statistics using the current window
5. Apply Sinc Comb Filter to extract fetal and maternal signals using known frequencies
6. Compute the Energy Ratio Metric between filtered fetal and filtered maternal signals (Fetal Selectivity)
7. Compute the Contrast-to-Noise Metric for the fetal signal (Using analytical noise equations)
8. Final Metric is the product of Energy Ratio and Contrast-to-Noise
9. Optimize the window parameters to maximize the final metric untill convergence - Optimization Loop Ends
10. Output the optimized window and training curves - the curves contain each of the three metrics at each epoch

Early Stopping Logic:
If the final metric does not improve by at least 1% over the best recorded metric in 'patience' epochs,
stop the optimization early.

Window Parameterization:
-The window is parameterized using exponentiation of unconstrained parameters to ensure positivity.
-The window is normalized to have unit energy at each epoch such that the optimizer does not trivially increase
the window energy.
"""

import logging
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.optim as optim

from joint_tof_opt import (
    AdditiveGaussianToFModifier,
    CombSeparator,
    CompactStatProcess,
    ContrastToNoiseMetric,
    EnergyRatioMetric,
    FourierSeparator,
    NoiseCalculator,
    OptimizationExperiment,
    PSAFESeparator,
    ToFData,
    WindowSumNoiseCalculator,
    WindowSumWithAdditiveGaussianNoiseCalculator,
    generate_tof,
    get_named_moment_module,
    load_tof_config,
    named_moment_types,
)
from joint_tof_opt.plotting import load_plot_config

from .sensitivity_compute import (
    AltPaperEvaluator3,
)

logger = logging.getLogger(__name__)

RegType = Literal["l1", "l2"]
FilterType = Literal["comb", "fourier", "psafe_same_width", "psafe_true_width", "comb_psafe_hybrid"]
NormalizationScheme = Literal["unit_sum", "unit_max"]


class DIGSSOptimizer(OptimizationExperiment):
    """
    Optimization experiment implementing the optimization loop used in the paper.

    This class optimizes a window function to maximize a combination of:
    - Energy Ratio Metric (fetal selectivity)
    - Contrast-to-Noise Metric (signal quality)

    Process Flow:
    1. Load DTOF dataset
    2. Extract metadata (frequencies, sampling rate, etc.)
    3. Initialize window vector as learnable parameter
    4. For each epoch:
       - Compute compact statistics using current window
       - Apply sinc comb filters to extract fetal/maternal signals
       - Compute energy ratio and contrast-to-noise metrics
       - Optimize window to maximize final metric
    5. Output optimized window and training curves

    Early stopping occurs if final metric doesn't improve by 1% over best recorded metric
    for 'patience' consecutive epochs.
    """

    def __init__(
        self,
        tof_data: ToFData,
        measurand: str | CompactStatProcess,
        noise_calc: NoiseCalculator | None = None,
        fetal_f: float | None = None,
        max_epochs: int = 2000,
        lr: float = 0.1,
        filter_hw: float = 0.01,
        patience: int = 50,
        grad_clip: bool = False,
        reg_type: RegType = "l1",
        reg_weight: float = 1e-4,
        window_smoothening: bool = True,
        normalize_reward: bool = True,
        filter_type: FilterType = "psafe_same_width",
        normalization_scheme: NormalizationScheme = "unit_sum",
        use_window_post_process: bool = True,
        use_snr_left_bound: bool = True,
    ):
        """
        Initialize the PaperOptimizer.

        :param tof_data: ToFData instance to optimize on.
        :param measurand: The measurand to optimize for ("abs", "m1", "V") or custom module.
        :param noise_calc: Noise calculator for custom measurands - defaults to WindowSumNoiseCalculator
        :param fetal_f: Central frequency of fetal comb filter (in Hz). If None, extracted from dataset metadata.
        :param max_epochs: Maximum number of optimization epochs.
        :param fetal_f: Central frequency of fetal comb filter (in Hz). If None, extracted from dataset metadata.
        :param lr: Learning rate for the optimizer.
        :param filter_hw: Half width of the sinc comb filter (in Hz).
        :param patience: Number of epochs to wait for improvement before early stopping.
        :param grad_clip: Whether to apply gradient clipping.
        :param reg_type: Regularization type ("l1" or "l2").
        :param reg_weight: Regularization weight (must be non-negative).
        :param window_smoothening: If true - sets all window weights below 1% of the max weight to 0
        :param normalize_reward: Whether to normalize the final reward/metric for better optimization stability
        :param filter_type: Type of filter to use
         - "comb": Standard sinc comb filter
         - "fourier": Uses the RFFT to manually zero out irrelevant frequencies
         - "psafe_same_width": Uses the PSAFE filter with the outputs being the same length as the input
         - "psafe_true_width": Uses the PSAFE filter with the outputs being the length of the true fetal/maternal period
         - "comb_psafe_hybrid": Uses the comb for maternal/psafe for fetal to account for FHR being a multiple of MHR
        :param normalization_scheme: How is the window normalized? All weights sum to 1 ("unit_sum") or the max
            per bin equals 1 ("unit_max")? Default is "unit_sum".
        """
        # Handle measurand and noise function
        if isinstance(measurand, str):
            if measurand not in named_moment_types:
                raise ValueError(f"Invalid measurand string: {measurand}. Must be one of {named_moment_types}.")
        self.noise_calc: NoiseCalculator = noise_calc if noise_calc is not None else WindowSumNoiseCalculator()

        if isinstance(measurand, str):
            measurand = get_named_moment_module(measurand, tof_data)

        super().__init__(tof_data, measurand, lr)

        self.max_epochs = max_epochs
        self.filter_hw = filter_hw
        self.patience = patience
        self.grad_clip = grad_clip
        self.reg_type = reg_type
        self.reg_weight = reg_weight
        self.filter_type = filter_type
        self.window_smoothening = window_smoothening
        self.normalization_scheme = normalization_scheme
        self.use_window_post_process = use_window_post_process
        self.use_snr_left_bound = use_snr_left_bound
        self.impulse_window_snr_list = [0.0] * self.tof_data.tof_series.shape[1]
        self.impulse_window_selectivity_list = [0.0] * self.tof_data.tof_series.shape[1]
        self.impulse_window_product_list = [0.0] * self.tof_data.tof_series.shape[1]
        self.unprocessed_window = torch.ones(self.tof_data.tof_series.shape[1])  # For logging purposes

        if self.reg_type not in ("l1", "l2"):
            raise ValueError(f"Unsupported reg_type: {self.reg_type}. Use 'l1' or 'l2'.")
        if self.reg_weight < 0:
            raise ValueError("reg_weight must be non-negative.")

        # Extract additional metadata
        assert self.tof_data.meta_data is not None, "ToFData meta_data cannot be None"
        self.sampling_rate: float = self.tof_data.meta_data["sampling_rate"]
        self.fetal_f: float = fetal_f if fetal_f is not None else self.tof_data.meta_data["fetal_f"]
        self.maternal_f: float = self.tof_data.meta_data["maternal_f"]

        self.fetal_filter, self.maternal_filter = self._get_filters(self.filter_type)

        self.normalize_reward = normalize_reward
        self.training_curves: npt.NDArray[np.float64] = np.zeros((self.max_epochs, 3), dtype=np.float64)
        self.training_curves_extra: npt.NDArray[np.float64] = np.zeros((self.max_epochs, 10), dtype=np.float64)

        # Compute max single bin values
        self.max_snr, self.max_selectivity, self.max_snr_index, self.max_selectivity_index = self._compute_max_values()
        # Learnable parameter initialized with uniform weights
        num_bins = self.tof_data.tof_series.shape[1]
        time_points = self.tof_data.tof_series.shape[0]

        # max_snr_index must always fall inside learnable_component, so fixed_left stops right before it.
        self.left_bound_length = self.max_snr_index if self.use_snr_left_bound else 0

        # Computing the right bound - power should always be greater than variance
        mean_frame = self.tof_data.tof_series.mean(dim=0)  # Shape: (num_bins,)
        signal_power = mean_frame**2  # Shape: (num_bins,)
        unity_window = torch.ones(num_bins, device=self.tof_data.bin_edges.device)
        total_noise_variance = self.noise_calc.compute_noise(self.tof_data, unity_window, sum_axis=0) #Shape:(num_bins,)
        mean_noise_variance = total_noise_variance / time_points 
        viable_bins = torch.where(signal_power >= mean_noise_variance)[0]
        right_most_bin = int(viable_bins[-1].item()) if viable_bins.numel() > 0 else -1
        assert right_most_bin >= self.left_bound_length, (
            "No viable bins at/after max_snr_index - every trailing bin's signal power is below its noise variance."
        )

        learnable_len = (right_most_bin + 1) - self.left_bound_length
        # initialize uniform weights
        self.learnable_component_exponents = nn.Parameter(torch.ones(learnable_len) * 0.0)
        self.learnable_component = self._winexp_to_win_func(self.learnable_component_exponents)
        self.fixed_left = (
            torch.ones(self.left_bound_length, device=self.tof_data.bin_edges.device) * 1e-4
            if self.left_bound_length > 0
            else torch.tensor([], device=self.tof_data.bin_edges.device)
        )
        self.fixed_right = (
            torch.ones(num_bins - right_most_bin - 1, device=self.tof_data.bin_edges.device) * 1e-4
            if right_most_bin < num_bins - 1
            else torch.tensor([], device=self.tof_data.bin_edges.device)
        )

        unnormalized_window = torch.cat([self.fixed_left, self.learnable_component, self.fixed_right])

        self.window = self._win_norm_func(unnormalized_window, self.normalization_scheme)
        self.window_norm = self.window.clone()

        # Labels for the 3 metrics recorded in training_curves
        self.training_curve_labels = ["Energy Ratio", "Contrast-to-Noise", "Final Metric"]
        self.training_curve_extra_labels = [
            "Fetal Filtered Energy",
            "Maternal Filtered Energy",
            "Fetal AC Energy",
            "Noise STD",
            "Fetal AC Amp",
            "Maternal AC Amp",
            "SNR Metric 1",
            "Loss",
            "Gradient Norm",
            "Reg Loss",
        ]

    def _get_filters(self, filter_type: FilterType) -> tuple[nn.Module, nn.Module]:
        datapoint_count = int(self.tof_data.tof_series.shape[0])
        sampling_rate = self.sampling_rate
        if filter_type == "comb":
            fetal_filter = CombSeparator(
                fs=sampling_rate,
                f0=self.fetal_f,
                f1=2 * self.fetal_f,
                half_width=self.filter_hw,
                filter_length=datapoint_count // 2 + 1,
            )
            maternal_filter = CombSeparator(
                fs=sampling_rate,
                f0=self.maternal_f,
                f1=2 * self.maternal_f,
                half_width=self.filter_hw,
                filter_length=datapoint_count // 2 + 1,
            )
        elif filter_type == "fourier":
            fetal_filter = FourierSeparator(
                fs=sampling_rate,
                f0=self.fetal_f,
                f1=2 * self.fetal_f,
                half_width=self.filter_hw,
            )
            maternal_filter = FourierSeparator(
                fs=sampling_rate,
                f0=self.maternal_f,
                f1=2 * self.maternal_f,
                half_width=self.filter_hw,
            )
        elif filter_type == "psafe_same_width":
            fetal_filter = PSAFESeparator(fs=sampling_rate, center_freq=self.fetal_f, equate_length=True)
            maternal_filter = PSAFESeparator(fs=sampling_rate, center_freq=self.maternal_f, equate_length=True)
        elif filter_type == "psafe_true_width":
            fetal_filter = PSAFESeparator(fs=sampling_rate, center_freq=self.fetal_f, equate_length=False)
            maternal_filter = PSAFESeparator(fs=sampling_rate, center_freq=self.maternal_f, equate_length=False)
        elif filter_type == "comb_psafe_hybrid":
            fetal_filter = PSAFESeparator(fs=sampling_rate, center_freq=self.fetal_f, equate_length=True)
            maternal_filter = CombSeparator(
                fs=sampling_rate,
                f0=self.maternal_f,
                f1=2 * self.maternal_f,
                half_width=self.filter_hw,
                filter_length=datapoint_count // 2 + 1,
            )
        else:
            raise ValueError(f"Unknown filter_type: {filter_type}")
        return fetal_filter, maternal_filter

    def _win_norm_func(self, window: torch.Tensor, scheme: str) -> torch.Tensor:
        if scheme == "unit_sum":
            return window / torch.norm(window, p=1)
        elif scheme == "unit_max":
            return window / torch.max(window)
        else:
            raise ValueError(f"Unknown normalization scheme: {scheme}")

    def _winexp_to_win_func(self, window_exp: torch.Tensor) -> torch.Tensor:
        return torch.exp(window_exp)

    def _compute_max_values(self) -> tuple[float, float, int, int]:
        """
        Compute maximum SNR and Selectivity for single-bin windows.

        This method computes single-bin metrics for normalization purposes.

        :return: (max_snr, max_selectivity, max_snr_index, max_selectivity_index)
        """
        num_bins = self.tof_data.tof_series.shape[1]
        snr_metric_list = []
        selectivity_metric_list = []
        product_metric_list = []

        snr_calc = ContrastToNoiseMetric(noise_calc=self.noise_calc, tof_data=self.tof_data)
        selectivity_calc = EnergyRatioMetric()

        for i in range(num_bins):
            # Single-bin window (one-hot vector)
            single_bin_window = torch.zeros(num_bins)
            single_bin_window[i] = 1.0

            # Compute compact statistics
            compact_stats = self.moment_module(single_bin_window)
            compact_stats = compact_stats - compact_stats.mean()
            compact_stats_reshaped = compact_stats.unsqueeze(0).unsqueeze(0)
            maternal_filtered_signal = self.maternal_filter(compact_stats_reshaped)
            fetal_filtered_signal = self.fetal_filter(compact_stats_reshaped)
            maternal_energy = torch.sum(maternal_filtered_signal**2)
            fetal_energy = torch.sum(fetal_filtered_signal**2)

            # Compute selectivity
            selectivity = selectivity_calc(fetal_energy, maternal_energy)
            selectivity_metric_list.append(selectivity.item())

            # Compute SNR using noise calculator
            snr = snr_calc(single_bin_window, fetal_filtered_signal)
            snr_metric_list.append(snr.item())

            # Store for logging
            product_metric_list.append(snr.item() * selectivity.item())

        max_snr = max(snr_metric_list)
        max_selectivity = max(selectivity_metric_list)
        max_snr_index = snr_metric_list.index(max_snr)
        max_selectivity_index = selectivity_metric_list.index(max_selectivity)

        # Store impulse response metrics for logging
        self.impulse_window_snr_list = snr_metric_list
        self.impulse_window_selectivity_list = selectivity_metric_list
        self.impulse_window_product_list = product_metric_list

        return max_snr, max_selectivity, max_snr_index, max_selectivity_index

    def smoothen_window(self, window: torch.Tensor) -> torch.Tensor:
        """
        Apply thresholding: set weights < 1% of max to zero.

        :param window: Window tensor to smoothen.
        :type window: torch.Tensor
        :return: Smoothened window tensor.
        :rtype: torch.Tensor
        """
        threshold = 0.01 * torch.max(window)
        return torch.where(window < threshold, torch.zeros_like(window), window)

    def optimize(self) -> None:
        """
        Execute the optimization process to find the optimal window function.

        This method initializes parameters, sets up the optimizer, and runs the training loop with early stopping.
        """
        optimizer = optim.Adam([self.learnable_component_exponents], lr=self.lr)

        best_metric = -float("inf")
        patience_counter = 0

        # Loss history tracking - 3 metrics per epoch
        loss_history = []
        loss_history_extra = []

        snr_calc = ContrastToNoiseMetric(noise_calc=self.noise_calc, tof_data=self.tof_data)
        selectivity_calc = EnergyRatioMetric()

        for _ in range(self.max_epochs):
            optimizer.zero_grad()

            # Window parameterization with non-negativity constraint
            self.learnable_component = self._winexp_to_win_func(self.learnable_component_exponents)
            unnormalized_window = torch.cat([self.fixed_left, self.learnable_component, self.fixed_right])
            self.window = self._win_norm_func(unnormalized_window, self.normalization_scheme)

            # Extract compact statistics and apply comb filtering
            compact_stats = self.moment_module(self.window)

            # Center the signal
            compact_stats = compact_stats - compact_stats.mean()
            compact_stats_reshaped = compact_stats.unsqueeze(0).unsqueeze(0)

            maternal_filtered_signal = self.maternal_filter(compact_stats_reshaped)
            fetal_filtered_signal = self.fetal_filter(compact_stats_reshaped)
            self.final_signal = fetal_filtered_signal.squeeze().detach().cpu()

            # Compute energies
            maternal_energy = torch.sum(maternal_filtered_signal**2)
            fetal_energy = torch.sum(fetal_filtered_signal**2)

            # Compute metrics
            energy_ratio = selectivity_calc(fetal_energy, maternal_energy)
            contrast_to_noise = snr_calc(self.window, fetal_filtered_signal)

            # Normalize rewards if requested
            if self.normalize_reward:
                energy_ratio_norm = energy_ratio / self.max_selectivity
                contrast_to_noise_norm = contrast_to_noise / self.max_snr
                final_metric = energy_ratio_norm * contrast_to_noise_norm
            else:
                final_metric = energy_ratio * contrast_to_noise

            # Add regularization
            if self.reg_type == "l1":
                reg_loss = self.reg_weight * torch.sum(torch.abs(self.window))
            elif self.reg_type == "l2":
                reg_loss = self.reg_weight * torch.sum(self.window**2)
            else:
                reg_loss = torch.tensor(0.0)

            loss = -final_metric + reg_loss

            # Record loss history - 3 metrics: Selectivity, SNR, Final Metric
            loss_history.append([energy_ratio.item(), contrast_to_noise.item(), final_metric.item()])
            loss_history_extra.append(
                [
                    fetal_energy.item(),
                    maternal_energy.item(),
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    loss.item(),
                    0.0,
                    reg_loss.item(),
                ]
            )

            # Optimization step
            loss.backward()

            # Optional gradient clipping
            if self.grad_clip:
                nn.utils.clip_grad_norm_([self.learnable_component_exponents], max_norm=1.0)

            optimizer.step()

            # Early stopping check: 1% improvement threshold
            current_metric = final_metric.item()
            if current_metric > best_metric * 1.01:
                best_metric = current_metric
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        self.training_curves = np.array(loss_history, dtype=np.float64)
        self.training_curves_extra = np.array(loss_history_extra, dtype=np.float64)

        # Apply post-processing smoothening if requested
        if self.window_smoothening:
            self.window = self.smoothen_window(self.window)

        # Detach learnable parameter from computation graph
        self.learnable_component = self.learnable_component.detach()

        # Update final window representation
        self.window = self.window.detach()
        self.unprocessed_window = self.window.clone()

        if self.use_window_post_process:
            self.window = self.window_post_process(self.window)

    def window_post_process(self, window: torch.Tensor) -> torch.Tensor:
        """
        Post-processes the window: sets all elements from the left up to and including
        the maximum element to 1.0, and leaves the remaining elements untouched.

        Parameters:
            window (torch.Tensor): 1D tensor representing the window function.

        Returns:
            torch.Tensor: The post-processed window.
        """
        # Find the index of the first occurrence of the maximum value
        max_idx = torch.argmax(window).item()

        # Clone the window to avoid modifying in-place if necessary
        post_processed_window = window.clone()

        # Set all elements up to and including the max_idx to 1.0
        post_processed_window[: max_idx + 1] = 1.0

        return post_processed_window

    def __str__(self) -> str:
        return (
            f"DIGSSOptimizer(normalization_scheme={self.normalization_scheme}, "
            f"use_window_post_process={self.use_window_post_process}, "
            f"use_snr_left_bound={self.use_snr_left_bound})"
        )

    def components(self) -> dict[str, nn.Module]:
        return {
            "fetal_filter": self.fetal_filter,
            "maternal_filter": self.maternal_filter,
            "measurand": self.moment_module,
        }


def plot_training_curves_and_window(
    training_curves: npt.NDArray[np.float64],
    window: torch.Tensor,
    bin_edges: npt.NDArray[np.float64],
    training_curve_labels: list[str],
    save_path: str = "./results/optimization_results",
) -> None:
    """
    Plot the training curves and the optimized window function.

    :param training_curves: 2D numpy array of metric values (epochs x metrics).
    :type training_curves: npt.NDArray[np.float64]
    :param window: 1D torch tensor of optimized window weights.
    :type window: torch.Tensor
    :param bin_edges: 1D numpy array of bin edges for plotting the window.
    :type bin_edges: npt.NDArray[np.float64]
    :param training_curve_labels: List of labels for each metric in training_curves.
    :type training_curve_labels: list[str]
    :param save_path: Path prefix for saving plot images (without extension).
    :type save_path: str
    """
    # Load standardized plot configuration
    plot_config = load_plot_config()

    plt.figure(
        figsize=(
            plot_config.figure_sizes.double_column[0],
            plot_config.figure_sizes.double_column[1],
        )
    )

    # Plot 1: Training Curves
    plt.subplots(figsize=(6, 4))
    epochs = range(training_curves.shape[0])

    plt.subplot(1, 2, 1)
    for i in range(training_curves.shape[1]):
        label = training_curve_labels[i] if i < len(training_curve_labels) else f"Metric {i + 1}"
        curve = training_curves[:, i]
        # Normalize each curve to start at 1
        curve_normalized = curve / curve[0] if curve[0] != 0 else curve
        plt.plot(epochs, curve_normalized, label=label)
    plt.xlabel("Epoch", fontsize=plot_config.fonts.label_size)
    plt.ylabel("Normalized Metric Value", fontsize=plot_config.fonts.label_size)
    plt.yscale("log")  # Logarithmic scale for better visualization of improvement
    # plt.ylim(bottom=1e-1, top=1e2)
    plt.title("Training Curves", fontsize=plot_config.fonts.title_size)
    plt.legend(fontsize=plot_config.fonts.legend_size)
    plt.grid(True, linestyle=plot_config.grid.style, alpha=plot_config.grid.alpha)

    # Plot 2: Optimized Window
    plt.subplot(1, 2, 2)
    plt.plot(bin_edges, window.numpy(), label="Optimized Window", color="orange")
    plt.xlabel("ToF Bins (ps)", fontsize=plot_config.fonts.label_size)
    plt.ylabel("Window Weight", fontsize=plot_config.fonts.label_size)
    plt.title("Optimized Window", fontsize=plot_config.fonts.title_size)
    # plt.grid(True, linestyle=plot_config.grid.style, alpha=plot_config.grid.alpha)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(f"{save_path}.png", dpi=plot_config.figure_export.dpi, bbox_inches="tight")
    plt.savefig(f"{save_path}.pdf", bbox_inches="tight")
    plt.close()


def main() -> None:
    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Load configuration
    tof_config = load_tof_config(Path("./experiments/tof_config.yaml"))

    # Load dataset
    tof_data = generate_tof(Path("./data/experiment_0003.npz"), tof_config, True, True)

    noise_var = 100.0
    tof_modifier = AdditiveGaussianToFModifier(noise_var=noise_var)
    noisy_tof = tof_modifier.modify(tof_data)

    noise_calc = WindowSumWithAdditiveGaussianNoiseCalculator(noise_var)

    measurand = "abs"
    # Create optimizer experiment instance
    experiment = DIGSSOptimizer(
        noisy_tof,
        measurand,
        max_epochs=2000,
        lr=0.1,
        filter_hw=0.01,
        patience=100,
        grad_clip=False,
        reg_type="l1",
        reg_weight=0.0,
        window_smoothening=False,
        normalize_reward=True,
        filter_type="psafe_same_width",
        normalization_scheme="unit_max",
        noise_calc=noise_calc,
    )

    # Run optimization
    experiment.optimize()

    # Log results
    logger.info("Optimization complete!")
    logger.info("Final window shape: %s", experiment.window.shape)
    logger.info("Final window weights: %s", experiment.window.numpy())
    logger.info("Final Energy Ratio: %s", experiment.training_curves[-1, 0])
    logger.info("Final SNR: %s", experiment.training_curves[-1, 1])
    logger.info("Final Metric: %s", experiment.training_curves[-1, 2])

    # Extract bin edges for plotting
    assert experiment.tof_data.meta_data is not None, "ToFData meta_data cannot be None"
    bin_edges = experiment.tof_data.bin_edges.numpy()

    plot_training_curves_and_window(
        experiment.training_curves, experiment.window, bin_edges, experiment.training_curve_labels
    )
    evaluator = AltPaperEvaluator3(
        Path("./data/experiment_0003.npz"),
        experiment.window,
        measurand,
        tof_config,
        filter_hw=0.01,
        gaussian_noise_var=noise_var,
    )
    logger.info("Evaluator log:")
    logger.info(evaluator.evaluate())
    logger.info("Unprocessed Window:")
    logger.info(experiment.unprocessed_window.numpy())


if __name__ == "__main__":
    main()
