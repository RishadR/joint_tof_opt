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
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from joint_tof_opt import (
    AdditiveGaussianToFModifier,
    CombSeparator,
    CompactStatProcess,
    FourierSeparator,
    NoiseCalculator,
    OptimizationExperiment,
    PSAFESeparator,
    ToFData,
    WindowSumNoiseCalculator,
    WindowSumWithAdditiveGaussianNoiseCalculator,
    generate_tof,
    get_named_moment_module,
    named_moment_types,
)
from joint_tof_opt.plotting import load_plot_config
from sensitivity_compute import (
    AltPaperEvaluator3,
)

logger = logging.getLogger(__name__)


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
        tof_dataset_path: Path,
        measurand: str | CompactStatProcess,
        noise_calc: NoiseCalculator | None = None,
        fetal_f: float | None = None,
        max_epochs: int = 2000,
        lr: float = 0.1,
        filter_hw: float = 0.01,
        patience: int = 50,
        grad_clip: bool = False,
        reg_type: Literal["l1", "l2"] = "l1",
        reg_weight: float = 1e-4,
        window_smoothening: bool = True,
        normalize_reward: bool = True,
        filter_type: Literal[
            "comb", "fourier", "psafe_same_width", "psafe_true_width", "comb_psafe_hybrid"
        ] = "psafe_same_width",
        normalization_scheme: Literal["unit_sum", "unit_max"] = "unit_sum",
    ):
        """
        Initialize the PaperOptimizer.

        :param tof_dataset_path: Path to the ToF dataset (.npz file).
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
        else:
            self.noise_calc = noise_calc
        self.noise_calc = noise_calc if noise_calc is not None else WindowSumNoiseCalculator()

        if isinstance(measurand, str):
            tof_data = ToFData.from_npz(tof_dataset_path)
            measurand = get_named_moment_module(measurand, tof_data)

        super().__init__(tof_dataset_path, measurand, lr)

        self.max_epochs = max_epochs
        self.filter_hw = filter_hw
        self.patience = patience
        self.grad_clip = grad_clip
        self.reg_type = reg_type
        self.reg_weight = reg_weight
        self.filter_type = filter_type
        self.window_smoothening = window_smoothening
        self.normalization_scheme = normalization_scheme
        self.impulse_window_snr_list = [0.0] * len(self.tof_data.bin_edges)
        self.impulse_window_selectivity_list = [0.0] * len(self.tof_data.bin_edges)
        self.impulse_window_product_list = [0.0] * len(self.tof_data.bin_edges)
        self.unprocessed_window = torch.ones(len(self.tof_data.bin_edges))  # For logging purposes

        if self.reg_type not in ("l1", "l2"):
            raise ValueError(f"Unsupported reg_type: {self.reg_type}. Use 'l1' or 'l2'.")
        if self.reg_weight < 0:
            raise ValueError("reg_weight must be non-negative.")

        # Extract additional metadata
        assert self.tof_data.meta_data is not None, "ToFData meta_data cannot be None"
        self.sampling_rate = self.tof_data.meta_data["sampling_rate"]
        self.fetal_f = fetal_f if fetal_f is not None else self.tof_data.meta_data["fetal_f"]
        self.maternal_f = self.tof_data.meta_data["maternal_f"]
        _, num_bins = self.tof_data.tof_series.shape
        self.fetal_filter, self.maternal_filter = self._get_filters(filter_type)
        self.training_curves: np.ndarray = np.zeros((self.max_epochs, 3))
        self.normalize_reward = normalize_reward
        self.training_cruves_extra = np.zeros((self.max_epochs, 10))  # Logging the independent 3 elements

        ## Compute Best Case Windows
        self.max_snr, self.max_selectivity, self.max_snr_index, self.max_selectivity_index = self._compute_max_values()

        # Initalize the learnable window parameters - all else is fixed to 0.0
        initial_params = torch.zeros(self.max_selectivity_index - self.max_snr_index + 1, dtype=torch.float32)
        # initial_params[-2: 0] = 10.0  # Initialize the last 3 learnable parameters to 1 (before exponentiation)

        self.learnable_component_exponents = torch.nn.Parameter(initial_params, requires_grad=True)
        self.learnable_component = self._winexp_to_win_func(self.learnable_component_exponents)
        self.fixed_left = torch.zeros(
            self.max_snr_index - 1,
            dtype=self.learnable_component_exponents.dtype,
            device=self.learnable_component_exponents.device,
        )
        self.fixed_right = torch.zeros(
            num_bins - self.max_selectivity_index,
            dtype=self.learnable_component_exponents.dtype,
            device=self.learnable_component_exponents.device,
        )
        self.window = torch.cat([self.fixed_left, self.learnable_component, self.fixed_right], dim=0)
        self.window_norm = self._win_norm_func(self.window)

        # Set training curve labels
        self.training_curve_labels = ["Normalized Selectivity", "Normalized SNR", "Final Metric"]
        self.training_curve_extra_labels = ["Fetal Energy", "Maternal Energy", "Baseline Noise STD"]

    def _get_filters(self, filter_type: str):
        filter_len = int(2 * self.sampling_rate / self.fetal_f)  # Ensure at least 2 periods are captured
        if filter_type == "comb":
            f1 = CombSeparator(self.sampling_rate, self.fetal_f, self.fetal_f * 2, self.filter_hw, filter_len, True)
            f2 = CombSeparator(
                self.sampling_rate,
                self.maternal_f,
                self.maternal_f * 2,
                self.filter_hw,
                filter_len,
                True,
            )
        elif filter_type == "fourier":
            f1 = FourierSeparator(self.sampling_rate, self.fetal_f, self.fetal_f * 2, self.filter_hw)
            f2 = FourierSeparator(self.sampling_rate, self.maternal_f, self.maternal_f * 2, self.filter_hw)
        elif filter_type == "psafe_same_width":
            f1 = PSAFESeparator(self.sampling_rate, self.fetal_f, True)
            f2 = PSAFESeparator(self.sampling_rate, self.maternal_f, True)
        elif filter_type == "psafe_true_width":
            f1 = PSAFESeparator(self.sampling_rate, self.fetal_f, False)
            f2 = PSAFESeparator(self.sampling_rate, self.maternal_f, False)
        elif filter_type == "comb_psafe_hybrid":
            f1 = PSAFESeparator(self.sampling_rate, self.fetal_f, True)
            f2 = CombSeparator(
                self.sampling_rate,
                self.maternal_f,
                self.maternal_f * 2,
                self.filter_hw,
                filter_len,
                True,
            )
        else:
            raise NotImplementedError(f"Filter type {filter_type} not recognized")
        return f1, f2

    def _win_norm_func(self, win: torch.Tensor) -> torch.Tensor:
        if self.normalization_scheme == "unit_sum":
            return win / torch.norm(win, p=1)
        elif self.normalization_scheme == "unit_max":
            return torch.clamp(win, max=1.0)
        else:
            raise NotImplementedError(f"Normalization scheme {self.normalization_scheme} not recognized")

    @staticmethod
    def _winexp_to_win_func(win_exp: torch.Tensor) -> torch.Tensor:
        return torch.exp(win_exp)

    def _compute_max_values(self):
        """
        Compute the maximum possible SNR and Selectivity by brute force turning each bin "on" individually.
        Returns:
            max_snr: The maximum SNR achieved by any single-bin window
            max_selectivity: The maximum selectivity achieved by any single-bin window
            max_snr_index: The index of the bin that achieves the maximum SNR
            max_selectivity_index: The index of the bin that achieves the maximum selectivity
        """
        _, num_bins = self.tof_data.tof_series.shape
        best_snr = 0.0
        best_selectivity = 0.0
        best_product = 0.0
        best_product_index = -1
        best_snr_index = -1
        best_selectivity_index = -1
        for i in range(num_bins):
            window = torch.zeros(num_bins)
            window[i] = 1.0
            noise_var = torch.mean(self.noise_calc.compute_noise(self.tof_data, window))
            noise_std = torch.sqrt(noise_var)
            compact_stats = self.moment_module(window)
            compact_stats = compact_stats - compact_stats.mean()
            compact_stats_reshaped = compact_stats.unsqueeze(0).unsqueeze(0)
            maternal_filtered_signal = self.maternal_filter(compact_stats_reshaped)
            fetal_filtered_signal = self.fetal_filter(compact_stats_reshaped)
            fetal_energy = torch.sum(fetal_filtered_signal**2)
            maternal_energy = torch.sum(maternal_filtered_signal**2)
            snr = torch.sqrt(fetal_energy) / noise_std
            selectivity = torch.sqrt(fetal_energy / maternal_energy)
            if snr > best_snr:
                best_snr = snr.item()
                best_snr_index = i
            if selectivity > best_selectivity:
                best_selectivity = selectivity.item()
                best_selectivity_index = i
            if snr * selectivity > best_product:
                best_product = (snr * selectivity).item()
                best_product_index = i

            # Logging the impulse window results
            self.impulse_window_snr_list[i] = snr.item()
            self.impulse_window_selectivity_list[i] = selectivity.item()
            self.impulse_window_product_list[i] = (snr * selectivity).item()
        logger.info("Best Product : %.4f at index %d", best_product, best_product_index)
        return best_snr, best_selectivity, best_snr_index, best_selectivity_index

    def smoothen_window(self, threshold_prct: float = 0.03) -> torch.Tensor:
        """
        Smoothen the window & window norm by zeroing out values below a certain threshold.
        """
        smoothed_window = self.window_norm.clone()
        if self.window_smoothening:
            max_weight = torch.max(self.window)
            threshold = threshold_prct * max_weight
            smoothed_window[smoothed_window < threshold] = 0.0
        return smoothed_window

    def optimize(self):
        """
        Perform the optimization loop and populate self.window and self.training_curves.
        """
        best_metric = -np.inf
        epochs_no_improve = 0
        optimizer = optim.AdamW(
            [self.learnable_component_exponents],
            lr=self.lr,
            weight_decay=self.reg_weight if self.reg_type == "l2" else 0.0,
        )
        logger.info(
            "[DIGSSOptimizer] Using %s regularization (weight=%g)",
            self.reg_type.upper(),
            self.reg_weight,
        )

        epoch = 0
        for epoch in range(self.max_epochs):
            optimizer.zero_grad()
            self.learnable_component = self._winexp_to_win_func(self.learnable_component_exponents)
            self.window = torch.cat([self.fixed_left, self.learnable_component, self.fixed_right], dim=0)
            self.window_norm = self._win_norm_func(self.window)

            # Compute compact statistics
            compact_stats = self.moment_module(self.window_norm)

            # Apply comb filters
            compact_stats = compact_stats - compact_stats.mean()
            compact_stats_reshaped = compact_stats.unsqueeze(0).unsqueeze(0)  # For conv1d
            # Account for the filter's attenuation of energy by scaling
            maternal_filtered_signal = self.maternal_filter(compact_stats_reshaped)
            fetal_filtered_signal = self.fetal_filter(compact_stats_reshaped)
            self.final_signal = fetal_filtered_signal.squeeze().detach().cpu()

            ## Optimize the Target Directly
            fetal_energy = torch.sum(fetal_filtered_signal**2)
            maternal_energy = torch.sum(maternal_filtered_signal**2)
            baseline_noise_var = self.noise_calc.compute_noise(self.tof_data, self.window_norm).sum()
            baseline_noise_std = torch.sqrt(baseline_noise_var)
            selectivity = torch.sqrt(fetal_energy / maternal_energy)
            snr = torch.sqrt(fetal_energy) / baseline_noise_std

            # Normalize by the best possble values
            if self.normalize_reward:
                snr = snr / float(self.max_snr)
                selectivity = selectivity / float(self.max_selectivity)
            final_metric = selectivity * snr

            # Base objective (maximize final_metric)
            loss = -torch.log(final_metric)
            # loss = -selectivity

            self.training_curves[epoch, 0] = selectivity.item()
            self.training_curves[epoch, 1] = snr.item()
            self.training_curves[epoch, 2] = final_metric.item()
            self.training_cruves_extra[epoch, 0] = fetal_energy.item()
            self.training_cruves_extra[epoch, 1] = maternal_energy.item()
            self.training_cruves_extra[epoch, 2] = baseline_noise_std.item()

            # L1 regularization (L2 is already in Adam weight_decay)
            if self.reg_type == "l1" and self.reg_weight > 0:
                loss = loss + self.reg_weight * torch.sum(torch.abs(self.learnable_component_exponents))

            optimizer.zero_grad()
            loss.backward()

            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_([self.learnable_component_exponents], max_norm=1.0)

            optimizer.step()

            # Early stopping check
            if final_metric.item() > best_metric * 1.01:
                best_metric = final_metric.item()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    logger.info("Early stopping at epoch %d", epoch + 1)
                    self.training_curves = self.training_curves[: epoch + 1]
                    self.training_curves_extra = self.training_cruves_extra[: epoch + 1]
                    break

        # Trim training curves if early stopping occurred
        if self.training_curves.shape[0] > epoch + 1:
            self.training_curves = self.training_curves[: epoch + 1]
            self.training_curves_extra = self.training_curves_extra[: epoch + 1]

        # Set the normalized window as final window
        self.unprocessed_window = self.window_norm.clone().detach()
        self.window_norm = self.smoothen_window()
        # self.window_norm = self.window_post_process()
        self.window = self.window_norm.detach()

    def window_post_process(self) -> torch.Tensor:
        """
        Apply custom post-processing"
        """
        if self.normalization_scheme == "unit_max":
            # Find left and right edges
            left_edge = self.window.nonzero(as_tuple=True)[0][0].item()
            right_edge = self.window.nonzero(as_tuple=True)[0][-1].item()
            # Fill everything in-between with 1s
            processed_window = torch.zeros_like(self.window)
            processed_window[left_edge : right_edge + 1] = 1.0
            return processed_window
        else:
            return self.window_norm

    def __str__(self) -> str:
        return (
            f"DIGSSOptimizer(measurand={self.moment_module.__class__.__name__}, "
            f"lr={self.lr}, filter_hw={self.filter_hw}, patience={self.patience}, grad_clip={self.grad_clip},"
            f"fetal_f={self.fetal_f}), type={self.filter_type}, filter_smoothening={self.window_smoothening},"
            f"reg_type={self.reg_type}, reg_weight={self.reg_weight}, normalize_reward={self.normalize_reward},"
            f"normalization_scheme={self.normalization_scheme}"
        )

    def components(self) -> dict[str, nn.Module]:
        """Return the internal components/modules used in optimization."""
        return {
            "moment_module": self.moment_module,
            "fetal_comb_filter": self.fetal_filter,
            "maternal_comb_filter": self.maternal_filter,
        }


def plot_training_curves_and_window(
    training_curves: np.ndarray,
    curve_column_labels: list[str],
    optimized_window: torch.Tensor,
    bin_edges: np.ndarray,
    fig_size: tuple[int, int] = (10, 6),
    grid: bool = False,
    normalize_curves: bool = True,
    filename: str = "optimization_results",
) -> None:
    """
    Plot the training curves and optimized window in two subplots and save to a file.

    :param training_curves: Numpy array of shape (num_epochs, num_metrics) containing the training curves.
    :type training_curves: np.ndarray
    :param curve_column_labels: List of labels for each metric in the training curves.
    :type curve_column_labels: list[str]
    :param fig_size: Figure size for the plots. Defaults to (10, 6).
    :type fig_size: tuple[int, int]
    :param optimized_window: Optimized window tensor.
    :type optimized_window: torch.Tensor
    :param bin_edges: Bin edges tensor.
    :type bin_edges: np.ndarray
    :param grid: Whether to show grid on the training plots. Defaults to False.
    :type grid: bool
    :param normalize_curves: Whether to normalize each training curves for better visualization. Defaults to True.
    :type normalize_curves: bool
    :param filename: Filename to save the plots. Defaults to "optimization_results". Saves to ./figures/{filename}.svg
    and ./figures/{filename}.pdf (Like the professionals we are)
    :type filename: str
    """
    ## Validity Checks
    assert training_curves.shape[1] == len(curve_column_labels), "Number of curve labels must match number of metrics."

    ## Bin Centers
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_centers_ns = bin_centers * 1e9  # Convert to nanoseconds for better readability
    bin_centers_ns = np.round(bin_centers_ns, decimals=2)

    ## Load config for plotting if available
    load_plot_config()

    plt.subplots(1, 2, figsize=fig_size)

    # Plot Training Curves
    plt.subplot(1, 2, 1)
    for i in range(training_curves.shape[1]):
        if normalize_curves:
            curve = training_curves[:, i] / (np.max(np.abs(training_curves[:, i])) + 1e-40)
        else:
            curve = training_curves[:, i]
        plt.plot(curve, label=curve_column_labels[i])
    plt.xlabel("Epoch")
    plt.ylabel("Metric Value")
    plt.yscale("log")
    axes_title = "Normalized Training Metrics" if normalize_curves else "Training Metrics"
    plt.title(axes_title)
    plt.legend()
    plt.grid(grid)

    # Plot Optimized Window
    plt.subplot(1, 2, 2)
    plt.plot(bin_centers_ns, optimized_window.detach().cpu().numpy(), marker="o")
    plt.xlabel("Bin Center (ns)")
    plt.ylabel("Window Value")
    plt.title("Optimized Window")
    plt.tight_layout()

    plt.savefig(f"./figures/{filename}.svg")
    plt.savefig(f"./figures/{filename}.pdf")
    plt.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    # file_idx = 7
    for file_idx in range(4, 5):
        measurand = "abs"
        ppath_file = Path(f"./data/experiment_{file_idx:04d}.npz")
        logger.info("Running optimization loop for file: %04d.npz | Measurand: %s", file_idx, measurand)
        tof_dataset_path = Path("./data") / f"generated_tof_set_{ppath_file.stem}.npz"
        gen_config: dict = yaml.safe_load(open("./experiments/tof_config.yaml"))
        filter_hw = 0.01
        noise_var = 1000.0
        generate_tof(ppath_file, gen_config, tof_dataset_path, True, True)
        tof_data = ToFData.from_npz(tof_dataset_path)
        modifier = AdditiveGaussianToFModifier(noise_var)
        modified_tof = modifier.modify(tof_data)
        modified_tof.to_npz(tof_dataset_path)  # Overwrite
        noise_calc = WindowSumWithAdditiveGaussianNoiseCalculator(noise_var)
        experiment = DIGSSOptimizer(
            tof_dataset_path=tof_dataset_path,
            measurand=measurand,
            noise_calc=noise_calc,
            fetal_f=gen_config["fetal_f"],
            normalize_reward=False,
            lr=0.1,
            filter_hw=filter_hw,
            patience=50,
            reg_type="l2",
            reg_weight=0.00,
            filter_type="psafe_same_width",
            normalization_scheme="unit_max",
        )
        experiment.optimize()

        optimized_window = experiment.window  # type: ignore
        result_curves = experiment.training_curves
        logger.info("Optimized Window: %s", optimized_window.numpy())
        logger.info("Best Selectivity: %s", result_curves[-1, 0])
        logger.info("Best SNR: %s", result_curves[-1, 1])
        logger.info("Best Final Metric: %s", result_curves[-1, 2])
        logger.info("Total Epochs: %s", result_curves.shape[0])
        logger.info("Best SNR Index : %d", experiment.max_snr_index)
        logger.info("Best Selectivity Index : %d", experiment.max_selectivity_index)
        loss_names = experiment.training_curve_labels
        bin_edges = np.load(tof_dataset_path)["bin_edges"]
        logger.info("Training curves sample (every 50 epochs): %s", result_curves[::50, :])
        plot_training_curves_and_window(result_curves, loss_names, optimized_window, bin_edges, normalize_curves=False)

        # Evaluate using an Evaluator and print log
        evaluator = AltPaperEvaluator3(ppath_file, optimized_window, measurand, gen_config, filter_hw)
        eval_results = evaluator.evaluate()
        logger.info("Evaluation Results: %s", eval_results)
        logger.info("Evaluator log: %s", evaluator.get_log())
        logger.info("Max SNR(SB): %.4f at i %d", experiment.max_snr, experiment.max_snr_index)
        logger.info(
            "Max Selectivity(SB): %.4f at i %d",
            experiment.max_selectivity,
            experiment.max_selectivity_index,
        )
        logger.info("Impulse Window SNR List: %s", experiment.impulse_window_snr_list)
        logger.info("Impulse Window Selectivity List: %s", experiment.impulse_window_selectivity_list)
        logger.info("Impulse Window Product List: %s", experiment.impulse_window_product_list)
        logger.info("Unprocessed Window: %s", experiment.unprocessed_window.numpy())
        # Clean up
        tof_dataset_path.unlink()  # Remove the generated ToF dataset to save space


if __name__ == "__main__":
    main()
