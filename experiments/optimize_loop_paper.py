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

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Callable, Literal
from pathlib import Path
import matplotlib.pyplot as plt
from joint_tof_opt import *
from sensitivity_compute import *


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
        noise_calc: None | NoiseCalculator = None,
        fetal_f: float | None = None,
        max_epochs: int = 2000,
        lr: float = 0.001,
        filter_hw: float = 0.3,
        patience: int = 20,
        grad_clip: bool = False,
        l2_reg: float = 1e-4,
        normalize_tof: bool = False,
        filter_type: Literal[
            "comb", "fourier", "psafe_same_width", "psafe_true_width", "comb_psafe_hybrid"
        ] = "psafe_same_width",
    ):
        """
        Initialize the PaperOptimizer.

        :param tof_dataset_path: Path to the ToF dataset (.npz file).
        :param measurand: The measurand to optimize for ("abs", "m1", "V") or custom module.
        :param noise_calc: Noise calculator for custom measurands.
        :param fetal_f: Central frequency of fetal comb filter (in Hz). If None, extracted from dataset metadata.
        :param max_epochs: Maximum number of optimization epochs.
        :param fetal_f: Central frequency of fetal comb filter (in Hz). If None, extracted from dataset metadata.
        :param lr: Learning rate for the optimizer.
        :param filter_hw: Half width of the sinc comb filter (in Hz).
        :param patience: Number of epochs to wait for improvement before early stopping.
        :param grad_clip: Whether to apply gradient clipping.
        :param l2_reg: L2 regularization weight.
        :param normalize_tof: Whether to convert tof series into a probability density (Sums to 1) or keep original
        counts. Default is False (keep original counts).
        :param filter_type: Type of filter to use ("comb", "fourier", "psafe_same_width", "psafe_true_width", "comb_psafe_hybrid").
         - "comb": Standard sinc comb filter
         - "fourier": Uses the RFFT to manually zero out irrelevant frequencies
         - "psafe_same_width": Uses the PSAFE filter with the outputs being the same length as the input
         - "psafe_true_width": Uses the PSAFE filter with the outputs being the length of the true fetal/maternal period
         - "comb_psafe_hybrid": Uses the comb for maternal/psafe for fetal to account for FHR being a multiple of MHR
        """
        # Handle measurand and noise function
        if isinstance(measurand, str):
            if measurand not in named_moment_types:
                raise ValueError(f"Invalid measurand string: {measurand}. Must be one of {named_moment_types}.")
            if noise_calc is not None:
                print("Warning: noise_calc is ignored since a predefined measurand string is given")
            self.noise_calc = get_noise_calculator(measurand)
        else:
            if noise_calc is None:
                raise ValueError("noise_calc must be provided when using a custom measurand module.")
            self.noise_calc = noise_calc

        if isinstance(measurand, str):
            tof_data = ToFData.from_npz(tof_dataset_path)
            if normalize_tof:
                tof_data.tof_series /= torch.sum(tof_data.tof_series, dim=1, keepdim=True)
            measurand = get_named_moment_module(measurand, tof_data)
            if normalize_tof:
                measurand.tof_data.tof_series /= torch.sum(measurand.tof_data.tof_series, dim=1, keepdim=True)
        super().__init__(tof_dataset_path, measurand, lr)

        self.max_epochs = max_epochs
        self.filter_hw = filter_hw
        self.patience = patience
        self.grad_clip = grad_clip
        self.normalize_tof = normalize_tof
        self.l2_reg = l2_reg
        self.filter_type = filter_type

        # Extract additional metadata
        assert self.tof_data.meta_data is not None, "ToFData meta_data cannot be None"
        self.sampling_rate = self.tof_data.meta_data["sampling_rate"]
        self.fetal_f = fetal_f if fetal_f is not None else self.tof_data.meta_data["fetal_f"]
        self.maternal_f = self.tof_data.meta_data["maternal_f"]
        num_timepoints, num_bins = self.tof_data.tof_series.shape

        # Compute the Average ToF Frame
        average_tof_frame = self.tof_data.tof_series.mean(dim=0, keepdim=False)
        ## Only have Non-Zero, Learnable Parameters **AFTER** the max index
        max_index = int(torch.argmax(average_tof_frame).item()) + 1
        # max_index = 0
        self.learnable_component_exponents = torch.nn.Parameter(torch.zeros(num_bins - max_index), requires_grad=True)
        self.learnable_component = self._winexp_to_win_func(self.learnable_component_exponents)
        self.fixed_components = torch.zeros(
            max_index,
            dtype=self.learnable_component_exponents.dtype,
            device=self.learnable_component_exponents.device,
        )
        self.window = torch.cat([self.fixed_components, self.learnable_component], dim=0)
        self.window_norm = self._win_norm_func(self.window)

        self.fetal_filter, self.maternal_filter = self._get_filters(filter_type)

        # Set training curve labels
        self.training_curve_labels = ["Energy Ratio", "Contrast-to-Noise", "Final Metric"]

    def _get_filters(self, filter_type: str):
        filter_len = int(2 * self.sampling_rate / self.fetal_f)  # Ensure at least 2 periods are captured
        if filter_type == "comb":
            f1 = CombSeparator(self.sampling_rate, self.fetal_f, self.fetal_f * 2, self.filter_hw, filter_len, True)
            f2 = CombSeparator(
                self.sampling_rate, self.maternal_f, self.maternal_f * 2, self.filter_hw, filter_len, True
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
                self.sampling_rate, self.maternal_f, self.maternal_f * 2, self.filter_hw, filter_len, True
            )
        else:
            raise NotImplementedError(f"Filter type {filter_type} not recognized")
        return f1, f2

    @staticmethod
    def _win_norm_func(win: torch.Tensor) -> torch.Tensor:
        return win / torch.norm(win, p=1)

    @staticmethod
    def _winexp_to_win_func(win_exp: torch.Tensor) -> torch.Tensor:
        # return torch.sigmoid(win_exp)
        # return torch.clamp(torch.relu(win_exp), min=1e-9, max=1.0)
        return torch.exp(win_exp)

    def optimize(self):
        """
        Perform the optimization loop and populate self.window and self.training_curves.
        """
        # Prep for optimization loop
        best_metric = -np.inf
        epochs_no_improve = 0
        optimizer = optim.Adam([self.learnable_component_exponents], lr=self.lr, weight_decay=self.l2_reg)
        self.training_curves = np.zeros((self.max_epochs, 3))

        # Optimization Loop
        epoch = 0
        for epoch in range(self.max_epochs):
            optimizer.zero_grad()
            self.learnable_component = self._winexp_to_win_func(self.learnable_component_exponents)
            self.window = torch.cat([self.fixed_components, self.learnable_component], dim=0)
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
            avergae_tof_frame = self.tof_data.tof_series.sum(dim=0, keepdim=True)
            windowed_average_tof_frame = avergae_tof_frame * self.window_norm.reshape(1, -1)
            baseline_noise_var = windowed_average_tof_frame.sum()
            baseline_noise_std = torch.sqrt(baseline_noise_var)
            final_metric = (fetal_energy) / (torch.sqrt(maternal_energy) * baseline_noise_std)

            # Log metrics
            self.training_curves[epoch, 0] = fetal_energy.item()
            self.training_curves[epoch, 1] = (torch.sqrt(maternal_energy) * baseline_noise_std).item()
            self.training_curves[epoch, 2] = final_metric.item()

            # Backpropagation
            ## Divisions and Products are very unstable - Use Logarithms
            ## Also, we want to maximize the final metric, so minimize negative final metric
            loss = -(torch.log(fetal_energy) - 0.5 * torch.log(maternal_energy) - torch.log(baseline_noise_std))
            # loss = - (torch.log(fetal_energy * 3.0) - 0.5 * torch.log(maternal_energy*9) - torch.log(baseline_noise_std))
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
                    print(f"Early stopping at epoch {epoch + 1}")
                    self.training_curves = self.training_curves[: epoch + 1]
                    break

        # Trim training curves if early stopping occurred
        if self.training_curves.shape[0] > epoch + 1:
            self.training_curves = self.training_curves[: epoch + 1]

        # Set the normalized window as final window
        self.window = self.window_norm.detach()

    def __str__(self) -> str:
        return (
            f"DIGSSOptimizer(measurand={self.moment_module.__class__.__name__}, "
            f"lr={self.lr}, filter_hw={self.filter_hw}, patience={self.patience}, grad_clip={self.grad_clip},"
            f"normalize_tof={self.normalize_tof}, fetal_f={self.fetal_f}), type={self.filter_type}"
        )

    def components(self) -> dict[str, nn.Module]:
        """Return the internal components/modules used in optimization."""
        return {
            "moment_module": self.moment_module,
            "fetal_comb_filter": self.fetal_filter,
            "maternal_comb_filter": self.maternal_filter,
        }


# Functional Interface
def main_optimize(
    tof_dataset_path: Path,
    measurand: str | CompactStatProcess,
    noise_calc: None | NoiseCalculator = None,
    max_epochs: int = 2000,
    lr: float = 0.01,
    l2_reg: float = 1e-4,
    filter_hw: float = 0.3,
    patience: int = 50,
    normalize_tof: bool = False,
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
    :param noise_calc: Noise calculator for the given measurand. Required if a custom measurand module is
    provided. Note: The function signature should be:
        noise_calc(tof_series: torch.Tensor, bin_edges: torch.Tensor, window: torch.Tensor, ) -> torch.Tensor
    :type noise_calc: None | NoiseCalculator
    :param max_epochs: Maximum number of optimization epochs.
    :type max_epochs: int
    :param lr: Learning rate for the optimizer.
    :type lr: float
    :param filter_hw: Half width of the sinc comb filter(in Hz).
    :type filter_hw: float
    :param patience: Number of epochs to wait for improvement before early stopping.
    :type patience: int
    :param normalize_tof: Whether to normalize the TOF data before optimization. Defaults to False.
    :type normalize_tof: bool
    :return: Tuple containing the optimized window tensor and a numpy array of training curves. The training curves
    array has shape (max_epochs, 3) corresponding to [Energy Ratio, Contrast-to-Noise, Final Metric] at each epoch.
    :rtype: tuple[torch.Tensor, np.ndarray]
    """
    optimizer = DIGSSOptimizer(
        tof_dataset_path=tof_dataset_path,
        measurand=measurand,
        noise_calc=noise_calc,
        max_epochs=max_epochs,
        lr=lr,
        l2_reg=l2_reg,
        filter_hw=filter_hw,
        patience=patience,
        normalize_tof=normalize_tof,
    )
    optimizer.optimize()
    return optimizer.window.detach(), optimizer.training_curves


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
    config_path = Path("./plotting_codes/plot_config.yaml")
    if config_path.exists():
        with open(config_path, "r") as f:
            plot_config = yaml.safe_load(f)
            plt.rcParams.update(plot_config)

    linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2, 1, 2))]

    plt.subplots(1, 2, figsize=fig_size)

    # Plot Training Curves
    plt.subplot(1, 2, 1)
    for i in range(training_curves.shape[1]):
        if normalize_curves:
            curve = training_curves[:, i] / (np.max(np.abs(training_curves[:, i])) + 1e-40)
        else:
            curve = training_curves[:, i]
        plt.plot(curve, label=curve_column_labels[i], linestyle=linestyles[i])
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


if __name__ == "__main__":
    file_idx = 0
    measurand = "abs"
    ppath_file = Path(f"./data/experiment_{file_idx:04d}.npz")
    print(f"Running optimization loop for file: {file_idx:04d}.npz | Measurand: {measurand}")
    tof_dataset_path = Path("./data") / f"generated_tof_set_{ppath_file.stem}.npz"
    gen_config = yaml.safe_load(open("./experiments/tof_config.yaml", "r"))
    filter_hw = 0.001
    generate_tof(ppath_file, gen_config, tof_dataset_path, True, True)

    optimized_window, training_curves = main_optimize(
        tof_dataset_path=tof_dataset_path,
        measurand=measurand,
        max_epochs=2000,
        lr=0.01,
        filter_hw=filter_hw,
        patience=100,
        normalize_tof=False,
    )
    print("Optimized Window:", optimized_window.numpy())
    print("Best Final Metric:", training_curves[-1, 2])
    print("Total Epochs:", training_curves.shape[0])
    loss_names = ["Fetal Energy", "Noise x Maternal Amp", "Final Metric"]
    bin_edges = np.load(tof_dataset_path)["bin_edges"]
    print(training_curves[::50, :])
    plot_training_curves_and_window(
        training_curves, loss_names, optimized_window, bin_edges, normalize_curves=False, grid=True
    )

    # Evaluate using an Evaluator and print log
    evaluator = AltPaperEvaluator2(ppath_file, optimized_window, measurand, gen_config, filter_hw)
    eval_results = evaluator.evaluate()
    print(f"Evaluation Results: {eval_results}")
    print(f"Evaluator log: {evaluator.get_log()}")
