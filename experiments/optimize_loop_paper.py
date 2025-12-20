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
from joint_tof_opt import (
    CombSeparator,
    EnergyRatioMetric,
    ContrastToNoiseMetric,
    get_named_moment_module,
    named_moment_types,
    noise_func_table,
    OptimizationExperiment,
    CompactStatProcess,
)

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
        noise_func: None | Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor] = None,
        max_epochs: int = 2000,
        lr: float = 0.001,
        filter_hw: float = 0.3,
        patience: int = 20,
    ):
        """
        Initialize the PaperOptimizer.

        :param tof_dataset_path: Path to the ToF dataset (.npz file).
        :param measurand: The measurand to optimize for ("abs", "m1", "V") or custom module.
        :param noise_func: Noise function for custom measurands.
        :param max_epochs: Maximum number of optimization epochs.
        :param lr: Learning rate for the optimizer.
        :param filter_hw: Half width of the sinc comb filter (in Hz).
        :param patience: Number of epochs to wait for improvement before early stopping.
        """
        super().__init__(tof_dataset_path, measurand, lr)

        self.max_epochs = max_epochs
        self.filter_hw = filter_hw
        self.patience = patience

        # Handle measurand and noise function
        if isinstance(measurand, str):
            if measurand not in named_moment_types:
                raise ValueError(f"Invalid measurand string: {measurand}. Must be one of {named_moment_types}.")
            if noise_func is not None:
                print("Warning: noise_func is ignored when using a predefined measurand string.")
            self.noise_func = noise_func_table[measurand]
        else:
            if noise_func is None:
                raise ValueError("noise_func must be provided when using a custom measurand module.")
            self.noise_func = noise_func

        # Convert bin edges to nanoseconds for numerical stability
        self.bin_edges *= 1e9
        # moment_module is originally initialized by the parent class but since we change the bin_centers and edges
        # Keep everything else unchanged
        self.moment_module.bin_edges = self.bin_edges
        self.moment_module.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2

        # Extract additional metadata
        self.sampling_rate = self.tof_data["sampling_rate"]
        self.fetal_f = self.tof_data["fetal_f"]
        self.maternal_f = self.tof_data["maternal_f"]
        num_timepoints, num_bins = self.tof_series.shape

        # Initialize components (to be created in optimize())
        self.fetal_comb_filter = CombSeparator(
            self.sampling_rate, 
            self.fetal_f, 2 * self.fetal_f,
            self.filter_hw,
            num_timepoints // 2 + 1,
            False
        )
        self.maternal_comb_filter = CombSeparator(
            self.sampling_rate,
            self.maternal_f,
            2 * self.maternal_f,
            self.filter_hw,
            num_timepoints // 2 + 1,
            False
        )
        self.contrast_to_noise_metric = ContrastToNoiseMetric(self.noise_func, self.tof_series, self.bin_edges, False)
        self.energy_ratio_metric = EnergyRatioMetric()

        # Parameters
        self.window_exponents = nn.Parameter(0.5 * torch.zeros(num_bins, dtype=torch.float32, requires_grad=True))
        self.window = torch.exp(self.window_exponents)
        self.window_normalized = self.window / torch.norm(self.window)

        # Set training curve labels
        self.training_curve_labels = ["Energy Ratio", "Contrast-to-Noise", "Final Metric"]

    def optimize(self):
        """
        Perform the optimization loop and populate self.window and self.training_curves.
        """
        num_timepoints, num_bins = self.tof_series.shape
        # Initialize window parameters

        # Prep for optimization loop
        best_metric = -np.inf
        epochs_no_improve = 0
        optimizer = optim.Adam([self.window_exponents], lr=self.lr)
        self.training_curves = np.zeros((self.max_epochs, 3))

        epoch = 0
        for epoch in range(self.max_epochs):
            optimizer.zero_grad()
            # Reparameterize window using exponentiation to ensure positivity
            self.window = torch.exp(self.window_exponents)
            # Normalize window to unit energy
            self.window_normalized = self.window / torch.norm(self.window)

            # Compute compact statistics
            compact_stats = self.moment_module(self.window_normalized)

            # Apply comb filters
            compact_stats_reshaped = compact_stats.unsqueeze(0).unsqueeze(0)  # For conv1d
            fetal_filtered_signal = self.fetal_comb_filter(compact_stats_reshaped)
            maternal_filtered_signal = self.maternal_comb_filter(compact_stats_reshaped)
            self.final_signal = fetal_filtered_signal.squeeze().detach().cpu()

            # Compute metrics
            energy_ratio = self.energy_ratio_metric(fetal_filtered_signal, maternal_filtered_signal)
            contrast_value = self.contrast_to_noise_metric(self.window_normalized, fetal_filtered_signal)
            final_metric = energy_ratio * contrast_value

            # Log metrics
            self.training_curves[epoch, 0] = energy_ratio.item()
            self.training_curves[epoch, 1] = contrast_value.item()
            self.training_curves[epoch, 2] = final_metric.item()

            # Backpropagation
            loss = -torch.log(final_metric)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([self.window_exponents], max_norm=1.0)
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

        # Store final window as a normalized tensor
        self.window = self.window_normalized

        # Trim training curves if early stopping occurred
        if self.training_curves.shape[0] > epoch + 1:
            self.training_curves = self.training_curves[: epoch + 1]

    def __str__(self) -> str:
        return (
            f"DIGSSOptimizer(measurand={self.moment_module.__class__.__name__}, "
            f"lr={self.lr}, filter_hw={self.filter_hw}, patience={self.patience})"
        )

    def components(self) -> dict[str, nn.Module]:
        """Return the internal components/modules used in optimization."""
        return {
            "moment_module": self.moment_module,
            "fetal_comb_filter": self.fetal_comb_filter,
            "maternal_comb_filter": self.maternal_comb_filter,
            "contrast_to_noise_metric": self.contrast_to_noise_metric,
            "energy_ratio_metric": self.energy_ratio_metric,
        }

# Functional Interface
def main_optimize(
    tof_dataset_path: Path,
    measurand: str | CompactStatProcess,
    noise_func: None | Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor] = None,
    max_epochs: int = 2000,
    lr: float = 0.001,
    filter_hw: float = 0.3,
    patience: int = 20,
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
    :param noise_func: Function to compute the noise for the given measurand. Required if a custom measurand module is
    provided. Note: The function signature should be:
        noise_func(tof_series: torch.Tensor, bin_edges: torch.Tensor, window: torch.Tensor, ) -> torch.Tensor
    :type noise_func: None | Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
    :param max_epochs: Maximum number of optimization epochs.
    :type max_epochs: int
    :param lr: Learning rate for the optimizer.
    :type lr: float
    :param filter_hw: Half width of the sinc comb filter(in Hz).
    :type filter_hw: float
    :param patience: Number of epochs to wait for improvement before early stopping.
    :type patience: int
    :return: Tuple containing the optimized window tensor and a numpy array of training curves. The training curves
    array has shape (max_epochs, 3) corresponding to [Energy Ratio, Contrast-to-Noise, Final Metric] at each epoch.
    :rtype: tuple[torch.Tensor, np.ndarray]
    """
    optimizer = DIGSSOptimizer(
        tof_dataset_path=tof_dataset_path,
        measurand=measurand,
        noise_func=noise_func,
        max_epochs=max_epochs,
        lr=lr,
        filter_hw=filter_hw,
        patience=patience,
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
    config_path = Path("./experiments/plot_config.yaml")
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
            curve = training_curves[:, i] / (np.max(np.abs(training_curves[:, i])) + 1e-20)
        else:
            curve = training_curves[:, i]
        plt.plot(curve, label=curve_column_labels[i], linestyle=linestyles[i])
    plt.xlabel("Epoch")
    plt.ylabel("Metric Value")
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
    tof_dataset_path = Path("./data/generated_tof_set_experiment_0000.npz")
    optimized_window, training_curves = main_optimize(
        tof_dataset_path=tof_dataset_path,
        measurand="V",
        max_epochs=2000,
        lr=1e-5,
        filter_hw=0.3,
        patience=50,
    )
    print("Optimized Window:", optimized_window.numpy())
    print("Best Final Metric:", training_curves[-1, 2])
    print("Total Epochs:", training_curves.shape[0])
    loss_names = ["Energy Ratio", "Contrast-to-Noise", "Final Metric"]
    bin_edges = np.load(tof_dataset_path)["bin_edges"]
    # print(training_curves[::10, :])
    # plot_training_curves_and_window(training_curves, loss_names, optimized_window, bin_edges, grid=True)
