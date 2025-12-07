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
)


def main_optimize(
    tof_dataset_path: Path,
    measurand: str | nn.Module,
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
    # Step 1
    data = np.load(tof_dataset_path)

    # Step 2
    tof_series = data["tof_dataset"]  # Shape: (num_timepoints, num_bins)
    bin_edges = data["bin_edges"]  # Shape: (num_bins + 1,)
    sampling_rate = data["sampling_rate"]  # Sampling rate in Hz
    fetal_f = data["fetal_f"]  # Fetal heartbeat frequency in Hz
    maternal_f = data["maternal_f"]  # Maternal heartbeat frequency in Hz
    time_axis = data["time_axis"]  # Time axis
    tof_series_tensor = torch.tensor(tof_series, dtype=torch.float32)
    bin_edges_tensor = torch.tensor(bin_edges, dtype=torch.float32)
    num_timepoints, num_bins = tof_series_tensor.shape
    # Handle measurand and corresponding noise function
    if isinstance(measurand, str):
        if measurand not in named_moment_types:
            raise ValueError(f"Invalid measurand string: {measurand}. Must be one of {named_moment_types}.")
        moment_calculator = get_named_moment_module(measurand, tof_series_tensor, bin_edges_tensor)
        if noise_func is not None:
            print("Warning: noise_func is ignored when using a predefined measurand string.")
        noise_func = noise_func_table[measurand]
    else:
        # Custom measurand module provided
        if noise_func is None:
            raise ValueError("noise_func must be provided when using a custom measurand module.")
        moment_calculator = measurand

    ## Define Comb Filters
    fetal_comb_filter = CombSeparator(sampling_rate, fetal_f, 2 * fetal_f, filter_hw, len(time_axis) // 2 + 1)
    maternal_comb_filter = CombSeparator(sampling_rate, maternal_f, 2 * maternal_f, filter_hw, len(time_axis) // 2 + 1)

    ## Define Metrics
    contrast_to_noise_metric = ContrastToNoiseMetric(noise_func, tof_series_tensor, bin_edges_tensor, False)
    energy_ratio_metric = EnergyRatioMetric()

    # Step 3
    window_exponents = nn.Parameter(torch.ones(num_bins, dtype=torch.float32, requires_grad=True))
    window = torch.exp(window_exponents)
    window_normalized = window / (torch.norm(window) + 1e-20)

    # Prep for Optimization Loop
    best_metric = -np.inf
    epochs_no_improve = 0
    optimizer = optim.Adam([window_exponents], lr=lr)
    training_curves = np.zeros((max_epochs, 3))  # Columns: [Energy Ratio, Contrast-to-Noise, Final Metric]
    epoch = 0
    for epoch in range(max_epochs):
        optimizer.zero_grad()

        # Reparameterize window using exponentiation to ensure positivity
        window = torch.exp(window_exponents)
        # Normalize window to unit energy
        window_normalized = window / (torch.norm(window) + 1e-20)

        # Step 4
        # Compute compact statistics
        compact_stats = moment_calculator(window_normalized)  # Shape: (num_timepoints,)

        # Step 5
        # Apply comb filters
        compact_stats_reshaped = compact_stats.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, num_timepoints) for conv1d
        fetal_filtered_signal = fetal_comb_filter(compact_stats_reshaped)
        maternal_filtered_signal = maternal_comb_filter(compact_stats_reshaped)

        # Step 6, 7, 8
        # Compute metrics
        energy_ratio = energy_ratio_metric(fetal_filtered_signal, maternal_filtered_signal)
        contrast_value = contrast_to_noise_metric(window_normalized, fetal_filtered_signal)
        final_metric = energy_ratio * contrast_value

        # Logging
        training_curves[epoch, 0] = energy_ratio.item()
        training_curves[epoch, 1] = contrast_value.item()
        training_curves[epoch, 2] = final_metric.item()

        # Step 9
        # Backpropagation
        loss = -final_metric
        loss.backward()
        optimizer.step()

        # Early Stopping Check
        if final_metric.item() > best_metric * 1.01:
            best_metric = final_metric.item()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Step 10
    # Return the optimized window and training curves up to the last epoch
    return window_normalized.detach(), training_curves[: epoch + 1]


def plot_training_curves_and_window(
    training_curves: np.ndarray,
    curve_column_labels: list[str],
    optimized_window: torch.Tensor,
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
    plt.plot(optimized_window.numpy(), marker="o")
    plt.xlabel("Timebin Index")
    plt.ylabel("Window Value")
    plt.title("Optimized Window")
    plt.tight_layout()

    plt.savefig(f"./figures/{filename}.svg")
    plt.savefig(f"./figures/{filename}.pdf")


if __name__ == "__main__":
    data_path = Path("./data/generated_tof_set.npz")
    optimized_window, training_curves = main_optimize(
        tof_dataset_path=data_path,
        measurand="V",
        max_epochs=2000,
        lr=0.01,
        filter_hw=0.3,
        patience=50,
    )
    print("Optimized Window:", optimized_window.numpy())
    print("Best Final Metric:", training_curves[-1, 2])
    print("Total Epochs:", training_curves.shape[0])
    loss_names = ["Energy Ratio", "Contrast-to-Noise", "Final Metric"]
    plot_training_curves_and_window(training_curves, loss_names, optimized_window, grid=True)
