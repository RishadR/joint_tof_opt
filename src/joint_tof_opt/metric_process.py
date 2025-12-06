"""
The final block to the optimization problem. Computes the the final metric that should be maximized.

METRIC MODULES RULE:
- Each metric is a torch.nn.Module
- The forward() method signature varies per metric (different inputs)
- The output must always be a single scalar tensor (shape: ())
"""

import torch
import torch.nn as nn
from joint_tof_opt.noise_calc import NoiseFunc


class EnergyRatioMetric(nn.Module):
    """
    Computes the energy ratio between filtered and original signal as a metric.

    The energy ratio is defined as:
        metric = energy(filtered_signal) / energy(original_signal)

    A higher ratio indicates that more energy is concentrated in the filtered signal,
    suggesting better signal separation or filtering effectiveness.

    Input:
        filtered_signal: Tensor of shape (batch_size, signal_length) or (signal_length,)
        original_signal: Tensor of shape (batch_size, signal_length) or (signal_length,)

    Output:
        Scalar tensor with the energy ratio
    """

    def __init__(self):
        super().__init__()

    def forward(
        self, filtered_signal: torch.Tensor, original_signal: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute energy ratio metric.

        Parameters
        ----------
        filtered_signal : torch.Tensor
            The filtered signal
        original_signal : torch.Tensor
            The original unfiltered signal

        Returns
        -------
        torch.Tensor
            Scalar tensor containing the energy ratio
        """
        # Compute energy (L2 norm squared) for each signal
        filtered_energy = torch.sum(filtered_signal**2)
        original_energy = torch.sum(original_signal**2)

        # Compute ratio, adding small epsilon to avoid division by zero
        epsilon = 1e-20
        energy_ratio = filtered_energy / (original_energy + epsilon)

        return energy_ratio


class ContrastToNoiseMetric(nn.Module):
    """
    Computes the contrast-to-noise ratio (CNR) as a metric.

    The CNR is defined as:
        CNR = (mean_signal - mean_background) / std_background
    A higher CNR indicates better distinguishability of the signal from the background noise.
    Input:
        signal_region: Tensor of shape (batch_size, signal_length) or (signal_length,)
        background_region: Tensor of shape (batch_size, background_length) or (background_length,)
    Output:
        Scalar tensor with the CNR value
    """

    def __init__(
        self, noise_func: NoiseFunc, tof_series: torch.Tensor, bin_edges: torch.Tensor, dB_scale: bool = False
    ):
        super().__init__()
        self.noise_func = noise_func
        self.tof_series = tof_series
        self.bin_edges = bin_edges
        self.dB_scale = dB_scale

    def forward(self, window: torch.Tensor, filtered_signal: torch.Tensor) -> torch.Tensor:
        noise = self.noise_func(self.tof_series, self.bin_edges, window)    # sigma^2
        noise_std = noise.mean().sqrt()
        filtered_signal_energy = torch.mean(filtered_signal**2)  # mu^2
        filtered_signal_amp = torch.sqrt(filtered_signal_energy)  # mu
        contrast = filtered_signal_amp / (noise_std + 1e-20)
        if self.dB_scale:
            contrast = 20 * torch.log10(contrast + 1e-30)  # Convert to dB scale, add epsilon to avoid log(0)
        return contrast