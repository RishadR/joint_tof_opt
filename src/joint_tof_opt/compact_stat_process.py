"""
Code to compute compact statistics from time-of-flight (TOF) data for joint optimization tasks.
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from joint_tof_opt.core import CompactStatProcess, ToFData


class WindowedSum(CompactStatProcess):
    """
    Applies point-wise multiplication with a window and computes row-wise sum.

    This module multiplies each ToF histogram with a window function and sums
    across bins to produce a single value per ToF.
    """

    def __init__(self, tof_data: ToFData):
        super().__init__(tof_data)
        self.num_tofs, self.num_bins = tof_data.tof_series.shape

    def forward(self, window: torch.Tensor) -> torch.Tensor:
        """
        Apply window and compute row-wise sum.

        :param window: 1D tensor with same length as number of bins.
        :type window: torch.Tensor
        :return: 1D tensor of summed values for each ToF (flattened).
        :rtype: torch.Tensor
        """
        # Point-wise multiply and sum across bins (dim=1)
        # normalizer = torch.sum(self.tof_series, dim=1) + 1e-20
        # result = (self.tof_series * window).sum(dim=1).flatten() / normalizer
        result = (self.tof_series * window).sum(dim=1).flatten()
        return result


class NthOrderMoment(CompactStatProcess):
    """
    Computes the n-th order moment of time for windowed ToF histograms.

    This module multiplies each ToF histogram with a window function and computes
    the n-th order moment using bin centers: Σ(t^n * h(t)) / Σ(h(t))
    """

    def __init__(self, tof_data: ToFData, order: int):
        super().__init__(tof_data)
        self.num_tofs, self.num_bins = tof_data.tof_series.shape
        self.order = order

    def forward(self, window: torch.Tensor) -> torch.Tensor:
        """
        Apply window and compute n-th order moment for each ToF.

        :param window: 1D tensor with same length as number of bins.
        :type window: torch.Tensor
        :return: 1D tensor of n-th order moments for each ToF (flattened).
        :rtype: torch.Tensor
        """
        # Point-wise multiply with window
        normalizer = torch.sum(self.tof_series, dim=1)
        windowed_histograms = self.tof_series * window

        # Compute numerator: Σ(t^n * h(t))
        moment = (windowed_histograms * self.bin_centers**self.order).sum(dim=1) / normalizer
        return moment.flatten()


class NthOrderCenteredMoment(CompactStatProcess):
    """
    Computes the n-th order centered moment of time for windowed ToF histograms.

    This module multiplies each ToF histogram with a window function and computes
    the n-th order centered moment: Σ((t - μ)^n * h(t)) / Σ(h(t))
    where μ is the mean time (1st moment).
    """

    def __init__(self, tof_data: ToFData, order: int):
        super().__init__(tof_data)
        self.num_tofs, self.num_bins = tof_data.tof_series.shape
        self.order = order

    def forward(self, window: torch.Tensor) -> torch.Tensor:
        """
        Apply window and compute n-th order centered moment for each ToF.

        :param window: 1D tensor with same length as number of bins.
        :type window: torch.Tensor
        :return: 1D tensor of n-th order centered moments for each ToF (flattened).
        :rtype: torch.Tensor
        """
        # Point-wise multiply with window
        windowed_histograms = self.tof_series * window
        normalizer = torch.sum(windowed_histograms, dim=1, keepdim=True)
        mean_time = (windowed_histograms * self.bin_centers.unsqueeze(0)).sum(dim=1, keepdim=True) / normalizer

        # Compute centered values: (t - μ)
        centered_times = self.bin_centers.unsqueeze(0) - mean_time

        # Compute (t - μ)^n
        centered_times_pow = centered_times**self.order

        # Compute n-th centered moment: Σ((t - μ)^n * h(t)) / Σ(h(t))
        centered_moment = (windowed_histograms * centered_times_pow).sum(dim=1) / normalizer.flatten()
        return centered_moment


# Gather all moment modules for easy access
# Single source of truth: define moment configurations once
# Add new moment types here to make them accessible throughout the package
MOMENT_CONFIGS = {
    "abs": lambda tof_data: WindowedSum(tof_data),
    "m1": lambda tof_data: NthOrderMoment(tof_data, order=1),
    "V": lambda tof_data: NthOrderCenteredMoment(tof_data, order=2),
}

named_moment_types = list(MOMENT_CONFIGS.keys())


def get_named_moment_module(moment_type: str, tof_data: ToFData) -> CompactStatProcess:
    if moment_type not in MOMENT_CONFIGS:
        raise ValueError(f"Invalid moment type: {moment_type}")

    config = MOMENT_CONFIGS[moment_type]
    return config(tof_data)
