"""
Code to compute compact statistics from time-of-flight (TOF) data for joint optimization tasks.
"""

import torch
import torch.nn as nn


class WindowedSum(nn.Module):
    """
    Applies point-wise multiplication with a window and computes row-wise sum.

    This module multiplies each ToF histogram with a window function and sums
    across bins to produce a single value per ToF.
    """

    def __init__(self, tof_series: torch.Tensor, bin_edges: torch.Tensor):
        """
        Initialize the MultiplyAndSum module.

        :param tof_histograms: 2D tensor where each row is a ToF histogram and each column is a bin.
        :type tof_histograms: torch.Tensor
        :param bin_edges: 1D tensor of bin edges (length = num_bins + 1).
        :type bin_edges: torch.Tensor
        """
        super().__init__()
        self.num_tofs, self.num_bins = tof_series.shape

        # Compute bin centers from edges
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        self.bin_centers = bin_centers
        self.tof_series = tof_series

    def forward(self, window: torch.Tensor) -> torch.Tensor:
        """
        Apply window and compute row-wise sum.

        :param window: 1D tensor with same length as number of bins.
        :type window: torch.Tensor
        :return: 1D tensor of summed values for each ToF (flattened).
        :rtype: torch.Tensor
        """
        # Point-wise multiply and sum across bins (dim=1)
        normalizer = torch.sum(self.tof_series, dim=1) + 1e-20
        result = (self.tof_series * window).sum(dim=1).flatten() / normalizer
        return result


class NthOrderMoment(nn.Module):
    """
    Computes the n-th order moment of time for windowed ToF histograms.

    This module multiplies each ToF histogram with a window function and computes
    the n-th order moment using bin centers: Σ(t^n * h(t)) / Σ(h(t))
    """

    def __init__(self, tof_series: torch.Tensor, bin_edges: torch.Tensor, order: int):
        """
        Initialize the NthOrderMoment module.

        :param tof_histograms: 2D tensor where each row is a ToF histogram and each column is a bin.
        :type tof_histograms: torch.Tensor
        :param bin_edges: 1D tensor of bin edges (length = num_bins + 1).
        :type bin_edges: torch.Tensor
        :param order: The order of the moment (e.g., 1 for mean, 2 for second moment).
        :type order: int
        """
        super().__init__()
        self.num_tofs, self.num_bins = tof_series.shape
        self.order = order

        # Compute bin centers from edges
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        self.bin_centers = bin_centers
        self.order = order
        self.tof_series = tof_series

    def forward(self, window: torch.Tensor) -> torch.Tensor:
        """
        Apply window and compute n-th order moment for each ToF.

        :param window: 1D tensor with same length as number of bins.
        :type window: torch.Tensor
        :return: 1D tensor of n-th order moments for each ToF (flattened).
        :rtype: torch.Tensor
        """
        # Point-wise multiply with window
        normalizer = torch.sum(self.tof_series, dim=1) + 1e-20
        windowed_histograms = self.tof_series * window

        # Compute numerator: Σ(t^n * h(t))
        moment = (windowed_histograms * self.bin_centers**self.order).sum(dim=1) / normalizer
        return moment.flatten()


class NthOrderCenteredMoment(nn.Module):
    """
    Computes the n-th order centered moment of time for windowed ToF histograms.

    This module multiplies each ToF histogram with a window function and computes
    the n-th order centered moment: Σ((t - μ)^n * h(t)) / Σ(h(t))
    where μ is the mean time (1st moment).
    """

    def __init__(self, tof_series: torch.Tensor, bin_edges: torch.Tensor, order: int):
        """
        Initialize the NthOrderCenteredMoment module.

        :param tof_series: 2D tensor where each row is a ToF histogram and each column is a bin.
        :type tof_series: torch.Tensor
        :param bin_edges: 1D tensor of bin edges (length = num_bins + 1).
        :type bin_edges: torch.Tensor
        :param order: The order of the centered moment (e.g., 2 for variance, 3 for skewness basis).
        :type order: int
        """
        super().__init__()
        self.num_tofs, self.num_bins = tof_series.shape
        self.order = order

        # Compute bin centers from edges
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        self.bin_centers = bin_centers
        self.tof_series = tof_series
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
        normalizer = torch.sum(self.tof_series, dim=1) + 1e-20
        windowed_histograms = self.tof_series * window
        mean_time = (windowed_histograms * self.bin_centers.unsqueeze(0)).sum(dim=1, keepdim=True) / normalizer.unsqueeze(1)

        # Compute centered values: (t - μ)
        centered_times = self.bin_centers.unsqueeze(0) - mean_time

        # Compute (t - μ)^n
        centered_times_pow = centered_times**self.order

        # Compute n-th centered moment: Σ((t - μ)^n * h(t)) / Σ(h(t))
        centered_moment = (windowed_histograms * centered_times_pow).sum(dim=1) / normalizer
        return centered_moment.flatten()


