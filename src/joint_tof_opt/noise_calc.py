"""
Analytical noise calculation for different compact statistics. The noise is always expressed as noise variance
(sigma^2)
Taken from: https://doi.org/10.1117/1.JBO.17.5.057005
"""

from typing import Callable
from joint_tof_opt.compact_stat_process import NthOrderMoment, WindowedSum, NthOrderCenteredMoment
import torch

NoiseFunc = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


def compute_noise_window_sum(tof_series: torch.Tensor, bin_edges: torch.Tensor, window: torch.Tensor):
    """
    Compute the analytical noise for the windowed sum compact statistic.

    :param tof_series: 2D tensor where each row is a ToF histogram and each column is a bin.
    :type tof_series: torch.Tensor
    :param bin_edges: 1D tensor of bin edges (length = num_bins + 1).
    :type bin_edges: torch.Tensor
    :param window: 1D tensor with same length as number of bins.
    :type window: torch.Tensor
    :return: Analytical noise values for each of the ToF for windowed sum statistic.
    :rtype: torch.Tensor
    """
    # Compute the weighted sum of the ToF series with the window
    weighted_tof = tof_series * window.unsqueeze(0).abs()  # Shape: (num_timepoints, num_bins)
    # The absolute value ensures that noise contributions are non-negative
    noise = weighted_tof.sum(dim=1)  # Shape: (num_timepoints,)
    return noise


def compute_noise_m1(tof_series: torch.Tensor, bin_edges: torch.Tensor, window: torch.Tensor):
    """
    Compute the analytical noise for the first order non-centered moment compact statistic.

    :param tof_series: 2D tensor where each row is a ToF histogram and each column is a bin.
    :type tof_series: torch.Tensor
    :param bin_edges: 1D tensor of bin edges (length = num_bins + 1).
    :type bin_edges: torch.Tensor
    :param window: 1D tensor with same length as number of bins.
    :type window: torch.Tensor
    :return: Analytical noise value for the first order moment statistic.
    :rtype: torch.Tensor
    """
    variance_calculator = NthOrderCenteredMoment(tof_series, bin_edges, order=2)
    variance = variance_calculator(window)  # Shape: (num_timepoints,)
    
    N_calculator = WindowedSum(tof_series, bin_edges)
    N = N_calculator(window)  # Shape: (num_timepoints,)
    
    # # Compute bin centers from edges
    # bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0  # Shape: (num_bins,)

    # # Compute weighted ToF series
    # weighted_tof = tof_series * window.unsqueeze(0).abs()  # Shape: (num_timepoints, num_bins)
    # # The absolute value ensures that noise contributions are non-negative
    # N = weighted_tof.sum(dim=1)  # Shape: (num_timepoints,)

    # mean_times = (weighted_tof * bin_centers.unsqueeze(0)).sum(dim=1)  # Shape: (num_timepoints,)
    # variance = (weighted_tof * (bin_centers.unsqueeze(0) - mean_times.unsqueeze(1)) ** 2).sum(
    #     dim=1
    # )  # Shape: (num_timepoints,)
    noise = variance / (N + 1e-20)  # Shape: (num_timepoints,)
    return noise


def compute_noise_variance(tof_series: torch.Tensor, bin_edges: torch.Tensor, window: torch.Tensor):
    """
    Compute the analytical noise for the second order centered moment (variance) compact statistic.

    :param tof_series: 2D tensor where each row is a ToF histogram and each column is a bin.
    :type tof_series: torch.Tensor
    :param bin_edges: 1D tensor of bin edges (length = num_bins + 1).
    :type bin_edges: torch.Tensor
    :param window: 1D tensor with same length as number of bins.
    :type window: torch.Tensor
    :return: Analytical noise value for the variance statistic.
    :rtype: torch.Tensor
    """
    # # Compute bin centers from edges
    # bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0  # Shape: (num_bins,)

    # # Compute weighted ToF series
    # weighted_tof = tof_series * window.unsqueeze(0).abs()  # Shape: (num_timepoints, num_bins)
    # # The absolute value ensures that noise contributions are non-negative
    # N = weighted_tof.sum(dim=1)  # Shape: (num_timepoints,)

    # mean_times = (weighted_tof * bin_centers.unsqueeze(0)).sum(dim=1)  # Shape: (num_timepoints,)
    # fourth_moment = (weighted_tof * (bin_centers.unsqueeze(0) - mean_times.unsqueeze(1)) ** 4).sum(
    #     dim=1
    # )  # Shape: (num_timepoints,)
    # variance = (weighted_tof * (bin_centers.unsqueeze(0) - mean_times.unsqueeze(1)) ** 2).sum(
    #     dim=1
    # )  # Shape: (num_timepoints,)
    variance_calculator = NthOrderCenteredMoment(tof_series, bin_edges, order=2)
    variance = variance_calculator(window)  # Shape: (num_timepoints,)
    
    N_calculator = WindowedSum(tof_series, bin_edges)
    N = N_calculator(window)  # Shape: (num_timepoints,)
    
    fourth_centered_moment_calculator = NthOrderCenteredMoment(tof_series, bin_edges, order=4)
    fourth_centered_moment = fourth_centered_moment_calculator(window)  # Shape: (num_timepoints,)

    noise = (fourth_centered_moment - (variance**2)) / (N)  # Shape: (num_timepoints,)
    return noise
