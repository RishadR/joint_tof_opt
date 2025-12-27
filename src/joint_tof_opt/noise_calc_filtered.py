"""
Filtered Version of each of the noise calculation functions.
"""

"""
Analytical noise calculation for different compact statistics. The noise is always expressed as noise variance
(sigma^2)
Taken from: https://doi.org/10.1117/1.JBO.17.5.057005
"""

from typing import Callable
from joint_tof_opt.core import NoiseCalculator
from joint_tof_opt.compact_stat_process import NthOrderMoment, WindowedSum, NthOrderCenteredMoment
import torch.nn as nn
import torch


class FilteredWindowSumNoiseCalculator(NoiseCalculator):
    """
    OOP wrapper for computing analytical noise for the windowed sum compact statistic with a filter applied at the
    very end.
    
    Note: Typically assumes the input to the filter module as a 3D tensor with shape (1, 1, signal_length).
    """

    def __init__(self,filter_module: nn.Module):
        """
        Initialize the noise calculator.
        
        :param filter_module: PyTorch module that applies the desired filter to the signal.
        :type filter_module: nn.Module    
        """
        self.filter_module = filter_module

    def compute_noise(self, tof_series: torch.Tensor, bin_edges: torch.Tensor, window: torch.Tensor) -> torch.Tensor:
        # Compute the weighted sum of the ToF series with the window
        weighted_tof = tof_series * window.unsqueeze(0).abs()  # Shape: (num_timepoints, num_bins)
        # The absolute value ensures that noise contributions are non-negative
        noise = weighted_tof.sum(dim=1)  # Shape: (num_timepoints,)
        noise_reshaped = noise.reshape(1, 1, -1) # Reshape to (1, 1, signal_length) for filtering
        filtered_noise = self.filter_module(noise_reshaped)
        return filtered_noise.flatten()

    def __str__(self) -> str:
        return "FilteredWindowSumNoiseCalculator"

class FilteredFirstMomentNoiseCalculator(NoiseCalculator):
    """
    OOP wrapper for computing analytical noise for the first order non-centered moment compact statistic with a filter
    applied at the very end.
    
    Note: Typically assumes the input to the filter module as a 3D tensor with shape (1, 1, signal_length).
    """

    def __init__(self,filter_module: nn.Module):
        """
        Initialize the noise calculator.
        
        :param filter_module: PyTorch module that applies the desired filter to the signal.
        :type filter_module: nn.Module    
        """
        self.filter_module = filter_module

    def compute_noise(self, tof_series: torch.Tensor, bin_edges: torch.Tensor, window: torch.Tensor) -> torch.Tensor:
        variance_calculator = NthOrderCenteredMoment(tof_series, bin_edges, order=2)
        variance = variance_calculator.forward(window)  # Shape: (num_timepoints,)

        N_calculator = WindowedSum(tof_series, bin_edges)
        N = N_calculator.forward(window)  # Shape: (num_timepoints,)
        assert torch.all(N > 0), "Weighted counts N must be positive to compute noise."
        variance_reshaped = variance.reshape(1, 1, -1) # Reshape to (1, 1, signal_length) for filtering
        filtered_variance = self.filter_module(variance_reshaped).flatten()
        N_reshaped = N.reshape(1, 1, -1) # Reshape to (1, 1, signal_length) for filtering
        filtered_N = self.filter_module(N_reshaped).flatten()
        filtered_noise = filtered_variance / (filtered_N)  # Shape: (num_timepoints,)
        return filtered_noise.flatten()

    def __str__(self) -> str:
        return "FilteredFirstMomentNoiseCalculator"


class FilteredVarianceNoiseCalculator(NoiseCalculator):
    """
    OOP wrapper for computing analytical noise for the second order centered moment (variance) compact statistic
    with a filter applied at the very end.
    
    Note: Typically assumes the input to the filter module as a 3D tensor with shape (1, 1, signal_length).
    """

    def __init__(self,filter_module: nn.Module):
        """
        Initialize the noise calculator.
        
        :param filter_module: PyTorch module that applies the desired filter to the signal.
        :type filter_module: nn.Module    
        """
        self.filter_module = filter_module

    def compute_noise(self, tof_series: torch.Tensor, bin_edges: torch.Tensor, window: torch.Tensor) -> torch.Tensor:
        variance_calculator = NthOrderCenteredMoment(tof_series, bin_edges, order=2)
        variance = variance_calculator.forward(window)  # Shape: (num_timepoints,)

        N_calculator = WindowedSum(tof_series, bin_edges)
        N = N_calculator.forward(window)  # Shape: (num_timepoints,)

        fourth_centered_moment_calculator = NthOrderCenteredMoment(tof_series, bin_edges, order=4)
        fourth_centered_moment = fourth_centered_moment_calculator.forward(window)  # Shape: (num_timepoints,)
        assert torch.all(N > 0), "Weighted counts N must be positive to compute noise."
        
        variance_reshaped = variance.reshape(1, 1, -1) # Reshape to (1, 1, signal_length) for filtering
        filtered_variance = self.filter_module(variance_reshaped).flatten()
        N_reshaped = N.reshape(1, 1, -1) # Reshape to (1, 1, signal_length) for filtering
        filtered_N = self.filter_module(N_reshaped).flatten()
        fourth_centered_moment_reshaped = fourth_centered_moment.reshape(1, 1, -1) # Reshape to (1, 1, signal_length) for filtering
        filtered_fourth_centered_moment = self.filter_module(fourth_centered_moment_reshaped).flatten()
        
        filtered_noise = (filtered_fourth_centered_moment - (filtered_variance**2)) / (filtered_N)  # Shape: (num_timepoints,)
        return filtered_noise.flatten()