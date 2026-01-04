"""
Analytical noise calculation for different compact statistics. The noise is always expressed as noise variance
(sigma^2)
Taken from: https://doi.org/10.1117/1.JBO.17.5.057005
"""

from typing import Callable
from joint_tof_opt.core import NoiseCalculator, ToFData
from joint_tof_opt.compact_stat_process import NthOrderMoment, WindowedSum, NthOrderCenteredMoment
import torch


# Type alias for backward compatibility
NoiseFunc = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


class WindowSumNoiseCalculator(NoiseCalculator):
    """
    OOP wrapper for computing analytical noise for the windowed sum compact statistic.
    """

    def compute_noise(self, tof_data: ToFData, window: torch.Tensor) -> torch.Tensor:
        # Compute the weighted sum of the ToF series with the window
        weighted_tof = tof_data.tof_series * window.unsqueeze(0).abs()  # Shape: (num_timepoints, num_bins)
        # The absolute value ensures that noise contributions are non-negative
        noise = weighted_tof.sum(dim=1)  # Shape: (num_timepoints,)
        return noise

    def __str__(self) -> str:
        return "WindowSumNoiseCalculator"


class FirstMomentNoiseCalculator(NoiseCalculator):
    """
    OOP wrapper for computing analytical noise for the first order non-centered moment compact statistic.
    """

    def compute_noise(self, tof_data: ToFData, window: torch.Tensor) -> torch.Tensor:
        variance_calculator = NthOrderCenteredMoment(tof_data, order=2)
        variance = variance_calculator.forward(window)  # Shape: (num_timepoints,)

        N_calculator = WindowedSum(tof_data)
        N = N_calculator.forward(window)  # Shape: (num_timepoints,)
        assert torch.all(N > 0), "Weighted counts N must be positive to compute noise."
        noise = variance / (N)  # Shape: (num_timepoints,)
        return noise

    def __str__(self) -> str:
        return "FirstMomentNoiseCalculator"


class VarianceNoiseCalculator(NoiseCalculator):
    """
    OOP wrapper for computing analytical noise for the second order centered moment (variance) compact statistic.
    """

    def compute_noise(self, tof_data: ToFData, window: torch.Tensor) -> torch.Tensor:
        variance_calculator = NthOrderCenteredMoment(tof_data, order=2)
        variance = variance_calculator.forward(window)  # Shape: (num_timepoints,)

        N_calculator = WindowedSum(tof_data)
        N = N_calculator.forward(window)  # Shape: (num_timepoints,)

        fourth_centered_moment_calculator = NthOrderCenteredMoment(tof_data=tof_data, order=4)
        fourth_centered_moment = fourth_centered_moment_calculator.forward(window)  # Shape: (num_timepoints,)
        assert torch.all(N > 0), "Weighted counts N must be positive to compute noise."
        noise = (fourth_centered_moment - (variance**2)) / (N)  # Shape: (num_timepoints,)
        return noise

    def __str__(self) -> str:
        return "VarianceNoiseCalculator"


def get_noise_calculator(moment_type: str) -> NoiseCalculator:
    """
    Factory function to get the appropriate noise calculator for a given moment type.

    :param moment_type: The type of moment ("abs", "m1", or "V")
    :param tof_series_tensor: 2D tensor of TOF series data
    :param bin_edges_tensor: 1D tensor of bin edges
    :return: An instance of the appropriate NoiseCalculator subclass
    """
    if moment_type == "abs":
        return WindowSumNoiseCalculator()
    elif moment_type == "m1":
        return FirstMomentNoiseCalculator()
    elif moment_type == "V":
        return VarianceNoiseCalculator()
    else:
        raise ValueError(f"Invalid moment type: {moment_type}")
