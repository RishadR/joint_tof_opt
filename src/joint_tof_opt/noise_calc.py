"""
Analytical noise calculation for different compact statistics. The noise is always expressed as noise variance
(sigma^2)
Taken from: https://doi.org/10.1117/1.JBO.17.5.057005
"""

from collections.abc import Callable

import torch

from joint_tof_opt.compact_stat_process import NthOrderCenteredMoment, WindowedSum
from joint_tof_opt.core import NoiseCalculator, ToFData, ToFModifier

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


class WindowSumWithAdditiveGaussianNoiseCalculator(NoiseCalculator):
    """
    OOP wrapper for computing analytical noise for the windowed sum compact statistic, including an additional constant
    instrument noise variance.
    """

    def __init__(self, noise_var: float):
        """
        Initialize the noise calculator with a specified instrument noise variance.

        :param noise_var: The constant variance of the instrument noise additive to each TOF bin individually
        """
        self.noise_var = noise_var

    def compute_noise(self, tof_data: ToFData, window: torch.Tensor) -> torch.Tensor:
        # Compute the weighted sum of the ToF series with the window
        weighted_tof = tof_data.tof_series * window.unsqueeze(0).abs()  # Shape: (num_timepoints, num_bins)
        signal_dependent_noise = weighted_tof.sum(dim=1)  # Shape: (num_timepoints,)
        instrument_noise = (self.noise_var * window.square()).sum()  # Shape: scalar
        total_noise = signal_dependent_noise + instrument_noise  # Shape: (num_timepoints,)
        return total_noise

    def __str__(self) -> str:
        return f"WindowSumWithInstrumentNoiseCalculator(noise_var={self.noise_var})"


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


class AdditiveGaussianToFModifier(ToFModifier):
    """
    ToFModifier that adds Gaussian noise with a specified variance to each bin within the ToF series.
    Using this to emulate instrument noise
    """

    def __init__(self, noise_var: float):
        self.noise_var = noise_var

    def modify(self, tof_data: ToFData) -> ToFData:
        noise = (torch.randn_like(tof_data.tof_series) - 0.5) * torch.sqrt(torch.tensor(self.noise_var))
        modified_tof_series = tof_data.tof_series + noise
        modified_tof_series = torch.clamp(modified_tof_series, min=0.0)
        # Create a perfect copy & keep OG intact (Perhaps create a copy method in ToFData class later?)
        if tof_data.meta_data is not None:
            meta_data = tof_data.meta_data.copy()
        else:
            meta_data = None
        return ToFData(
            tof_series=modified_tof_series,
            bin_edges=tof_data.bin_edges,
            bin_centers=tof_data.bin_centers,
            var_series=tof_data.var_series,
            inner_moments=tof_data.inner_moments,
            meta_data=meta_data,
        )

    def __str__(self) -> str:
        return f"AdditiveGaussianToFModifier(noise_var={self.noise_var})"
