"""
Analytical noise calculation for different compact statistics. The noise is always expressed as noise variance
(sigma^2)
Taken from: https://doi.org/10.1117/1.JBO.17.5.057005
"""

import math
from collections.abc import Callable
from dataclasses import replace

import torch
from typing_extensions import override

from joint_tof_opt.compact_stat_process import NthOrderCenteredMoment, WindowedSum
from joint_tof_opt.core import NoiseCalculator, ToFData, ToFModifier

# Type alias for backward compatibility
NoiseFunc = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


class WindowSumNoiseCalculator(NoiseCalculator):
    """
    OOP wrapper for computing analytical noise for the windowed sum compact statistic.
    """

    @override
    def compute_noise(self, tof_data: ToFData, window: torch.Tensor, sum_axis: int = 1) -> torch.Tensor:
        # Compute the weighted sum of the ToF series with the window
        weighted_tof = tof_data.tof_series * window.unsqueeze(0).abs()  # Shape: (num_timepoints, num_bins)
        # The absolute value ensures that noise contributions are non-negative
        if sum_axis == -1:
            return weighted_tof
        return weighted_tof.sum(dim=sum_axis)  # Shape: (num_timepoints,)

    @override
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
        self.noise_var: float = noise_var

    @override
    def compute_noise(self, tof_data: ToFData, window: torch.Tensor, sum_axis: int = 1) -> torch.Tensor:
        # Compute the weighted sum of the ToF series with the window
        weighted_tof = tof_data.tof_series * window.unsqueeze(0).abs()  # Shape: (num_timepoints, num_bins)
        instrument_noise_per_bin = self.noise_var * window.square()  # Shape: (num_bins,)
        total_noise = weighted_tof + instrument_noise_per_bin.unsqueeze(0)
        if sum_axis == -1:
            return total_noise  # Shape: (num_timepoints, num_bins)
        return total_noise.sum(dim=sum_axis)

    @override
    def __str__(self) -> str:
        return f"WindowSumWithInstrumentNoiseCalculator(noise_var={self.noise_var})"


class FirstMomentNoiseCalculator(NoiseCalculator):
    """
    OOP wrapper for computing analytical noise for the first order non-centered moment compact statistic.
    """

    @override
    def compute_noise(self, tof_data: ToFData, window: torch.Tensor, sum_axis: int = 1) -> torch.Tensor:
        if sum_axis == -1:
            raise NotImplementedError(
                "FirstMomentNoiseCalculator's noise is a ratio of quantities already collapsed across "
                + "bins internally (by NthOrderCenteredMoment/WindowedSum) - there is no per-bin result."
            )
        variance_calculator = NthOrderCenteredMoment(tof_data, order=2)
        variance = variance_calculator.forward(window)  # Shape: (num_timepoints,)

        N_calculator = WindowedSum(tof_data)
        N = N_calculator.forward(window)  # Shape: (num_timepoints,)
        assert torch.all(N > 0), "Weighted counts N must be positive to compute noise."
        noise = variance / (N)  # Shape: (num_timepoints,)
        return noise

    @override
    def __str__(self) -> str:
        return "FirstMomentNoiseCalculator"


class VarianceNoiseCalculator(NoiseCalculator):
    """
    OOP wrapper for computing analytical noise for the second order centered moment (variance) compact statistic.
    """

    @override
    def compute_noise(self, tof_data: ToFData, window: torch.Tensor, sum_axis: int = 1) -> torch.Tensor:
        if sum_axis == -1:
            raise NotImplementedError(
                "VarianceNoiseCalculator's noise is a ratio of quantities already collapsed across "
                + "bins internally (by NthOrderCenteredMoment/WindowedSum) - there is no per-bin result."
            )
        variance_calculator = NthOrderCenteredMoment(tof_data, order=2)
        variance = variance_calculator.forward(window)  # Shape: (num_timepoints,)

        N_calculator = WindowedSum(tof_data)
        N = N_calculator.forward(window)  # Shape: (num_timepoints,)

        fourth_centered_moment_calculator = NthOrderCenteredMoment(tof_data=tof_data, order=4)
        fourth_centered_moment = fourth_centered_moment_calculator.forward(window)  # Shape: (num_timepoints,)
        assert torch.all(N > 0), "Weighted counts N must be positive to compute noise."
        noise = (fourth_centered_moment - (variance**2)) / (N)  # Shape: (num_timepoints,)
        return noise

    @override
    def __str__(self) -> str:
        return "VarianceNoiseCalculator"


class AdditiveNoiseCalculator(NoiseCalculator):
    """
    Adds a noise value to each time series point output to its underlying NoiseCalculator
    """

    def __init__(self, additional_noise_variance: float, noise_calc: NoiseCalculator):
        self.noise_variance: float = additional_noise_variance
        self.noise_calc: NoiseCalculator = noise_calc

    @override
    def compute_noise(self, tof_data: ToFData, window: torch.Tensor, sum_axis: int = 1) -> torch.Tensor:
        baseline_noise_var = self.noise_calc.compute_noise(tof_data, window, sum_axis)
        return baseline_noise_var + self.noise_variance

    @override
    def __str__(self) -> str:
        return f"Baseline of '{self.noise_calc} with an additional noise of {self.noise_variance}"


class UnityTofModifier(ToFModifier):
    """
    ToFModifier that does nothing. Useful as a dummy replacement when a modifier is expected
    """

    @override
    def modify(self, tof_data: ToFData) -> ToFData:
        return tof_data

    @override
    def __str__(self) -> str:
        return "UnityModifier()"


class AdditiveGaussianToFModifier(ToFModifier):
    """
    ToFModifier that adds Gaussian noise with a specified variance to each bin within the ToF series.
    Using this to emulate instrument noise
    """

    def __init__(self, noise_var: float, seed: int = 42):
        self.noise_var: float = noise_var
        self.seed = seed

    @override
    def modify(self, tof_data: ToFData) -> ToFData:
        noise_std = math.sqrt(self.noise_var)
        torch.manual_seed(self.seed)   # Not sure how this interacts with threading - avoid threading for now
        noise = torch.normal(
            mean=0.0,
            std=noise_std,
            size=tuple(tof_data.tof_series.shape),
            dtype=tof_data.tof_series.dtype,
            device=tof_data.tof_series.device,
        )
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

    @override
    def reseed(self, seed: int) -> None:
        self.seed = seed

    @override
    def __str__(self) -> str:
        return f"AdditiveGaussianToFModifier(noise_var={self.noise_var})"


class ShotNoiseToFModifier(ToFModifier):
    """
    ToFModifier that emulates shot noise via Gaussian: each bin's value N is both the mean and variance for draw,
    and that draw is added on top of N.
    """

    def __init__(self, mean_multiplier: float = 1.0, seed: int = 42) -> None:
        super().__init__()
        self.mean_multiplier: float = mean_multiplier
        self.seed = seed

    @override
    def modify(self, tof_data: ToFData) -> ToFData:
        torch.manual_seed(self.seed)    # Again - same issue - not sure how multi-threading interacts - avoid threading
        noise = torch.normal(tof_data.tof_series * self.mean_multiplier, torch.sqrt(tof_data.tof_series))
        # copy metadata by value rather than by reference!
        meta_data = tof_data.meta_data.copy() if tof_data.meta_data is not None else None
        return replace(tof_data, tof_series=tof_data.tof_series + noise, meta_data=meta_data)

    @override
    def reseed(self, seed: int) -> None:
        self.seed = seed

    @override
    def __str__(self) -> str:
        return "ShotNoiseToFModifier()"


class SumToFModifier(ToFModifier):
    """
    ToFModifier that applies two ToFModifiers consecutively: `first`, then `second` on first's output.
    """

    def __init__(self, first: ToFModifier, second: ToFModifier):
        self.first: ToFModifier = first
        self.second: ToFModifier = second

    @override
    def modify(self, tof_data: ToFData) -> ToFData:
        return self.second.modify(self.first.modify(tof_data))

    @override
    def reseed(self, seed: int) -> None:
        self.first.reseed(seed)
        self.second.reseed(seed)

    @override
    def __str__(self) -> str:
        return f"SumToFModifier({self.first}, {self.second})"
