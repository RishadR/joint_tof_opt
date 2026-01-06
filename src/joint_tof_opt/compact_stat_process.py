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


class CorrectedNthOrderMoment(CompactStatProcess):
    """
    Computes the n-th order moment with correction for discretization.
    This requires the n-th order inner momenets to be computed during ToF generation.

    Equation: M_n = Σ(f_i * w_i * m_i) / Σ(f_i * w_i)
    where m_i is the inner n-th moment for the i-th ToF bin, f_i is the frequency (histogram value), and w_i is
    the window value.

    Example Code:
    ----------------------------------------------------------------------------------
    from joint_tof_opt.tof_batch_process import compute_tof_data_series
    n = 2  # Example for 2nd order moment
    tof_data = compute_tof_data_series(ppath_file, gen_config, inner_moment_orders=[n])
    moment_module = CorrectedNthOrderMoment(tof_data, order=n)
    moment_values = moment_module(window)
    """

    def __init__(self, tof_data: ToFData, order: float):
        super().__init__(tof_data)
        self.num_tofs, self.num_bins = tof_data.tof_series.shape
        self.order = order
        assert tof_data.inner_moments[order] is not None, f"Inner moment of order {order} not found in ToFData."
        self.inner_moments = tof_data.inner_moments[order]

    def forward(self, window: torch.Tensor) -> torch.Tensor:
        """
        Apply window and compute corrected n-th order moment for each ToF.

        :param window: 1D tensor with same length as number of bins.
        :type window: torch.Tensor
        :return: 1D tensor of corrected n-th order moments for each ToF (flattened).
        :rtype: torch.Tensor
        """
        # Point-wise multiply with window
        windowed_histograms = self.tof_series * window
        normalizer = torch.sum(windowed_histograms, dim=1)

        # Compute corrected n-th moment: Σ(f_i * w_i * m_i) / Σ(f_i * w_i)
        corrected_moment = (windowed_histograms * self.inner_moments).sum(dim=1) / normalizer
        return corrected_moment.flatten()


class CorrectedVarianceMoment(CompactStatProcess):
    """
    Computes the variance using the Total Variance Theorem to correct for discretization errors.
    Equation: Var(X) = E[Var(X|Y)] + Var(E[X|Y])
    where Y represents the ToF bins, E[X|Y] is the first moment, and Var(X|Y) is the inner variance.
    """

    def __init__(self, tof_data: ToFData):
        super().__init__(tof_data)
        self.num_tofs, self.num_bins = tof_data.tof_series.shape
        assert tof_data.inner_moments[1.0] is not None, "Inner moment of order 1 not found in ToFData."
        assert tof_data.inner_moments[2.0] is not None, "Inner moment of order 2 not found in ToFData."
        self.inner_mean = tof_data.inner_moments[1.0]
        self.inner_second_moment = tof_data.inner_moments[2.0]

    def forward(self, window: torch.Tensor) -> torch.Tensor:
        inner_variance = self.inner_second_moment - self.inner_mean**2
        windowed_histograms = self.tof_series * window  # f_i * w_i
        normalizer = torch.sum(windowed_histograms, dim=1)  # Σ(f_i * w_i)
        mean_time = (windowed_histograms * self.inner_mean).sum(dim=1) / normalizer  # E[X|Y]
        inter_bin_variace = window * (self.inner_mean - mean_time) ** 2  # (μ_i - E[X|Y])^2
        total_variance = ((windowed_histograms * inner_variance).sum(dim=1) +inter_bin_variace.sum(dim=1)) / normalizer
        return total_variance.flatten()


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
