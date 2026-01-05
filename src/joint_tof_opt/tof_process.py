"""
Code that processes time-of-flight (TOF) data for joint optimization tasks. Mainly for internal usage. For
experiments, use the functions inside tof_batch_process.py - which provide higher level abstractions.
"""

from typing import Any
import numpy as np
import torch
from joint_tof_opt.core import ToFData


def compute_arrival_times(partialpath_table: np.ndarray, light_speed: list[float]) -> np.ndarray:
    """
    Computes arrival times based on partial path lengths and light speeds in different mediums

    :param partialpath_table: 2D array, each row corresponds to a photon and each column to the path
    within medium (in mm).
    :type partialpath_table: np.ndarray
    :param light_speed: List of light speeds corresponding to each medium (in m/s).
    :type light_speed: list[float]
    :return: 1D array of arrival times for each photon path.
    """
    assert partialpath_table.shape[1] == len(light_speed), "Number of media must match length of light_speed list."
    num_photons, num_media = partialpath_table.shape
    arrival_times = np.zeros(num_photons)

    for i in range(num_media):
        path_lengths = partialpath_table[:, i]  # in mm
        speed = light_speed[i]  # in m/s
        # Convert path length from mm to m and compute time in seconds
        times = (path_lengths * 1e-3) / speed
        arrival_times += times
    return arrival_times


def compute_weighted_intensity(partialpath_table: np.ndarray, tissue_model: Any) -> np.ndarray:
    """
    Computes weighted intensity for each photon path based on tissue model properties.

    :param partialpath_table: 2D array, each row corresponds to a photon and each column to the path
    within medium (in mm).
    :type partialpath_table: np.ndarray
    :param tissue_model: Tissue model object that provides a "prop" property - which contains the optical properties of
    each medium as a list of lists. The first element of each list is mu_a (in mm-1).
    Example structure:
    tissue_model.prop = [
        [0.1, 1.0, 0.9],   # Medium 1: [mu_a, mu_s, g]
        [0.2, 0.8, 0.85],  # Medium 2: [mu_a, mu_s, g]
        ... ]
    :type tissue_model: Any
    :return: 1D array of weighted intensities for each photon path.
    """
    assert hasattr(tissue_model, "prop"), "Tissue model must have a 'prop' attribute."
    _, num_media = partialpath_table.shape
    prop_array = tissue_model.prop[1:]  # Exclude background properties
    assert len(prop_array) == num_media, "Number of media must match tissue model properties len - 1"
    mu_a_array = np.array([prop[0] for prop in prop_array])  # in mm-1
    intensities = np.exp(np.sum(-partialpath_table * mu_a_array, axis=1))
    return intensities


def compute_tof_discrete(
    partialpath_table: np.ndarray,
    light_speed: list[float],
    tissue_model: Any,
    num_bins: int,
    weight_threshold_fraction: float | None = 0.99,
    time_limits: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes a weighted time-of-flight (ToF) histogram for photon paths with per-bin variance.

    This function calculates arrival times and weights for each photon, then creates
    a histogram and computes the weighted variance of arrival times within each bin.
    The time range can be determined either by weight_threshold_fraction or by explicitly
    providing time_limits. Exactly one of these must be provided.

    :param partialpath_table: 2D array, each row corresponds to a photon and each column to the path
        within medium (in mm).
    :type partialpath_table: np.ndarray
    :param light_speed: List of light speeds corresponding to each medium (in m/s).
    :type light_speed: list[float]
    :param tissue_model: Tissue model object that provides optical properties for computing weights. The class only
    needs to have a "prop" attribute as described in compute_weighted_intensity.
    :type tissue_model: Any
    :param num_bins: Number of bins for the ToF histogram.
    :type num_bins: int
    :param weight_threshold_fraction: Fraction of cumulative weight (0 to 1) used to determine
        the histogram upper limit. Default is 0.99. Must be None if time_limits is provided.
    :type weight_threshold_fraction: float | None
    :param time_limits: Tuple of (time_min, time_max) to explicitly set the histogram range.
        Must be None if weight_threshold_fraction is provided.
    :type time_limits: tuple[float, float] | None
    :return: Tuple containing:
        - 1D array representing the weighted ToF histogram.
        - 1D array representing the bin edges of the histogram.
        - 1D array representing the weighted variance of arrival times per bin.
    :rtype: tuple[np.ndarray, np.ndarray, np.ndarray]
    """
    # Validate that exactly one method is specified
    if (weight_threshold_fraction is None) == (time_limits is None):
        raise ValueError(
            "Exactly one of 'weight_threshold_fraction' or 'time_limits' must be provided (not both or neither)."
        )

    # Compute arrival times for all photons
    arrival_times = compute_arrival_times(partialpath_table, light_speed)

    # Compute weights for all photons
    weights = compute_weighted_intensity(partialpath_table, tissue_model)

    # Determine time limits based on the provided method
    if time_limits is not None:
        # Use explicitly provided time limits
        time_min, time_max = time_limits
    else:
        # Compute time limits based on weight threshold
        if weight_threshold_fraction is not None:
            assert 0 < weight_threshold_fraction <= 1.0, "weight_threshold_fraction must be between 0 and 1"

            # Sort photons by arrival time to determine the cutoff
            sorted_indices = np.argsort(arrival_times)
            sorted_times = arrival_times[sorted_indices]
            sorted_weights = weights[sorted_indices]

            # Calculate cumulative weight fraction
            cumulative_weights = np.cumsum(sorted_weights)
            total_weight = cumulative_weights[-1]
            cumulative_fraction = cumulative_weights / total_weight

            # Find the time cutoff where we reach the weight threshold
            cutoff_idx = np.searchsorted(cumulative_fraction, weight_threshold_fraction)
            if cutoff_idx >= len(sorted_times):
                cutoff_idx = len(sorted_times) - 1

            time_max = sorted_times[cutoff_idx]
            time_min = sorted_times[0]
        else:
            raise ValueError("Either weight_threshold_fraction or time_limits must be provided.")

    # Create histogram with weights, only considering photons within time limits
    mask = (arrival_times >= time_min) & (arrival_times <= time_max)
    filtered_times = arrival_times[mask]
    filtered_weights = weights[mask]

    # Generate the weighted histogram
    histogram, bin_edges = np.histogram(
        filtered_times, bins=num_bins, range=(time_min, time_max), weights=filtered_weights
    )

    # Compute weighted variance per bin
    bin_indices = np.digitize(filtered_times, bin_edges) - 1
    # Clip indices to valid range [0, num_bins-1]
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)

    variance_per_bin = np.zeros(num_bins)
    for bin_idx in range(num_bins):
        bin_mask = bin_indices == bin_idx
        if np.any(bin_mask):
            bin_times = filtered_times[bin_mask]
            bin_weights = filtered_weights[bin_mask]

            # Compute weighted mean
            total_weight_in_bin = np.sum(bin_weights)
            weighted_mean = np.sum(bin_weights * bin_times) / total_weight_in_bin

            # Compute weighted variance
            variance_per_bin[bin_idx] = np.sum(bin_weights * (bin_times - weighted_mean) ** 2) / total_weight_in_bin

    return histogram, bin_edges, variance_per_bin

def compute_inner_bin_moment(
    partialpath_table: np.ndarray,
    light_speed: list[float],
    tissue_model: Any,
    num_bins: int,
    order: float,
    time_limits: tuple[float, float],
) -> np.ndarray:
    """
    Computes the n-th order moment of arrival times within each bin of the ToF histogram.
        m_i = Σ(t^n * w) / Σ(w) for all photons in bin i

    :param partialpath_table: 2D array, each row corresponds to a photon and each column to the path
        within medium (in mm).
    :type partialpath_table: np.ndarray
    :param light_speed: List of light speeds corresponding to each medium (in m/s).
    :type light_speed: list[float]
    :param tissue_model: Tissue model object that provides optical properties for computing weights. The class only
    needs to have a "prop" attribute as described in compute_weighted_intensity.
    :type tissue_model: Any
    :param num_bins: Number of bins for the ToF histogram.
    :type num_bins: int
    :param order: Order of the moment to compute (e.g., 1 for mean, 2 for variance).
    :type order: float
    :param time_limits: Tuple of (time_min, time_max) to explicitly set the histogram range.
    :type time_limits: tuple[float, float]
    :return: 1D array of n-th order moments for each bin.
    :rtype: np.ndarray
    """
    # Compute arrival times and weights
    arrival_times = compute_arrival_times(partialpath_table, light_speed)
    weights = compute_weighted_intensity(partialpath_table, tissue_model)
    time_min, time_max = time_limits
    # Filter photons within time limits
    mask = (arrival_times >= time_min) & (arrival_times <= time_max)
    filtered_times = arrival_times[mask]
    filtered_weights = weights[mask]
    # Compute the n-th order Raw moment for each bin
    bin_edges = np.linspace(time_min, time_max, num_bins + 1)
    bin_indices = np.digitize(filtered_times, bin_edges) - 1
    # Clip indices to valid range [0, num_bins-1]
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)
    moment_per_bin = np.zeros(num_bins)
    for bin_idx in range(num_bins):
        bin_mask = bin_indices == bin_idx
        if np.any(bin_mask):
            bin_times = filtered_times[bin_mask]
            bin_weights = filtered_weights[bin_mask]
            # Compute weighted n-th order moment
            total_weight_in_bin = np.sum(bin_weights)
            moment_per_bin[bin_idx] = np.sum(bin_weights * (bin_times ** order)) / total_weight_in_bin
    return moment_per_bin
    

def compute_tof_data_single_time_point(
    partialpath_table: np.ndarray,
    light_speed: list[float],
    tissue_model: Any,
    num_bins: int,
    weight_threshold_fraction: float | None = 0.99,
    time_limits: tuple[float, float] | None = None,
    inner_moment_orders: list[float] = [],
) -> ToFData:
    """
    Computes ToFData for a single time point. An OOP wrapper around compute_tof_discrete.

    :param partialpath_table: 2D array, each row corresponds to a photon and each column to the path
        within medium (in mm).
    :type partialpath_table: np.ndarray
    :param light_speed: List of light speeds corresponding to each medium (in m/s).
    :type light_speed: list[float]
    :param tissue_model: Tissue model object that provides optical properties for computing weights. The class only
    needs to have a "prop" attribute as described in compute_weighted_intensity.
    :type tissue_model: Any
    :param num_bins: Number of bins for the ToF histogram.
    :type num_bins: int
    :param weight_threshold_fraction: Fraction of cumulative weight (0 to 1) used to determine
        the histogram upper limit. Default is 0.99. Must be None if time_limits is provided.
    :type weight_threshold_fraction: float | None
    :param time_limits: Tuple of (time_min, time_max) to explicitly set the histogram range.
        Must be None if weight_threshold_fraction is provided.
    :param inner_moment_orders: List of orders for which to compute inner moments.
    :type inner_moment_orders: list[float]
    :type time_limits: tuple[float, float] | None
    :return: ToFData object containing the computed ToF histogram, bin edges, and variance series.
    """
    tof_array, bin_edges, var_series = compute_tof_discrete(
        partialpath_table,
        light_speed,
        tissue_model,
        num_bins,
        weight_threshold_fraction,
        time_limits,
    )
    tof_tensor = torch.tensor(tof_array, dtype=torch.float32).reshape(1, -1)  # Shape: (1, num_bins)
    bin_edges_tensor = torch.tensor(bin_edges, dtype=torch.float32)
    var_series_tensor = torch.tensor(var_series, dtype=torch.float32).reshape(1, -1)  # Shape: (1, num_bins)
    inner_moments = {}
    for order in inner_moment_orders:
        moment_array = compute_inner_bin_moment(
            partialpath_table,
            light_speed,
            tissue_model,
            num_bins,
            order,
            (bin_edges[0], bin_edges[-1]),
        )
        inner_moments[order] = torch.tensor(moment_array, dtype=torch.float32).reshape(1, -1)
    
    tof_data = ToFData(
        tof_series=tof_tensor,
        bin_edges=bin_edges_tensor,
        bin_centers=(bin_edges_tensor[:-1] + bin_edges_tensor[1:]) / 2.0,
        var_series=var_series_tensor,
        inner_moments=inner_moments,
    )
    return tof_data
