"""
Code that processes time-of-flight (TOF) data for joint optimization tasks.
"""

from typing import Any
import numpy as np


def compute_arrival_times(
    partialpath_table: np.ndarray, light_speed: list[float]
) -> np.ndarray:
    """
    Computes arrival times based on partial path lengths and light speeds in different mediums

    :param partialpath_table: 2D array, each row corresponds to a photon and each column to the path
    within medium (in mm).
    :type partialpath_table: np.ndarray
    :param light_speed: List of light speeds corresponding to each medium (in m/s).
    :type light_speed: list[float]
    :return: 1D array of arrival times for each photon path.
    """
    assert partialpath_table.shape[1] == len(
        light_speed
    ), "Number of media must match length of light_speed list."
    num_photons, num_media = partialpath_table.shape
    arrival_times = np.zeros(num_photons)

    for i in range(num_media):
        path_lengths = partialpath_table[:, i]  # in mm
        speed = light_speed[i]  # in m/s
        # Convert path length from mm to m and compute time in seconds
        times = (path_lengths * 1e-3) / speed
        arrival_times += times
    return arrival_times


def compute_weighted_intensity(
    partialpath_table: np.ndarray, tissue_model: Any
) -> np.ndarray:
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
    assert (
        len(prop_array) == num_media
    ), "Number of media must match tissue model properties len - 1"
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
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes a weighted time-of-flight (ToF) histogram for photon paths.

    This function calculates arrival times and weights for each photon, then creates
    a histogram. The time range can be determined either by weight_threshold_fraction
    or by explicitly providing time_limits. Exactly one of these must be provided.

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
    :rtype: tuple[np.ndarray, np.ndarray]
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
            assert 0 < weight_threshold_fraction <= 1.0, (
                "weight_threshold_fraction must be between 0 and 1"
            )
        
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
        filtered_times,
        bins=num_bins,
        range=(time_min, time_max),
        weights=filtered_weights
    )
    return histogram, bin_edges
