"""
Unit tests for TOF processing functions.
"""

import numpy as np
import pytest
from unittest.mock import Mock
from joint_tof_opt.tof_process import (
    compute_arrival_times,
    compute_weighted_intensity,
    compute_tof_discrete,
)


class TestComputeArrivalTimes:
    """Tests for compute_arrival_times function."""

    def test_single_medium(self):
        """Test arrival time calculation with a single medium."""
        # 3 photons, 1 medium, path lengths in mm
        partialpath_table = np.array([[100.0], [200.0], [300.0]])
        light_speed = [3e8]  # m/s (speed of light in vacuum)
        
        arrival_times = compute_arrival_times(partialpath_table, light_speed)
        
        # Expected: (100mm * 1e-3 m/mm) / 3e8 m/s = 3.333e-10 s
        expected = np.array([100e-3 / 3e8, 200e-3 / 3e8, 300e-3 / 3e8])
        np.testing.assert_array_almost_equal(arrival_times, expected)

    def test_multiple_media(self):
        """Test arrival time calculation with multiple media."""
        # 2 photons, 3 media
        partialpath_table = np.array([
            [100.0, 200.0, 150.0],  # Photon 1
            [50.0, 100.0, 75.0],    # Photon 2
        ])
        light_speed = [3e8, 2e8, 2.5e8]  # Different speeds in each medium
        
        arrival_times = compute_arrival_times(partialpath_table, light_speed)
        
        # Photon 1: 100e-3/3e8 + 200e-3/2e8 + 150e-3/2.5e8
        expected_1 = 100e-3 / 3e8 + 200e-3 / 2e8 + 150e-3 / 2.5e8
        # Photon 2: 50e-3/3e8 + 100e-3/2e8 + 75e-3/2.5e8
        expected_2 = 50e-3 / 3e8 + 100e-3 / 2e8 + 75e-3 / 2.5e8
        expected = np.array([expected_1, expected_2])
        
        np.testing.assert_array_almost_equal(arrival_times, expected)

    def test_zero_path_length(self):
        """Test with zero path length in one medium."""
        partialpath_table = np.array([[0.0, 100.0], [50.0, 0.0]])
        light_speed = [3e8, 2e8]
        
        arrival_times = compute_arrival_times(partialpath_table, light_speed)
        
        expected = np.array([100e-3 / 2e8, 50e-3 / 3e8])
        np.testing.assert_array_almost_equal(arrival_times, expected)

    def test_mismatched_dimensions(self):
        """Test that assertion fails when dimensions don't match."""
        partialpath_table = np.array([[100.0, 200.0]])
        light_speed = [3e8]  # Only 1 speed, but 2 media in table
        
        with pytest.raises(AssertionError):
            compute_arrival_times(partialpath_table, light_speed)


class TestComputeWeightedIntensity:
    """Tests for compute_weighted_intensity function."""

    def test_single_medium(self):
        """Test intensity calculation with a single medium."""
        partialpath_table = np.array([[100.0], [200.0], [50.0]])
        
        # Mock tissue model
        tissue_model = Mock()
        tissue_model.prop = [
            [0.0, 0.0, 0.0],  # Background (ignored)
            [0.01, 1.0, 0.9],  # Medium 1: [mu_a, mu_s, g]
        ]
        
        intensities = compute_weighted_intensity(partialpath_table, tissue_model)
        
        # Expected: exp(-partialpath * mu_a)
        expected = np.exp(-partialpath_table[:, 0] * 0.01)
        np.testing.assert_array_almost_equal(intensities, expected)

    def test_multiple_media(self):
        """Test intensity calculation with multiple media."""
        partialpath_table = np.array([
            [100.0, 200.0, 50.0],
            [50.0, 100.0, 25.0],
        ])
        
        tissue_model = Mock()
        tissue_model.prop = [
            [0.0, 0.0, 0.0],  # Background
            [0.01, 1.0, 0.9],  # Medium 1: [mu_a, mu_s, g]
            [0.02, 0.8, 0.85], # Medium 2: [mu_a, mu_s, g]
            [0.015, 0.9, 0.88], # Medium 3: [mu_a, mu_s, g]
        ]
        
        intensities = compute_weighted_intensity(partialpath_table, tissue_model)
        
        # Expected: exp(-(path1*mu_a1 + path2*mu_a2 + path3*mu_a3))
        mu_a = np.array([0.01, 0.02, 0.015])
        expected = np.exp(-np.sum(partialpath_table * mu_a, axis=1))
        np.testing.assert_array_almost_equal(intensities, expected)

    def test_zero_absorption(self):
        """Test with zero absorption (all weights should be 1)."""
        partialpath_table = np.array([[100.0], [200.0]])
        
        tissue_model = Mock()
        tissue_model.prop = [
            [0.0, 0.0, 0.0],  # Background
            [0.0, 1.0, 0.9],  # No absorption: [mu_a, mu_s, g]
        ]
        
        intensities = compute_weighted_intensity(partialpath_table, tissue_model)
        
        expected = np.ones(2)
        np.testing.assert_array_almost_equal(intensities, expected)

    def test_missing_prop_attribute(self):
        """Test that assertion fails when tissue model lacks 'prop' attribute."""
        partialpath_table = np.array([[100.0]])
        tissue_model = Mock(spec=[])  # No attributes
        
        with pytest.raises(AssertionError):
            compute_weighted_intensity(partialpath_table, tissue_model)

    def test_mismatched_media_count(self):
        """Test that assertion fails when media count doesn't match."""
        partialpath_table = np.array([[100.0, 200.0]])
        
        tissue_model = Mock()
        tissue_model.prop = [
            [0.0, 0.0, 0.0],  # Background
            [0.01, 1.0, 0.9],  # Only 1 medium (need 2)
        ]
        
        with pytest.raises(AssertionError):
            compute_weighted_intensity(partialpath_table, tissue_model)


class TestComputeTofDiscrete:
    """Tests for compute_tof_discrete function."""

    def test_basic_histogram(self):
        """Test basic histogram generation."""
        # Create simple test data: 4 photons with known arrival times
        partialpath_table = np.array([
            [100.0],  # Fastest
            [200.0],
            [300.0],
            [400.0],  # Slowest
        ])
        light_speed = [1e8]  # Simplified for easy calculation
        
        tissue_model = Mock()
        tissue_model.prop = [
            [0.0, 0.0, 0.0],  # Background
            [0.01, 1.0, 0.9],  # Medium 1: [mu_a, mu_s, g]
        ]
        
        histogram, _ = compute_tof_discrete(
            partialpath_table, light_speed, tissue_model, num_bins=4, weight_threshold_fraction=1.0
        )
        
        # Should have 4 bins with one photon each (weighted by intensity)
        assert histogram.shape == (4,)
        assert np.sum(histogram) > 0  # Non-zero total weight

    def test_weight_threshold_cutoff(self):
        """Test that weight threshold properly cuts off late arrivals."""
        # Create photons with exponentially decreasing weights (due to longer paths)
        partialpath_table = np.array([
            [10.0],   # High weight
            [20.0],   # Medium weight
            [30.0],   # Lower weight
            [1000.0], # Very low weight (should be cut off)
        ])
        light_speed = [1e8]
        
        tissue_model = Mock()
        tissue_model.prop = [
            [0.0, 0.0, 0.0],  # Background
            [0.1, 1.0, 0.9],  # High absorption: [mu_a, mu_s, g]
        ]
        
        # With 99% threshold, the last photon with very low weight should be excluded
        histogram, _ = compute_tof_discrete(
            partialpath_table, light_speed, tissue_model, num_bins=10, weight_threshold_fraction=0.99
        )
        
        assert histogram.shape == (10,)
        assert np.sum(histogram) > 0

    def test_all_photons_same_time(self):
        """Test when all photons arrive at the same time."""
        partialpath_table = np.array([[100.0], [100.0], [100.0]])
        light_speed = [3e8]
        
        tissue_model = Mock()
        tissue_model.prop = [
            [0.0, 0.0, 0.0],  # Background
            [0.01, 1.0, 0.9],  # Medium 1: [mu_a, mu_s, g]
        ]
        
        histogram, _ = compute_tof_discrete(
            partialpath_table, light_speed, tissue_model, num_bins=5
        )
        
        assert histogram.shape == (5,)
        # All weight should be in one bin
        assert np.sum(histogram > 0) >= 1

    def test_invalid_threshold_fraction(self):
        """Test that invalid threshold fraction raises assertion error."""
        partialpath_table = np.array([[100.0]])
        light_speed = [3e8]
        tissue_model = Mock()
        tissue_model.prop = [
            [0.0, 0.0, 0.0],  # Background
            [0.01, 1.0, 0.9],  # Medium 1: [mu_a, mu_s, g]
        ]
        
        # Test threshold > 1
        with pytest.raises(AssertionError):
            compute_tof_discrete(
                partialpath_table, light_speed, tissue_model, num_bins=10, weight_threshold_fraction=1.5
            )
        
        # Test threshold <= 0
        with pytest.raises(AssertionError):
            compute_tof_discrete(
                partialpath_table, light_speed, tissue_model, num_bins=10, weight_threshold_fraction=0.0
            )

    def test_multiple_media_integration(self):
        """Test histogram with multiple media (integration test)."""
        partialpath_table = np.array([
            [50.0, 100.0],
            [100.0, 50.0],
            [75.0, 75.0],
        ])
        light_speed = [3e8, 2e8]
        
        tissue_model = Mock()
        tissue_model.prop = [
            [0.0, 0.0, 0.0],  # Background
            [0.01, 1.0, 0.9],  # Medium 1: [mu_a, mu_s, g]
            [0.02, 0.8, 0.85],  # Medium 2: [mu_a, mu_s, g]
        ]
        
        histogram, _ = compute_tof_discrete(
            partialpath_table, light_speed, tissue_model, num_bins=10, weight_threshold_fraction=0.95
        )
        
        assert histogram.shape == (10,)
        assert np.sum(histogram) > 0
        # Check that histogram values are non-negative
        assert np.all(histogram >= 0)

    def test_histogram_preserves_weight(self):
        """Test that histogram sum approximates total weight within threshold."""
        partialpath_table = np.array([
            [100.0],
            [150.0],
            [200.0],
            [250.0],
            [300.0],
        ])
        light_speed = [3e8]
        
        tissue_model = Mock()
        tissue_model.prop = [
            [0.0, 0.0, 0.0],  # Background
            [0.01, 1.0, 0.9],  # Medium 1: [mu_a, mu_s, g]
        ]
        
        # Compute total weight
        weights = compute_weighted_intensity(partialpath_table, tissue_model)
        total_weight = np.sum(weights)
        
        histogram, _ = compute_tof_discrete(
            partialpath_table, light_speed, tissue_model, num_bins=20, weight_threshold_fraction=1.0
        )
        
        # Histogram sum should equal total weight
        np.testing.assert_almost_equal(np.sum(histogram), total_weight)
