"""
Unit tests for the OOP versions of analytical noise calculators.
"""

import torch
import unittest
from joint_tof_opt.noise_calc import (
    get_noise_calculator,
    WindowSumNoiseCalculator,
    FirstMomentNoiseCalculator,
    VarianceNoiseCalculator,
)
from joint_tof_opt.compact_stat_process import named_moment_types
from joint_tof_opt.core import NoiseCalculator, ToFData
from joint_tof_opt.signal_process import CombSeparator


class TestGetNoiseCalculator(unittest.TestCase):
    """Test the factory function get_noise_calculator."""

    def setUp(self):
        """Create sample ToFData for testing."""
        num_timepoints = 10
        num_bins = 20
        tof_series = torch.rand(num_timepoints, num_bins)
        bin_edges = torch.linspace(0, 1, num_bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        var_series = torch.rand(num_timepoints, num_bins)
        self.window = torch.ones(num_bins)
        
        self.tof_data = ToFData(
            tof_series=tof_series,
            bin_edges=bin_edges,
            bin_centers=bin_centers,
            var_series=var_series,
            meta_data=None
        )

    def test_get_noise_calculator_returns_correct_type(self):
        """Test that get_noise_calculator returns the correct calculator type for all moment types."""
        for moment_type in named_moment_types:
            with self.subTest(moment_type=moment_type):
                calculator = get_noise_calculator(moment_type)
                self.assertIsInstance(calculator, NoiseCalculator)

    def test_compute_noise_returns_tensor(self):
        """Test that compute_noise returns a tensor for all moment types."""
        for moment_type in named_moment_types:
            with self.subTest(moment_type=moment_type):
                calculator = get_noise_calculator(moment_type)
                noise = calculator.compute_noise(self.tof_data, self.window)
                self.assertIsInstance(noise, torch.Tensor)

    def test_compute_noise_output_shape(self):
        """Test that compute_noise returns a 1D tensor with correct length for all moment types."""
        num_timepoints = self.tof_data.tof_series.shape[0]
        for moment_type in named_moment_types:
            with self.subTest(moment_type=moment_type):
                calculator = get_noise_calculator(moment_type)
                noise = calculator.compute_noise(self.tof_data, self.window)
                self.assertEqual(noise.shape, (num_timepoints,))

    def test_compute_noise_runs_without_error(self):
        """Test that compute_noise runs without errors for all moment types."""
        for moment_type in named_moment_types:
            with self.subTest(moment_type=moment_type):
                calculator = get_noise_calculator(moment_type)
                noise = calculator.compute_noise(self.tof_data, self.window)
                self.assertIsNotNone(noise)

    def test_get_noise_calculator_invalid_type_raises_error(self):
        """Test that get_noise_calculator raises error for invalid moment type."""
        with self.assertRaises(ValueError):
            get_noise_calculator("invalid_type")


if __name__ == "__main__":
    unittest.main()


