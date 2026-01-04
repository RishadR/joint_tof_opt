"""
Unit tests for the OOP versions of analytical noise calculators.
"""

import torch
import pytest
from joint_tof_opt.noise_calc import (
    get_noise_calculator,
    WindowSumNoiseCalculator,
    FirstMomentNoiseCalculator,
    VarianceNoiseCalculator,
)
from joint_tof_opt.compact_stat_process import named_moment_types
from joint_tof_opt.core import NoiseCalculator, ToFData
from joint_tof_opt.signal_process import CombSeparator


class TestGetNoiseCalculator:
    """Test the factory function get_noise_calculator."""

    @pytest.fixture
    def sample_tof_data(self):
        """Create sample ToFData for testing."""
        num_timepoints = 10
        num_bins = 20
        tof_series = torch.rand(num_timepoints, num_bins)
        bin_edges = torch.linspace(0, 1, num_bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        var_series = torch.rand(num_timepoints, num_bins)
        window = torch.ones(num_bins)
        
        tof_data = ToFData(
            tof_series=tof_series,
            bin_edges=bin_edges,
            bin_centers=bin_centers,
            var_series=var_series,
            meta_data=None
        )
        return tof_data, window

    @pytest.mark.parametrize("moment_type", named_moment_types)
    def test_get_noise_calculator_returns_correct_type(self, moment_type):
        """Test that get_noise_calculator returns the correct calculator type."""
        calculator = get_noise_calculator(moment_type)
        assert isinstance(calculator, NoiseCalculator)

    @pytest.mark.parametrize("moment_type", named_moment_types)
    def test_compute_noise_returns_tensor(self, moment_type, sample_tof_data):
        """Test that compute_noise returns a tensor."""
        tof_data, window = sample_tof_data
        calculator = get_noise_calculator(moment_type)
        noise = calculator.compute_noise(tof_data, window)
        assert isinstance(noise, torch.Tensor)

    @pytest.mark.parametrize("moment_type", named_moment_types)
    def test_compute_noise_output_shape(self, moment_type, sample_tof_data):
        """Test that compute_noise returns a 1D tensor with correct length."""
        tof_data, window = sample_tof_data
        num_timepoints = tof_data.tof_series.shape[0]
        calculator = get_noise_calculator(moment_type)
        noise = calculator.compute_noise(tof_data, window)
        assert noise.shape == (num_timepoints,)

    @pytest.mark.parametrize("moment_type", named_moment_types)
    def test_compute_noise_runs_without_error(self, moment_type, sample_tof_data):
        """Test that compute_noise runs without errors for all moment types."""
        tof_data, window = sample_tof_data
        calculator = get_noise_calculator(moment_type)
        # Should not raise any exceptions
        noise = calculator.compute_noise(tof_data, window)
        assert noise is not None

    def test_get_noise_calculator_invalid_type_raises_error(self):
        """Test that get_noise_calculator raises error for invalid moment type."""
        with pytest.raises(ValueError):
            get_noise_calculator("invalid_type")


