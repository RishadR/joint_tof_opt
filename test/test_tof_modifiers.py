"""
Unit tests for the Poisson and Sum ToFModifiers.
"""

import unittest

import torch

from joint_tof_opt.core import ToFData
from joint_tof_opt.noise_calc import (
    AdditiveGaussianToFModifier,
    ShotNoiseToFModifier,
    SumToFModifier,
    UnityTofModifier,
)


def _make_tof_data(rate: float = 50.0) -> ToFData:
    num_timepoints, num_bins = 200, 10
    bin_edges = torch.linspace(0, 1, num_bins + 1)
    return ToFData(
        tof_series=torch.full((num_timepoints, num_bins), rate),
        bin_edges=bin_edges,
        bin_centers=0.5 * (bin_edges[:-1] + bin_edges[1:]),
        var_series=torch.zeros(num_timepoints, num_bins),
        meta_data={"sampling_rate": 15.0},
    )


class TestShotNoiseToFModifier(unittest.TestCase):
    def test_poisson_draw_is_added_to_original(self):
        torch.manual_seed(0)
        tof_data = _make_tof_data(rate=50.0)
        modified = ShotNoiseToFModifier().modify(tof_data)
        noise = modified.tof_series - tof_data.tof_series
        self.assertAlmostEqual(noise.mean().item(), 50.0, delta=1.0)  # Poisson(N) has mean N
        self.assertAlmostEqual(noise.var().item(), 50.0, delta=5.0)  # ...and variance N

    def test_original_is_untouched(self):
        tof_data = _make_tof_data()
        before = tof_data.tof_series.clone()
        modified = ShotNoiseToFModifier().modify(tof_data)
        self.assertTrue(torch.equal(tof_data.tof_series, before))
        self.assertIsNot(modified.meta_data, tof_data.meta_data)


class TestSumToFModifier(unittest.TestCase):
    def test_applies_first_then_second(self):
        calls: list[str] = []

        class Recorder(UnityTofModifier):
            def __init__(self, name: str):
                self.name = name

            def modify(self, tof_data: ToFData) -> ToFData:
                calls.append(self.name)
                return tof_data

        SumToFModifier(Recorder("first"), Recorder("second")).modify(_make_tof_data())
        self.assertEqual(calls, ["first", "second"])

    def test_composes_real_modifiers(self):
        torch.manual_seed(0)
        tof_data = _make_tof_data()
        combined = SumToFModifier(ShotNoiseToFModifier(), AdditiveGaussianToFModifier(noise_var=1.0))
        modified = combined.modify(tof_data)
        self.assertEqual(modified.tof_series.shape, tof_data.tof_series.shape)
        self.assertFalse(torch.equal(modified.tof_series, tof_data.tof_series))
        self.assertIn("ShotNoiseToFModifier", str(combined))
        self.assertIn("AdditiveGaussianToFModifier", str(combined))


if __name__ == "__main__":
    unittest.main()
