"""
Computing Different Evaluation Metrics for Sensitivity Analysis.

- All modules here inherit from the Evaluator base class and implement the evaluate() method to compute
- Modules are lazy-evaluated; computation happens only when evaluate() is called.
- Modules can either recompute DTOF from partial path data (if measurand is a string) or
use internal data (if measurand is a custom module) - in which case the DTOF computations must be done beforehand
"""

from math import sqrt
from pathlib import Path
from typing import Any

import torch
from typing_extensions import override

from joint_tof_opt import (
    CombSeparator,
    Evaluator,
    PSAFESeparator,
    ToFConfig,
    ToFData,
    WindowSumNoiseCalculator,
    WindowSumWithAdditiveGaussianNoiseCalculator,
    generate_tof,
    get_named_moment_module,
)

__all__ = [
    "PaperEvaluator",
    "AltPaperEvaluator2",
    "AltPaperEvaluator3",
]


def _compute_baseline_noise_std(window: torch.Tensor, tof_data: ToFData, gaussian_noise_var: float = 0.0) -> float:
    """
    Computes the baseline noise standard deviation assuming a windowed sum approach.
    Formula:
        std = sqrt(sum_i(w_i * N_i)) ;
    where w_i is the window value at time bin i, and N_i is the photon count at time bin i.

    :param window: The window used for the ToF Data. Should be a 1D Tensor on the same device as ToF.tof_series
    :param tof_data: The ToF data object computed using generate_tof. Should be unnormalized!
    :param gaussian_noise_var: The variance of the additive Gaussian noise. Defaults to 0.0 (aka ignored)
    :return: The baseline noise standard deviation.
    :rtype: float
    """
    if gaussian_noise_var <= 0.0:
        noise_calc = WindowSumNoiseCalculator()
    else:
        noise_calc = WindowSumWithAdditiveGaussianNoiseCalculator(gaussian_noise_var)
    baseline_noise_var = noise_calc.compute_noise(tof_data, window).mean().item()
    baseline_noise_std = sqrt(baseline_noise_var)
    return baseline_noise_std


class PaperEvaluator(Evaluator):
    """ """

    def __init__(
        self, ppath_file: Path, window: torch.Tensor, measurand: str, gen_config: ToFConfig, filter_hw: float = 0.3
    ):
        super().__init__(ppath_file, window, measurand, gen_config)
        self.measurand: str = measurand
        self.fetal_ac_energy: float = 0.0  # Reflects the (M2 - M0)^2 term
        self.maternal_ac_energy: float = 0.0  # For selectivity calculation
        self.baseline_noise_std: float = 0.0  # Reflects the sigma(M0) term
        self.maternal_ac_amp: float = 0.0  # Reflects the (M1 - M0) term
        self.filter_hw: float = filter_hw
        self.filter_len: int = gen_config.datapoint_count // 2 + 1
        self.maternal_comb_filter: CombSeparator | PSAFESeparator = CombSeparator(
            gen_config.sampling_rate,
            gen_config.maternal_f,
            2 * gen_config.maternal_f,
            half_width=filter_hw,
            filter_length=self.filter_len,
            phase_preserve=True,
        )
        self.fetal_comb_filter: CombSeparator | PSAFESeparator = CombSeparator(
            gen_config.sampling_rate,
            gen_config.fetal_f,
            2 * gen_config.fetal_f,
            half_width=filter_hw,
            filter_length=self.filter_len,
            phase_preserve=True,
        )

    @override
    def __str__(self) -> str:
        return "Computes fetal AC Energy / (Baseline Noise Std * Maternal AC Amp)"

    @override
    def evaluate(self) -> float:
        tof_data = generate_tof(self.ppath_file, self.gen_config, True, True)
        self.baseline_noise_std = _compute_baseline_noise_std(self.window, tof_data)
        moment_module = get_named_moment_module(self.measurand, tof_data)
        compact_stats = moment_module.forward(self.window)  # Shape: (num_timepoints,)
        fetal_component = self.fetal_comb_filter.forward(compact_stats.unsqueeze(0).unsqueeze(0)).squeeze()
        maternal_component = self.maternal_comb_filter.forward(compact_stats.unsqueeze(0).unsqueeze(0)).squeeze()
        # Remove DC component and apply Hamming window
        fetal_component = fetal_component - fetal_component.mean()
        maternal_component = maternal_component - maternal_component.mean()
        hamming_window = torch.hamming_window(
            len(fetal_component), dtype=fetal_component.dtype, device=fetal_component.device
        )
        fetal_component = fetal_component * hamming_window
        maternal_component = maternal_component * hamming_window

        self.fetal_ac_energy = float(torch.sum(fetal_component**2).item())
        self.maternal_ac_energy = float(torch.sum(maternal_component**2).item())
        self.maternal_ac_amp = sqrt(self.maternal_ac_energy)
        self.final_metric: float = self.fetal_ac_energy / (self.baseline_noise_std * self.maternal_ac_amp)
        return self.final_metric

    @override
    def get_log(self) -> dict[str, float]:
        return {
            "fetal_ac_energy": self.fetal_ac_energy,
            "baseline_noise_std": self.baseline_noise_std,
            "maternal_ac_amp": self.maternal_ac_amp,
            "final_metric": self.final_metric,
            "selectivity": (self.fetal_ac_energy / self.maternal_ac_energy) ** (1 / 2),
            "fetal_snr": (self.fetal_ac_energy) ** (1 / 2) / self.baseline_noise_std,
        }


class AltPaperEvaluator2(PaperEvaluator):
    """
    An alternate version of PaperEvaluator that actually generates two time series rather than a 2 points. One series
    contains pure maternal and the other contains pure fetal pulsation. Both are passed through a CombFilter to filter
    out the respective AC components.

    """

    def __init__(
        self,
        ppath_file: Path,
        window: torch.Tensor,
        measurand: str,
        gen_config: ToFConfig,
        filter_hw: float = 0.3,
        gaussian_noise_var: float = 0.0,
    ):
        super().__init__(ppath_file, window, measurand, gen_config, filter_hw)
        self.gaussian_noise_var: float = gaussian_noise_var

    @override
    def __str__(self) -> str:
        return "Computes fetal AC Energy / (Baseline Noise Std * Maternal AC Amp)"

    @override
    def evaluate(self) -> float:
        baseline_tof_data = generate_tof(self.ppath_file, self.gen_config, True, True)
        self.baseline_noise_std: float = _compute_baseline_noise_std(
            self.window, baseline_tof_data, self.gaussian_noise_var
        )

        only_maternal_tof_data = generate_tof(self.ppath_file, self.gen_config, True, False)
        only_fetal_tof_data = generate_tof(self.ppath_file, self.gen_config, False, True)
        pure_maternal_measurand = get_named_moment_module(self.measurand, only_maternal_tof_data).forward(self.window)
        pure_fetal_measurand = get_named_moment_module(self.measurand, only_fetal_tof_data).forward(self.window)
        pure_maternal_measurand = pure_maternal_measurand - pure_maternal_measurand.mean()
        pure_maternal_measurand = self.maternal_comb_filter.forward(pure_maternal_measurand.unsqueeze(0).unsqueeze(0))
        pure_fetal_measurand = pure_fetal_measurand - pure_fetal_measurand.mean()
        pure_fetal_measurand = self.fetal_comb_filter.forward(pure_fetal_measurand.unsqueeze(0).unsqueeze(0))
        hamming_window = torch.hamming_window(
            len(pure_fetal_measurand.squeeze()), dtype=pure_fetal_measurand.dtype, device=pure_fetal_measurand.device
        )
        pure_maternal_measurand = pure_maternal_measurand.squeeze() * hamming_window
        pure_fetal_measurand = pure_fetal_measurand.squeeze() * hamming_window

        self.maternal_ac_energy: float = pure_maternal_measurand.square().sum().item()
        self.fetal_ac_energy: float = pure_fetal_measurand.square().sum().item()
        self.maternal_ac_amp: float = sqrt(self.maternal_ac_energy)
        self.final_metric: float = self.fetal_ac_energy / (self.baseline_noise_std * self.maternal_ac_amp)
        return self.final_metric

    @override
    def get_log(self) -> dict[str, float]:
        return {
            "final_metric": self.final_metric,
            "fetal_ac_energy": self.fetal_ac_energy,
            "maternal_ac_energy": self.maternal_ac_energy,
            "baseline_noise_std": self.baseline_noise_std,
            "maternal_ac_amp": self.maternal_ac_amp,
        }


class AltPaperEvaluator3(AltPaperEvaluator2):
    """
    Another version of AltPaperEvaluator2 that uses the PSAFE filter instead of the bandpass
    """

    def __init__(
        self,
        ppath_file: Path,
        window: torch.Tensor,
        measurand: str,
        gen_config: ToFConfig,
        filter_hw: float = 0.3,
        gaussian_noise_var: float = 0.0,
    ):
        super().__init__(ppath_file, window, measurand, gen_config, filter_hw, gaussian_noise_var)
        self.fetal_comb_filter: CombSeparator | PSAFESeparator = PSAFESeparator(
            gen_config.sampling_rate, gen_config.fetal_f, True
        )
        self.maternal_comb_filter: CombSeparator | PSAFESeparator = PSAFESeparator(
            gen_config.sampling_rate, gen_config.maternal_f, True
        )
