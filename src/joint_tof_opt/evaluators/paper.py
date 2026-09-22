"""
Computing Different Evaluation Metrics for Sensitivity Analysis.

- All classes here inherit from the Evaluator base class and implement the evaluate() method to compute
  the sensitivity metric for a given window.
- Evaluators are lazy-evaluated; computation happens only when evaluate() is called.
- Evaluators can either recompute DTOF from partial path data (if measurand is a string) or
use internal data (if measurand is a custom module) - in which case the DTOF computations must be done beforehand
"""

from math import sqrt
from pathlib import Path

import torch
from typing_extensions import override

from joint_tof_opt.compact_stat_process import get_named_moment_module
from joint_tof_opt.config_loader import ToFConfig
from joint_tof_opt.core import Evaluator, NoiseCalculator
from joint_tof_opt.signal_process import CombSeparator, PSAFESeparator
from joint_tof_opt.tof_batch_process import generate_tof

__all__ = [
    "PaperEvaluator",
    "AltPaperEvaluator2",
    "AltPaperEvaluator3",
]


class PaperEvaluator(Evaluator):
    """ """

    def __init__(
        self,
        ppath_file: Path,
        window: torch.Tensor,
        measurand: str,
        gen_config: ToFConfig,
        noise_calc: NoiseCalculator,
        filter_hw: float = 0.3,
    ):
        super().__init__(ppath_file, window, measurand, gen_config, noise_calc)
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
        self.baseline_noise_std = sqrt(self.noise_calc.compute_noise(tof_data, self.window).mean().item())
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
        noise_calc: NoiseCalculator,
        filter_hw: float = 0.3,
    ):
        super().__init__(ppath_file, window, measurand, gen_config, noise_calc, filter_hw)

    @override
    def __str__(self) -> str:
        return "Computes fetal AC Energy / (Baseline Noise Std * Maternal AC Amp)"

    @override
    def evaluate(self) -> float:
        baseline_tof_data = generate_tof(self.ppath_file, self.gen_config, True, True)
        self.baseline_noise_std: float = sqrt(
            self.noise_calc.compute_noise(baseline_tof_data, self.window).mean().item()
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
        noise_calc: NoiseCalculator,
        filter_hw: float = 0.3,
    ):
        super().__init__(ppath_file, window, measurand, gen_config, noise_calc, filter_hw)
        self.fetal_comb_filter: CombSeparator | PSAFESeparator = PSAFESeparator(
            gen_config.sampling_rate, gen_config.fetal_f, True
        )
        self.maternal_comb_filter: CombSeparator | PSAFESeparator = PSAFESeparator(
            gen_config.sampling_rate, gen_config.maternal_f, True
        )
