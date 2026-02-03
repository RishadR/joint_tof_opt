"""
Computing Different Evaluation Metrics for Sensitivity Analysis.

- All modules here inherit from the Evaluator base class and implement the evaluate() method to compute
- Modules are lazy-evaluated; computation happens only when evaluate() is called.
- Modules can either recompute DTOF from partial path data (if measurand is a string) or
use internal data (if measurand is a custom module) - in which case the DTOF computations must be done beforehand
"""

from math import sqrt
from typing import Any, Callable, Literal
import torch
import yaml
import torch.nn as nn
import numpy as np
from pathlib import Path
from joint_tof_opt.tof_process import compute_tof_discrete
from tfo_sim2.tissue_model_extended import DanModel4LayerX
from joint_tof_opt import (
    CombSeparator,
    EnergyRatioMetric,
    named_moment_types,
    get_named_moment_module,
    Evaluator,
    CompactStatProcess,
    generate_tof,
    NoiseCalculator,
    get_noise_calculator,
    ToFData,
    compute_tof_data_series,
)

__all__ = [
    "PureFetalSensitivityEvaluator",
    "NormalizedPureFetalSensitivityEvaluator",
    "FetalSensitivityEvaluator",
    "CorrelationEvaluator",
    "SNREvaluator",
    "NormalizedFetalSNREvaluator",
    "ProductEvaluator",
    "NormalizedFetalSensitivityEvaluator",
    "PaperEvaluator",
    "SpectralCorrelationEvaluator",
    "FetalSelectivityEvaluator",
    "NormalizedSNREvaluator",
]


def _metadata_check(moment_module: CompactStatProcess, required_fields: list[str]) -> None:
    assert moment_module.meta_data is not None, "Meta data must be provided in the measurand module"
    for field in required_fields:
        assert field in moment_module.meta_data, f"{field} must be in the meta data!"


class PureFetalSensitivityEvaluator(Evaluator):
    """
    Evaluator for computing fetal sensitivity w.r.t. fetal mu_a using partial path data.

    This evaluator computes how much the measurand changes per unit change in fetal mu_a
    (absorption coefficient). It does this by:
    1. Loading photon partial path data
    2. Computing TOF distributions for base and perturbed (increased fetal mu_a) models
    3. Computing the measurand for both distributions
    4. Computing sensitivity as delta measurand / delta mu_a_fetal

    The result is stored in self.final_metric as a float.

    Extra Note: If the measurand is a string, it uses the ppath & the tof_config.yaml to compute the TOF distributions.
    If the measurand is a custom nn.Module, it directly uses the measurand's internal TOF for computation - skipping
    the ppath generation step.
    """

    def __init__(
        self,
        ppath_file: Path,
        window: torch.Tensor,
        measurand: str | CompactStatProcess,
        gen_config: dict,
        delta_percnt: float = 5.0,
    ):
        """
        Initialize the FetalSensitivityEvaluator.

        :param ppath_file: Path to the ppath dataset (.npz file).
        :param window: The time-gating window to apply.
        :param measurand: The measurand to compute sensitivity for ("abs", "m1", "V") or custom module.
        :param gen_config: DTOF generation configs. This will be used on the ppath file to generate the ToF data.
        :param delta_percnt: Percentage increase in fetal mu_a for sensitivity computation. (Default: 2.5)
        """
        super().__init__(ppath_file, window, measurand, gen_config)
        self.delta_percnt = delta_percnt
        self.measurand_str = ""
        if isinstance(measurand, str):
            assert measurand in named_moment_types, f"Measurand string '{measurand}' not recognized"
            self.measurand_str = measurand
        self.delta_mu_a_fetal = 0.0
        self.delta_measurand = 0.0

    def __str__(self) -> str:
        return "Computes Fetal Sensitivity as delta measurand / delta mu_a_fetal assuming no maternal interference"

    def evaluate(self) -> float:
        """
        Compute the fetal sensitivity.

        :return: The computed sensitivity value as delta measurand / delta mu_a_fetal.
        The value could be negative. The units are (mm^-1 times units of measurand).
        """
        # Load configuration
        light_speeds = [float(speed) for speed in self.gen_config["light_speeds"]]  # in m/s for 4 layers

        # Load partial path data
        ppath_dataset = np.load(self.ppath_file)
        ppath = ppath_dataset["ppath"]  # Shape: (num_photons, num_layers + 1)
        bin_count = self.gen_config["bin_count"]
        assert bin_count == len(self.window), "Window length must match bin count in tof_config.yaml"
        fraction = self.gen_config["weight_threshold_fraction"]
        filtered_ppath = (ppath[ppath[:, 0] == self.gen_config["selected_sdd_index"]])[
            :, 1:
        ]  # Drop the sdd index column

        # Create base and perturbed tissue models
        base_model = DanModel4LayerX(
            self.gen_config["wavelength"],
            self.gen_config["epi_thickness_mm"],
            self.gen_config["derm_thickness_mm"],
            self.gen_config["maternal_hb_base"],
            self.gen_config["maternal_saturation"],
            self.gen_config["fetal_saturation"],
            self.gen_config["fetal_hb_base"],
        )
        perturbed_model = DanModel4LayerX(
            self.gen_config["wavelength"],
            self.gen_config["epi_thickness_mm"],
            self.gen_config["derm_thickness_mm"],
            self.gen_config["maternal_hb_base"],
            self.gen_config["maternal_saturation"],
            self.gen_config["fetal_saturation"],
            self.gen_config["fetal_hb_base"] * (1 + self.delta_percnt / 100),
        )

        if isinstance(self.measurand, str):
            # Compute TOF distributions
            base_tof, bin_edges, base_var = compute_tof_discrete(
                filtered_ppath, light_speeds, base_model, bin_count, fraction, None
            )
            time_limits = (bin_edges[0], bin_edges[-1])
            perturbed_tof, _, perturbed_var = compute_tof_discrete(
                filtered_ppath,
                light_speeds,
                perturbed_model,
                bin_count,
                None,
                time_limits,
            )
            tof_dataset = np.vstack([base_tof, perturbed_tof])  # Shape: (2, bin_count)
            var_dataset = np.vstack([base_var, perturbed_var])  # Shape: (2, bin_count)

            # Compute measurand
            tof_series_tensor = torch.tensor(tof_dataset, dtype=torch.float32)
            var_dataset_tensor = torch.tensor(var_dataset, dtype=torch.float32)
            bin_edges_tensor = torch.tensor(bin_edges, dtype=torch.float32)
            bin_centers_tensor = 0.5 * (bin_edges_tensor[:-1] + bin_edges_tensor[1:])
            tof_data = ToFData(tof_series_tensor, bin_edges_tensor, bin_centers_tensor, var_dataset_tensor)
            moment_calculator = get_named_moment_module(self.measurand, tof_data)
        else:
            moment_calculator = self.measurand

        measurand_values = moment_calculator.forward(self.window)
        measurand_values = measurand_values.detach().cpu().numpy()

        # Compute sensitivity
        self.delta_mu_a_fetal = perturbed_model.prop[-1][0] - base_model.prop[-1][0]  # Change in fetal mu_a in mm-1
        self.delta_measurand = float(measurand_values[1] - measurand_values[0])
        self.final_metric = -self.delta_measurand / self.delta_mu_a_fetal
        return self.final_metric

    def get_log(self) -> dict[str, Any]:
        return {
            "fetal_sensitivity": self.final_metric,
            "delta_percnt": self.delta_percnt,
            "delta_mu_a_fetal": self.delta_mu_a_fetal,
            "delta_measurand": self.delta_measurand,
        }


class FetalSensitivityEvaluator(Evaluator):
    def __init__(
        self,
        ppath_file: Path,
        window: torch.Tensor,
        measurand: str | CompactStatProcess,
        gen_config: dict,
        filter_hw: float = 0.3,
        output_sensitivity: Literal["maternal", "fetal"] = "fetal",
    ):
        super().__init__(ppath_file, window, measurand, gen_config)
        self.filter_hw = filter_hw
        self.output_sensitivity = output_sensitivity
        self.fetal_comb_filter = None
        self.maternal_comb_filter = None
        self.moment_module = None
        self.measurand_str = ""
        if isinstance(measurand, str):
            assert measurand in named_moment_types, f"Measurand string '{measurand}' not recognized"
            self.measurand_str = measurand
        self.fetal_measurand_energy = 0.0
        self.maternal_measurand_energy = 0.0
        self.fetal_hb_energy = 0.0
        self.maternal_hb_energy = 0.0
        self.fetal_sensitivity = 0.0
        self.maternal_sensitivity = 0.0

    def __str__(self) -> str:
        return "Computes Fetal and Maternal Sensitivities as delta filtered measurand / delta hb concentration"

    def evaluate(self) -> float:
        if isinstance(self.measurand, str):
            tof_data = compute_tof_data_series(self.ppath_file, self.gen_config, True, True)
            tof_series_tensor = tof_data.tof_series
            bin_edges_tensor = tof_data.bin_edges
            self.moment_module = get_named_moment_module(self.measurand, tof_data)
            sampling_rate = self.gen_config["sampling_rate"]
            maternal_f = self.gen_config["maternal_f"]
            fetal_f = self.gen_config["fetal_f"]
            assert tof_data.meta_data is not None, "ToF Generation Failed! No MetaData for Fetal Sensitivity Evaluation"
            maternal_hb_series = tof_data.meta_data["maternal_hb_series"]
            fetal_hb_series = tof_data.meta_data["fetal_hb_series"]
        else:
            assert self.measurand.meta_data is not None, "Meta data must be provided in the measurand module"
            _metadata_check(
                self.measurand,
                [
                    "sampling_rate",
                    "maternal_f",
                    "fetal_f",
                    "maternal_hb_series",
                    "fetal_hb_series",
                ],
            )
            self.moment_module = self.measurand
            tof_series = self.measurand.tof_series
            tof_series_tensor = torch.tensor(tof_series, dtype=torch.float32)
            sampling_rate = self.measurand.meta_data["sampling_rate"]
            maternal_f = self.measurand.meta_data["maternal_f"]
            fetal_f = self.measurand.meta_data["fetal_f"]
            maternal_hb_series = self.measurand.meta_data["maternal_hb_series"]
            fetal_hb_series = self.measurand.meta_data["fetal_hb_series"]

        # Compute compact statistics
        compact_stats = self.moment_module(self.window)  # Shape: (num_timepoints,)
        compact_stats_reshaped = compact_stats.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, num_timepoints)

        # Initialize comb filters
        filter_len = tof_series_tensor.shape[1] // 2 + 1
        self.fetal_comb_filter = CombSeparator(
            fs=sampling_rate,
            f0=fetal_f,
            f1=2 * fetal_f,
            half_width=self.filter_hw,
            filter_length=filter_len,
            phase_preserve=True,
        )
        self.maternal_comb_filter = CombSeparator(
            fs=sampling_rate,
            f0=maternal_f,
            f1=2 * maternal_f,
            half_width=self.filter_hw,
            filter_length=filter_len,
            phase_preserve=True,
        )
        fetal_filtered_signal = self.fetal_comb_filter(compact_stats_reshaped)
        fetal_filtered_signal -= fetal_filtered_signal.mean()
        maternal_filtered_signal = self.maternal_comb_filter(compact_stats_reshaped)
        maternal_filtered_signal -= maternal_filtered_signal.mean()

        # Load heartbeat series

        # Remove DC component
        maternal_hb_series = maternal_hb_series - np.mean(maternal_hb_series)
        fetal_hb_series = fetal_hb_series - np.mean(fetal_hb_series)
        maternal_hb_series_tensor = torch.tensor(maternal_hb_series, dtype=torch.float32)
        fetal_hb_series_tensor = torch.tensor(fetal_hb_series, dtype=torch.float32)

        # Compute sensitivities
        self.fetal_measurand_energy = float(torch.sum(fetal_filtered_signal**2).item())
        self.fetal_hb_energy = float(torch.sum(fetal_hb_series_tensor**2).item())
        self.maternal_measurand_energy = float(torch.sum(maternal_filtered_signal**2).item())
        self.maternal_hb_energy = float(torch.sum(maternal_hb_series_tensor**2).item())
        self.fetal_sensitivity = sqrt(self.fetal_measurand_energy / self.fetal_hb_energy)
        self.maternal_sensitivity = sqrt(self.maternal_measurand_energy / self.maternal_hb_energy)
        if self.output_sensitivity == "fetal":
            self.final_metric = self.fetal_sensitivity
        else:
            self.final_metric = self.maternal_sensitivity
        return self.final_metric

    def get_log(self) -> dict[str, Any]:
        return {
            "fetal_sensitivity": self.fetal_sensitivity,
            "maternal_sensitivity": self.maternal_sensitivity,
            "fetal_measurand_energy": self.fetal_measurand_energy,
            "fetal_hb_energy": self.fetal_hb_energy,
            "maternal_measurand_energy": self.maternal_measurand_energy,
            "maternal_hb_energy": self.maternal_hb_energy,
        }


class CorrelationEvaluator(Evaluator):
    """
    Computes the correlation between the pulsating mu_a signal and the measurand signal's filtered version.

    :param ppath_file: The path to the partial path dataset (.npz file).
    :type ppath_file: Path
    :param window: The time-gating window to apply.
    :type window: torch.Tensor
    :param measurand: The measurand to compute correlation for ("abs", "m1", "V") or custom module.
    :type measurand: str | CompactStatProcess
    :param signal_type: Type of hemoglobin signal to correlate with ("fetal" or "maternal"). For fetal, we filter
    around the fetal heart rate; for maternal, we filter around the maternal heart rate.
    :type signal_type: Literal["fetal", "maternal"]
    :param filter_hw: Half-width of the filter to apply.
    :type filter_hw: float
    :param terminal_ignore_points: Number of points to ignore at the start and end of the signal when computing correlation to account for edge effects.
    :type terminal_ignore_points: int
    """

    def __init__(
        self,
        ppath_file: Path,
        window: torch.Tensor,
        measurand: str | CompactStatProcess,
        gen_config: dict,
        filter_hw: float = 0.3,
        signal_type: Literal["fetal", "maternal"] = "fetal",
        terminal_ignore_points: int = 5,
    ):

        super().__init__(ppath_file, window, measurand, gen_config)
        self.signal_type = signal_type
        self.measurand_str = ""
        if isinstance(measurand, str):
            assert measurand in named_moment_types, f"Measurand string '{measurand}' not recognized"
            self.measurand_str = measurand
        self.filter_hw = filter_hw
        self.comb_filter = None
        self.filtered_signal = None
        self.moment_module = None
        self.terminal_ignore_points = terminal_ignore_points

    def __str__(self) -> str:
        return "Computes Correlation between measurand and fetal or maternal hb changes"

    def evaluate(self) -> float:
        if isinstance(self.measurand, str):
            tof_data = compute_tof_data_series(self.ppath_file, self.gen_config, True, True)
            meta_data = tof_data.meta_data
            tof_series_tensor = tof_data.tof_series
            bin_edges_tensor = tof_data.bin_edges
            self.moment_module = get_named_moment_module(self.measurand, tof_data)
            meta_data = tof_data.meta_data
            assert meta_data is not None, "ToF Generation Failed! No MetaData for Correlation Evaluation"
            fetal_hb_series = meta_data["fetal_hb_series"]
        else:
            self.moment_module = self.measurand
            assert self.measurand.meta_data is not None, "Meta data must be provided in the measurand module"
            assert self.measurand.meta_data["fetal_hb_series"] is not None, "Fetal hb series must be in the meta data"
            tof_series = self.measurand.tof_series
            tof_series_tensor = torch.tensor(tof_series, dtype=torch.float32)
            fetal_hb_series = self.measurand.meta_data["fetal_hb_series"]
        # Compute compact statistics
        compact_stats = self.moment_module(self.window)  # Shape: (num_timepoints,)
        assert self.moment_module.meta_data is not None, "Meta data must be provided in the measurand module"
        _metadata_check(self.moment_module, ["sampling_rate", "fetal_f", "maternal_f"])
        if self.signal_type == "fetal":
            target_f = self.moment_module.meta_data["fetal_f"]
        else:
            target_f = self.moment_module.meta_data["maternal_f"]
        self.comb_filter = CombSeparator(
            fs=self.moment_module.meta_data["sampling_rate"],
            f0=target_f,
            f1=2 * target_f,
            half_width=self.filter_hw,
            filter_length=tof_series_tensor.shape[1] // 2 + 1,
            phase_preserve=True,
        )
        self.filtered_signal = self.comb_filter(compact_stats.unsqueeze(0).unsqueeze(0))  # Shape:(1, 1, num_timepoints)
        self.filtered_signal = self.filtered_signal.squeeze()  # Shape: (num_timepoints,)
        # Load heartbeat series
        fetal_hb_series = fetal_hb_series - np.mean(fetal_hb_series)
        fetal_hb_series_tensor = torch.tensor(fetal_hb_series, dtype=torch.float32)
        fetal_hb_series_tensor = fetal_hb_series_tensor[self.terminal_ignore_points : -self.terminal_ignore_points]
        temp_sig = self.filtered_signal[self.terminal_ignore_points : -self.terminal_ignore_points]

        # Compute correlation
        correlation = torch.corrcoef(torch.stack([temp_sig, fetal_hb_series_tensor]))[0, 1].item()
        self.final_metric = correlation
        return self.final_metric

    def get_log(self) -> dict[str, Any]:
        return {
            "correlation": self.final_metric,
        }


class SpectralCorrelationEvaluator(Evaluator):
    """
    Same as CorrelationEvaluator but computes correlation in frequency domain.
    """

    def __init__(
        self,
        ppath_file: Path,
        window: torch.Tensor,
        measurand: str | CompactStatProcess,
        gen_config: dict,
        filter_hw: float = 0.3,
        signal_type: Literal["fetal", "maternal"] = "fetal",
        terminal_ignore_points: int = 3,
    ):
        super().__init__(ppath_file, window, measurand, gen_config)
        self.signal_type = signal_type
        self.measurand_str = ""
        if isinstance(measurand, str):
            assert measurand in named_moment_types, f"Measurand string '{measurand}' not recognized"
            self.measurand_str = measurand
        self.filter_hw = filter_hw
        self.comb_filter = None
        self.filtered_signal = None
        self.moment_module = None
        self.terminal_ignore_points = terminal_ignore_points

    def __str__(self) -> str:
        return "Computes Spectral Correlation between measurand and fetal or maternal hb changes"

    def evaluate(self) -> float:
        if isinstance(self.measurand, str):
            tof_data = compute_tof_data_series(self.ppath_file, self.gen_config, True, True)
            meta_data = tof_data.meta_data
            tof_series_tensor = tof_data.tof_series
            bin_edges_tensor = tof_data.bin_edges
            self.moment_module = get_named_moment_module(self.measurand, tof_data)
            meta_data = tof_data.meta_data
            assert meta_data is not None, "ToF Generation Failed! No MetaData for Correlation Evaluation"
            fetal_hb_series = meta_data["fetal_hb_series"]
        else:
            self.moment_module = self.measurand
            assert self.measurand.meta_data is not None, "Meta data must be provided in the measurand module"
            assert self.measurand.meta_data["fetal_hb_series"] is not None, "Fetal hb series must be in the meta data"
            tof_series = self.measurand.tof_series
            tof_series_tensor = torch.tensor(tof_series, dtype=torch.float32)
            fetal_hb_series = self.measurand.meta_data["fetal_hb_series"]

        # Compute compact statistics
        compact_stats = self.moment_module(self.window)  # Shape: (num_timepoints,)
        compact_stats = compact_stats[self.terminal_ignore_points : -self.terminal_ignore_points]
        self.filtered_signal = torch.abs(torch.fft.rfft(compact_stats))  # Frequency domain signal
        # Load heartbeat series
        fetal_hb_series = fetal_hb_series - np.mean(fetal_hb_series)
        fetal_hb_series_tensor = torch.tensor(fetal_hb_series, dtype=torch.float32)
        fetal_hb_series_tensor = fetal_hb_series_tensor[self.terminal_ignore_points : -self.terminal_ignore_points]
        fetal_hb_series_tensor = torch.abs(torch.fft.rfft(fetal_hb_series_tensor))

        # Compute correlation
        correlation = torch.corrcoef(torch.stack([self.filtered_signal, fetal_hb_series_tensor]))[0, 1].item()
        self.final_metric = correlation
        return self.final_metric

    def get_log(self) -> dict[str, Any]:
        return {
            "correlation": self.final_metric,
        }


class SNREvaluator(Evaluator):
    """
    Evaluator for computing SNR of a given measurand with a filter applied using a (custom) noise calculator.

    :param ppath_file: Path to the ppath dataset (.npz file).
    :param window: The time-gating window to apply.
    :param measurand: The measurand to compute SNR for ("abs", "m1", "V") or custom module.
    :param filter_module: PyTorch module that applies the desired filter to the signal.
    :param noise_calc: (Optional) Custom noise calculator to use. If None, a default filtered noise calculator is
                    computed based on the measurand type.
    :param gen_config: DTOF generation configs. This will be used on the ppath file to generate the ToF data.
    """

    def __init__(
        self,
        ppath_file: Path,
        window: torch.Tensor,
        measurand: str | CompactStatProcess,
        gen_config: dict,
        noise_calc: NoiseCalculator | None = None,
        filter_module: nn.Module | None = None,
    ):
        """
        Initialize the SNR evaluator.

        :param ppath_file: Path to the ppath dataset (.npz file).
        :param window: The time-gating window to apply.
        :param measurand: The measurand to compute SNR for ("abs", "m1", "V") or custom module.
        :param filter_module: PyTorch module that applies the desired filter to the signal.
        """
        super().__init__(ppath_file, window, measurand, gen_config)
        self.filter_module = filter_module
        self.measurand_str = ""
        if isinstance(measurand, str):
            assert measurand in named_moment_types, f"Measurand string '{measurand}' not recognized"
            self.measurand_str = measurand

        # Noise Calc logic: If it is provied, use it. Otherwise create one
        self.noise_calc: NoiseCalculator  # This will never be None
        if noise_calc is not None:
            self.noise_calc = noise_calc
        else:
            # The noise calculator does not care about the filter module
            self.noise_calc = get_noise_calculator(self.measurand_str)
        self.noise_var = 0.0
        self.signal_energy = 0.0

    def __str__(self) -> str:
        return f"Computes SNR of filtered {self.measurand_str} measurand using {str(self.noise_calc)}"

    def evaluate(self) -> float:
        # Two Paths: If measurand is string, generate new data via tof_config. Otherwise use internal data
        if isinstance(self.measurand, str):
            tof_data = compute_tof_data_series(self.ppath_file, self.gen_config, True, True)
            moment_module = get_named_moment_module(self.measurand, tof_data)
        else:
            moment_module = self.measurand
            tof_data = moment_module.tof_data
        # Compute compact statistics
        compact_stats = moment_module(self.window)  # Shape: (num_timepoints,)
        noise_var = self.noise_calc.compute_noise(tof_data, self.window)  # Shape: (num_timepoints,)
        self.noise_var = noise_var.mean().item()
        assert self.noise_var > 0, "Computed noise variance must be greater than zero!"
        if self.filter_module is not None:
            compact_stats_reshaped = compact_stats.reshape(1, 1, -1)  # Reshape to (1, 1, signal_length) for filtering
            compact_stats = self.filter_module(compact_stats_reshaped).flatten()
        self.signal_energy = float(torch.sum(compact_stats**2).item())
        self.final_metric = self.signal_energy / self.noise_var
        return self.final_metric

    def get_log(self) -> dict[str, Any]:
        return {
            "snr": self.final_metric,
            "signal_energy": self.signal_energy,
            "noise_variance": self.noise_var,
        }


class NormalizedSNREvaluator(SNREvaluator):
    """
    Normalized Version of SNREvaluator where the computed SNR is always between 0 and 1.

    This is done via computing the Best SNR
    """

    def __init__(
        self,
        ppath_file: Path,
        window: torch.Tensor,
        measurand: str | CompactStatProcess,
        gen_config: dict,
        noise_calc: NoiseCalculator | None = None,
        filter_module: nn.Module | None = None,
    ):
        super().__init__(ppath_file, window, measurand, gen_config, noise_calc, filter_module)
        self.best_snr = 0.0

    def __str__(self) -> str:
        return f"Computes Normalized SNR of filtered {self.measurand_str} measurand using {str(self.noise_calc)}"

    def evaluate(self) -> float:
        raw_snr = super().evaluate()
        # Compute Best SNR - The best SNR always appears when using a unit window!
        tof_data = compute_tof_data_series(self.ppath_file, self.gen_config, True, True)
        moment_module = get_named_moment_module(self.measurand_str, tof_data)
        unit_window = torch.ones_like(self.window)
        unit_window /= unit_window.sum()
        compact_stats = moment_module(unit_window)  # Shape: (num_timepoints,)
        signal_energy = float(torch.sum(compact_stats**2).item())
        noise_var = self.noise_calc.compute_noise(tof_data, unit_window)  # Shape: (num_timepoints,)
        noise_var_mean = noise_var.mean().item()
        assert noise_var_mean > 0, "Computed noise variance must be greater than zero!"
        self.best_snr = signal_energy / noise_var_mean
        self.final_metric = raw_snr / self.best_snr
        return self.final_metric

    def get_log(self) -> dict[str, Any]:
        base_log = super().get_log()
        base_log.update({"best_snr": self.best_snr})
        return base_log


class FetalSNREvaluator(SNREvaluator):
    """
    Specialized SNREvaluator class for computing Fetal SNR with a given filter with filter_hw(in Hz)

    Note: This computes noise using FilteredNoiseCalculators internally.
    """

    def __init__(
        self, ppath_file: Path, window: torch.Tensor, measurand: str, gen_config: dict, filter_hw: float = 0.3
    ):
        sampling_rate = gen_config["sampling_rate"]
        fetal_f = gen_config["fetal_f"]
        datapoint_count = gen_config["datapoint_count"]
        filter_len = datapoint_count // 2 + 1
        fetal_comb_filter = CombSeparator(
            fs=sampling_rate,
            f0=fetal_f,
            f1=2 * fetal_f,
            half_width=filter_hw,
            filter_length=filter_len,
        )
        super().__init__(ppath_file, window, measurand, gen_config, filter_module=fetal_comb_filter)


class FetalSelectivityEvaluator(Evaluator):
    def __init__(
        self, ppath_file: Path, window: torch.Tensor, measurand: str, gen_config: dict, filter_hw: float = 0.3
    ):
        super().__init__(ppath_file, window, measurand, gen_config)
        sampling_rate = gen_config["sampling_rate"]
        fetal_f = gen_config["fetal_f"]
        maternal_f = gen_config["maternal_f"]
        datapoint_count = gen_config["datapoint_count"]
        filter_len = datapoint_count // 2 + 1
        self.fetal_comb_filter = CombSeparator(
            fs=sampling_rate,
            f0=fetal_f,
            f1=2 * fetal_f,
            half_width=filter_hw,
            filter_length=filter_len,
            phase_preserve=True,
        )
        self.maternal_comb_filter = CombSeparator(
            fs=sampling_rate,
            f0=maternal_f,
            f1=2 * maternal_f,
            half_width=filter_hw,
            filter_length=filter_len,
            phase_preserve=True,
        )
        self.fetal_energy = 0.0
        self.maternal_energy = 0.0
        self.window = window
        self.measurand = measurand
        self.ppath_file = ppath_file

    def __str__(self) -> str:
        return "Computes Fetal Selectivity as Fetal SNR / Maternal SNR"

    def evaluate(self) -> float:
        tof_data = compute_tof_data_series(self.ppath_file, self.gen_config, True, True)
        moment_module = get_named_moment_module(self.measurand, tof_data)
        # Compute compact statistics
        compact_stats = moment_module(self.window)  # Shape: (num_timepoints,)
        fetal_filtered_stats = self.fetal_comb_filter(compact_stats.unsqueeze(0).unsqueeze(0)).squeeze()
        maternal_filtered_stats = self.maternal_comb_filter(compact_stats.unsqueeze(0).unsqueeze(0)).squeeze()
        self.fetal_energy = float(torch.sum(fetal_filtered_stats**2).item())
        self.maternal_energy = float(torch.sum(maternal_filtered_stats**2).item())
        self.final_metric = self.fetal_energy / self.maternal_energy
        return self.final_metric

    def get_log(self) -> dict[str, Any]:
        return {
            "fetal_snr": self.fetal_energy,
            "maternal_snr": self.maternal_energy,
            "fetal_selectivity": self.final_metric,
        }


class PureFetalSNREvaluator(SNREvaluator):
    """
    Specialized version of SNREvaluator that assumes no maternal pulsations are present and thus no filtering is needed.
    The entire signal is Fetal Signal. Ignores internal measurand data.
    """

    def __init__(self, ppath_file: Path, window: torch.Tensor, measurand: str, gen_config: dict):
        super().__init__(ppath_file, window, measurand, gen_config, filter_module=None)

    def str(self) -> str:
        return "Computes Pure Fetal SNR when there is no maternal interference"

    def evaluate(self) -> float:
        tof_data = compute_tof_data_series(self.ppath_file, self.gen_config, pulse_maternal=False, pulse_fetal=True)
        if isinstance(self.measurand, str):
            moment_module = get_named_moment_module(self.measurand, tof_data)
        else:
            moment_module = self.measurand
            tof_data = moment_module.tof_data
        # Compute compact statistics
        compact_stats = moment_module(self.window)  # Shape: (num_timepoints,)
        noise_var = self.noise_calc.compute_noise(tof_data, self.window)  # Shape: (num_timepoints,)
        self.noise_var = noise_var.mean().item()
        assert self.noise_var > 0, "Computed noise variance must be greater than zero!"
        self.signal_energy = float(torch.sum(compact_stats**2).item())
        self.final_metric = self.signal_energy / self.noise_var
        return self.final_metric


class NormalizedFetalSNREvaluator(Evaluator):
    """
    Normalized Version of FetalSNREvaluator where the computed SNR is always between 0 and 1.

    This is done via computing the Best SNR
    """

    def __init__(
        self, ppath_file: Path, window: torch.Tensor, measurand: str, gen_config: dict, filter_hw: float = 0.3
    ):
        super().__init__(ppath_file, window, measurand, gen_config)
        self.fetal_snr_evaluator = FetalSNREvaluator(ppath_file, window, measurand, gen_config, filter_hw)
        unit_window = torch.ones_like(window)
        unit_window /= unit_window.norm(p=2)
        self.best_snr_evaluator = PureFetalSNREvaluator(ppath_file, unit_window, measurand, gen_config)
        self.actual_snr = 0.0
        self.best_snr = 0.0

    def __str__(self) -> str:
        return "Computes Normalized Fetal SNR between 0 and 1"

    def evaluate(self) -> float:
        self.actual_snr = self.fetal_snr_evaluator.evaluate()
        self.best_snr = self.best_snr_evaluator.evaluate()
        normalized_snr = self.actual_snr / self.best_snr
        self.final_metric = normalized_snr
        return self.final_metric

    def get_log(self) -> dict[str, Any]:
        actual_snr_log = self.fetal_snr_evaluator.get_log()
        best_snr_log = self.best_snr_evaluator.get_log()
        final_log = {
            "normalized_fetal_snr": self.final_metric,
        }
        for key, value in actual_snr_log.items():
            final_log[f"actual_{key}"] = value
        for key, value in best_snr_log.items():
            final_log[f"best_{key}"] = value
        return final_log


class NormalizedFetalSensitivityEvaluator(Evaluator):
    """
    Normalized Version of FetalSensitivityEvaluator where the computed Sensitivity is always between 0 and 1.

    This is done via computing the Best Sensitivity
    """

    def __init__(
        self,
        ppath_file: Path,
        window: torch.Tensor,
        measurand: str | CompactStatProcess,
        gen_config: dict,
        filter_hw: float = 0.3,
    ):
        super().__init__(ppath_file, window, measurand, gen_config)
        self.fetal_sensitivity_evaluator = FetalSensitivityEvaluator(
            ppath_file, window, measurand, gen_config, filter_hw
        )
        unit_window = torch.ones_like(window)
        unit_window /= unit_window.norm(p=2)
        self.best_sensitivity_evaluator = FetalSensitivityEvaluator(
            ppath_file, unit_window, measurand, gen_config, filter_hw
        )

    def __str__(self) -> str:
        return "Computes Normalized Fetal Sensitivity between 0 and 1"

    def evaluate(self) -> float:
        actual_sensitivity = self.fetal_sensitivity_evaluator.evaluate()
        best_sensitivity = self.best_sensitivity_evaluator.evaluate()
        normalized_sensitivity = actual_sensitivity / best_sensitivity
        self.final_metric = normalized_sensitivity
        return self.final_metric

    def get_log(self) -> dict[str, Any]:
        actual_sensitivity_log = self.fetal_sensitivity_evaluator.get_log()
        best_sensitivity_log = self.best_sensitivity_evaluator.get_log()
        final_log = {
            "normalized_fetal_sensitivity": self.final_metric,
        }
        for key, value in actual_sensitivity_log.items():
            final_log[f"actual_{key}"] = value
        for key, value in best_sensitivity_log.items():
            final_log[f"best_{key}"] = value
        return final_log


class NormalizedPureFetalSensitivityEvaluator(Evaluator):
    """
    Normalized Version of FetalSensitivityEvaluator where the computed Sensitivity is always between 0 and 1.

    This is done via computing the Best Sensitivity
    """

    def __init__(self, ppath_file: Path, window: torch.Tensor, measurand: str | CompactStatProcess, gen_config: dict):
        super().__init__(ppath_file, window, measurand, gen_config)
        self.fetal_sensitivity_evaluator = PureFetalSensitivityEvaluator(ppath_file, window, measurand, gen_config)
        unit_window = torch.ones_like(window)
        unit_window /= unit_window.norm(p=2)
        self.best_sensitivity_evaluator = PureFetalSensitivityEvaluator(ppath_file, unit_window, measurand, gen_config)

    def __str__(self) -> str:
        return "Computes Normalized Fetal Sensitivity between 0 and 1"

    def evaluate(self) -> float:
        actual_sensitivity = self.fetal_sensitivity_evaluator.evaluate()
        best_sensitivity = self.best_sensitivity_evaluator.evaluate()
        normalized_sensitivity = actual_sensitivity / best_sensitivity
        self.final_metric = normalized_sensitivity
        return self.final_metric

    def get_log(self) -> dict[str, Any]:
        actual_sensitivity_log = self.fetal_sensitivity_evaluator.get_log()
        best_sensitivity_log = self.best_sensitivity_evaluator.get_log()
        final_log = {
            "normalized_fetal_sensitivity": self.final_metric,
        }
        for key, value in actual_sensitivity_log.items():
            final_log[f"actual_{key}"] = value
        for key, value in best_sensitivity_log.items():
            final_log[f"best_{key}"] = value
        return final_log


class ProductEvaluator(Evaluator):
    """
    Evaluator that computes the product of two evaluators.

    :param evaluator1: The first evaluator.
    :param evaluator2: The second evaluator.
    """

    def __init__(self, evaluator1: Evaluator, evaluator2: Evaluator):
        super().__init__(evaluator1.ppath_file, evaluator1.window, evaluator1.measurand, evaluator1.gen_config)
        self.evaluator1 = evaluator1
        self.evaluator2 = evaluator2

    def __str__(self) -> str:
        return f"Computes Product of {str(self.evaluator1)} and {str(self.evaluator2)}"

    def evaluate(self) -> float:
        metric1 = self.evaluator1.evaluate()
        metric2 = self.evaluator2.evaluate()
        self.final_metric = metric1 * metric2
        return self.final_metric

    def get_log(self) -> dict[str, Any]:
        log1 = self.evaluator1.get_log()
        log2 = self.evaluator2.get_log()
        final_log = {
            "product_metric": self.final_metric,
        }
        for key, value in log1.items():
            final_log[f"evaluator1_{key}"] = value
        for key, value in log2.items():
            final_log[f"evaluator2_{key}"] = value
        return final_log


class PaperEvaluator(Evaluator):
    """
    The final evaluator used in the paper! Uses the following equation:
    Final Metric = Fetal Selectivity x Normalized Fetal SNR x Fetal Correlation

    where,
    Fetal Selectivity: Computed using FetalSelectivityEvaluator
    Normalized Fetal SNR: Computed using NormalizedFetalSNREvaluator
    Fetal Correlation: Computed using CorrelationEvaluator

    :param ppath_file: Path to the ppath dataset (.npz file).
    :param window: The time-gating window to apply.
    :param measurand: The measurand to compute SNR for ("abs", "m1", "V") or custom module.
    :param filter_hw: Half-width of the filter to apply.
    """

    def __init__(
        self, ppath_file: Path, window: torch.Tensor, measurand: str, gen_config: dict, filter_hw: float = 0.3
    ):
        super().__init__(ppath_file, window, measurand, gen_config)
        self.measurand = measurand  # Overwrite to keep the type a string
        self.fetal_ac_energy = 0.0  # Reflects the (M2 - M0)^2 term
        self.baseline_noise_std = 0.0  # Reflects the sigma(M0) term
        self.maternal_ac_amp = 0.0  # Reflects the (M1 - M0) term
        self.filter_hw = filter_hw
        self.filter_len = gen_config["datapoint_count"] // 2 + 1
        self.maternal_comb_filter = CombSeparator(
            gen_config["sampling_rate"],
            gen_config["maternal_f"],
            2 * gen_config["maternal_f"],
            half_width=filter_hw,
            filter_length=self.filter_len,
            phase_preserve=True,
        )
        self.fetal_comb_filter = CombSeparator(
            gen_config["sampling_rate"],
            gen_config["fetal_f"],
            2 * gen_config["fetal_f"],
            half_width=filter_hw,
            filter_length=self.filter_len,
            phase_preserve=True,
        )

    def __str__(self) -> str:
        return "Computes fetal AC Energy / (Baseline Noise Std * Maternal AC Amp)"

    def _compute_baseline_noise_std(self, tof_data: ToFData) -> float:
        """
        Computes the baseline noise standard deviation assuming a windowed sum approach.
        Formula:
            std = sqrt(sum_i(w_i * N_i)) ;
        where w_i is the window value at time bin i, and N_i is the photon count at time bin i.

        :param tof_data: The ToF data object computed using compute_tof_data_series. Should be unnormalized!
        :return: The baseline noise standard deviation.
        :rtype: float
        """
        avg_tof_frame = tof_data.tof_series.sum(dim=0, keepdim=True)  # Shape: (1, num_timepoints)
        windowed_avg_tof_frame = avg_tof_frame * self.window.unsqueeze(0)  # Shape: (1, num_timepoints)
        baseline_noise_var = windowed_avg_tof_frame.sum().item()
        baseline_noise_std = sqrt(baseline_noise_var)
        return baseline_noise_std

    def evaluate(self) -> float:
        tof_data = compute_tof_data_series(self.ppath_file, self.gen_config, True, True)
        self.baseline_noise_std = self._compute_baseline_noise_std(tof_data)
        moment_module = get_named_moment_module(self.measurand, tof_data)
        compact_stats = moment_module(self.window)  # Shape: (num_timepoints,)
        fetal_component = self.fetal_comb_filter(compact_stats.unsqueeze(0).unsqueeze(0)).squeeze()
        maternal_component = self.maternal_comb_filter(compact_stats.unsqueeze(0).unsqueeze(0)).squeeze()
        self.fetal_ac_energy = float(torch.sum(fetal_component**2).item())
        maternal_ac_energy = float(torch.sum(maternal_component**2).item())
        self.maternal_ac_amp = sqrt(maternal_ac_energy)
        self.final_metric = self.fetal_ac_energy / (self.baseline_noise_std * self.maternal_ac_amp)
        return self.final_metric

    def get_log(self) -> dict[str, Any]:
        return {
            "fetal_ac_energy": self.fetal_ac_energy,
            "baseline_noise_std": self.baseline_noise_std,
            "maternal_ac_amp": self.maternal_ac_amp,
            "final_metric": self.final_metric,
        }
