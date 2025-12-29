"""
Computing Different Evaluation Metrics for Sensitivity Analysis.

- All modules here inherit from the Evaluator base class and implement the evaluate() method to compute
- Modules are lazy-evaluated; computation happens only when evaluate() is called.
- Modules can either recompute DTOF from partial path data (if measurand is a string) or
use internal data (if measurand is a custom module) - in which case the DTOF computations must be done beforehand
"""

from math import sqrt
from typing import Callable, Literal
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
    noise_func_table,
    generate_tof,
    NoiseCalculator,
    get_noise_calculator,
)
from joint_tof_opt.noise_calc_filtered import get_filtered_noise_calculator


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
        delta_percnt: float = 2.5,
    ):
        """
        Initialize the FetalSensitivityEvaluator.

        :param ppath_file: Path to the ppath dataset (.npz file).
        :param window: The time-gating window to apply.
        :param measurand: The measurand to compute sensitivity for ("abs", "m1", "V") or custom module.
        :param delta_percnt: Percentage increase in fetal mu_a for sensitivity computation. (Default: 2.5)
        """
        super().__init__(ppath_file, window, measurand)
        self.delta_percnt = delta_percnt
        self.measurand_str = ""
        if isinstance(measurand, str):
            assert measurand in named_moment_types, f"Measurand string '{measurand}' not recognized"
            self.measurand_str = measurand

    def __str__(self) -> str:
        return "Computes Fetal Sensitivity as delta measurand / delta mu_a_fetal assuming no maternal interference"

    def evaluate(self) -> float:
        """
        Compute the fetal sensitivity.

        :return: The computed sensitivity value as delta measurand / delta mu_a_fetal.
        The value could be negative. The units are (mm^-1 times units of measurand).
        """
        # Load configuration
        with open("./experiments/tof_config.yaml", "r") as f:
            config = yaml.safe_load(f)

        light_speeds = [float(speed) for speed in config["light_speeds"]]  # in m/s for 4 layers

        # Load partial path data
        ppath_dataset = np.load(self.ppath_file)
        ppath = ppath_dataset["ppath"]  # Shape: (num_photons, num_layers + 1)
        bin_count = config["bin_count"]
        assert bin_count == len(self.window), "Window length must match bin count in tof_config.yaml"
        fraction = config["weight_threshold_fraction"]
        filtered_ppath = (ppath[ppath[:, 0] == config["selected_sdd_index"]])[:, 1:]  # Drop the sdd index column

        # Create base and perturbed tissue models
        base_model = DanModel4LayerX(
            config["wavelength"],
            config["epi_thickness_mm"],
            config["derm_thickness_mm"],
            config["maternal_hb_base"],
            config["maternal_saturation"],
            config["fetal_saturation"],
            config["fetal_hb_base"],
        )
        perturbed_model = DanModel4LayerX(
            config["wavelength"],
            config["epi_thickness_mm"],
            config["derm_thickness_mm"],
            config["maternal_hb_base"],
            config["maternal_saturation"],
            config["fetal_saturation"],
            config["fetal_hb_base"] * (1 + self.delta_percnt / 100),
        )

        if isinstance(self.measurand, str):
            # Compute TOF distributions
            base_tof, bin_edges = compute_tof_discrete(
                filtered_ppath, light_speeds, base_model, bin_count, fraction, None
            )
            time_limits = (bin_edges[0], bin_edges[-1])
            perturbed_tof, _ = compute_tof_discrete(
                filtered_ppath,
                light_speeds,
                perturbed_model,
                bin_count,
                None,
                time_limits,
            )
            tof_dataset = np.vstack([base_tof, perturbed_tof])  # Shape: (2, bin_count)

            # Compute measurand
            tof_series_tensor = torch.tensor(tof_dataset, dtype=torch.float32)
            bin_edges_tensor = torch.tensor(bin_edges, dtype=torch.float32)
            moment_calculator = get_named_moment_module(self.measurand, tof_series_tensor, bin_edges_tensor)
        else:
            moment_calculator = self.measurand

        measurand_values = moment_calculator.forward(self.window)
        measurand_values = measurand_values.detach().cpu().numpy()

        # Compute sensitivity
        delta_mu_a_fetal = perturbed_model.prop[-1][0] - base_model.prop[-1][0]  # Change in fetal mu_a in mm-1
        delta_measurand = measurand_values[1] - measurand_values[0]
        self.final_metric = delta_measurand / delta_mu_a_fetal

        return self.final_metric


class FetalSensitivityEvaluator(Evaluator):
    def __init__(
        self,
        ppath_file: Path,
        window: torch.Tensor,
        measurand: str | CompactStatProcess,
        filter_hw: float = 0.3,
        output_sensitivity: Literal["maternal", "fetal"] = "fetal",
    ):
        super().__init__(ppath_file, window, measurand)
        self.filter_hw = filter_hw
        self.output_sensitivity = output_sensitivity
        self.fetal_comb_filter = None
        self.maternal_comb_filter = None
        self.moment_module = None
        self.measurand_str = ""
        if isinstance(measurand, str):
            assert measurand in named_moment_types, f"Measurand string '{measurand}' not recognized"
            self.measurand_str = measurand

    def __str__(self) -> str:
        return "Computes Fetal and Maternal Sensitivities as delta filtered measurand / delta hb concentration"

    def evaluate(self) -> float:
        if isinstance(self.measurand, str):
            gen_config = yaml.safe_load(open("./experiments/tof_config.yaml", "r"))
            tof_temp_path = Path("./data/temp_tof_dataset.npz")
            generate_tof(self.ppath_file, gen_config, tof_temp_path)
            tof_dataset = np.load(tof_temp_path)
            tof_data = tof_dataset["tof_dataset"]
            bin_edges = tof_dataset["bin_edges"]
            tof_series_tensor = torch.tensor(tof_data, dtype=torch.float32)
            bin_edges_tensor = torch.tensor(bin_edges, dtype=torch.float32)
            self.moment_module = get_named_moment_module(self.measurand, tof_series_tensor, bin_edges_tensor)
            sampling_rate = gen_config["sampling_rate"]
            maternal_f = gen_config["maternal_f"]
            fetal_f = gen_config["fetal_f"]
            maternal_hb_series = tof_dataset["maternal_hb_series"]
            fetal_hb_series = tof_dataset["fetal_hb_series"]
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
            tof_data = self.measurand.tof_series
            sampling_rate = self.measurand.meta_data["sampling_rate"]
            maternal_f = self.measurand.meta_data["maternal_f"]
            fetal_f = self.measurand.meta_data["fetal_f"]
            maternal_hb_series = self.measurand.meta_data["maternal_hb_series"]
            fetal_hb_series = self.measurand.meta_data["fetal_hb_series"]

        # Compute compact statistics
        compact_stats = self.moment_module(self.window)  # Shape: (num_timepoints,)
        compact_stats_reshaped = compact_stats.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, num_timepoints)

        # Initialize comb filters
        filter_len = tof_data.shape[1] // 2 + 1
        self.fetal_comb_filter = CombSeparator(
            fs=sampling_rate,
            f0=fetal_f,
            f1=2 * fetal_f,
            half_width=self.filter_hw,
            filter_length=filter_len,
        )
        self.maternal_comb_filter = CombSeparator(
            fs=sampling_rate,
            f0=maternal_f,
            f1=2 * maternal_f,
            half_width=self.filter_hw,
            filter_length=filter_len,
        )
        fetal_filtered_signal = self.fetal_comb_filter(compact_stats_reshaped)
        maternal_filtered_signal = self.maternal_comb_filter(compact_stats_reshaped)

        # Load heartbeat series

        # Remove DC component
        maternal_hb_series = maternal_hb_series - np.mean(maternal_hb_series)
        fetal_hb_series = fetal_hb_series - np.mean(fetal_hb_series)
        maternal_hb_series_tensor = torch.tensor(maternal_hb_series, dtype=torch.float32)
        fetal_hb_series_tensor = torch.tensor(fetal_hb_series, dtype=torch.float32)

        # Compute sensitivities
        energy_ratio_metric = EnergyRatioMetric()
        fetal_sensitivity = torch.sqrt(energy_ratio_metric(fetal_filtered_signal, fetal_hb_series_tensor)).item()
        maternal_sensitivity = torch.sqrt(
            energy_ratio_metric(maternal_filtered_signal, maternal_hb_series_tensor)
        ).item()
        if self.output_sensitivity == "fetal":
            self.final_metric = fetal_sensitivity
        else:
            self.final_metric = maternal_sensitivity
        return self.final_metric


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
        filter_hw: float = 0.3,
        signal_type: Literal["fetal", "maternal"] = "fetal",
        terminal_ignore_points: int = 5,
    ):

        super().__init__(ppath_file, window, measurand)
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
            gen_config = yaml.safe_load(open("./experiments/tof_config.yaml", "r"))
            tof_temp_path = Path("./data/temp_tof_dataset.npz")
            generate_tof(self.ppath_file, gen_config, tof_temp_path)
            tof_dataset = np.load(tof_temp_path)
            tof_data = tof_dataset["tof_dataset"]
            bin_edges = tof_dataset["bin_edges"]
            tof_series_tensor = torch.tensor(tof_data, dtype=torch.float32)
            bin_edges_tensor = torch.tensor(bin_edges, dtype=torch.float32)
            self.moment_module = get_named_moment_module(self.measurand, tof_series_tensor, bin_edges_tensor)
            fetal_hb_series = tof_dataset["fetal_hb_series"]
        else:
            self.moment_module = self.measurand
            assert self.measurand.meta_data is not None, "Meta data must be provided in the measurand module"
            assert self.measurand.meta_data["fetal_hb_series"] is not None, "Fetal hb series must be in the meta data"
            tof_data = self.measurand.tof_series
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
            filter_length=tof_data.shape[1] // 2 + 1,
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


class CorrelationxSNREvaluator(Evaluator):
    """
    Same as the CorrelationEvaluator but multiplies the correlation with the SNR
    """

    def __init__(
        self,
        ppath_file: Path,
        window: torch.Tensor,
        measurand: str | CompactStatProcess,
        noise_func: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        filter_hw: float = 0.3,
        signal_type: Literal["fetal", "maternal"] = "fetal",
        terminal_ignore_points: int = 5,
    ):

        super().__init__(ppath_file, window, measurand)
        self.signal_type = signal_type
        self.filter_hw = filter_hw
        self.comb_filter = None
        self.moment_module = None
        self._correlation_evaluator = CorrelationEvaluator(
            ppath_file,
            window,
            measurand,
            filter_hw,
            signal_type,
            terminal_ignore_points,
        )
        if not isinstance(measurand, str):
            assert noise_func is not None, "Noise function must be provided for custom measurand modules"
            self.noise_func = noise_func
            self.measurand_str = ""
        else:
            self.noise_func = noise_func_table[measurand]
            self.measurand_str = measurand

    def __str__(self) -> str:
        return "Computes Correlation x SNR between measurand and fetal or maternal hb changes"

    def evaluate(self) -> float:
        correlation = self._correlation_evaluator.evaluate()
        self.moment_module = self._correlation_evaluator.moment_module
        self.comb_filter = self._correlation_evaluator.comb_filter
        assert self.moment_module is not None, "Internal CorrelationEvaluator did not work properly!"
        assert (
            self._correlation_evaluator.filtered_signal is not None
        ), "Internal CorrelationEvaluator did not work properly!"
        tof_series = self.moment_module.tof_series
        bin_edges = self.moment_module.bin_edges
        noise_var = self.noise_func(tof_series, bin_edges, self.window)
        signal = self._correlation_evaluator.filtered_signal
        signal_energy = torch.sum(signal**2).item()
        snr = signal_energy / (torch.sum(noise_var).item() + 1e-40)
        self.final_metric = correlation * snr
        return self.final_metric


class SNREvaluator(Evaluator):
    """
    Evaluator for computing SNR of a given measurand with a filter applied using a (custom) noise calculator.

    :param ppath_file: Path to the ppath dataset (.npz file).
    :param window: The time-gating window to apply.
    :param measurand: The measurand to compute SNR for ("abs", "m1", "V") or custom module.
    :param filter_module: PyTorch module that applies the desired filter to the signal.
    :param noise_calc: (Optional) Custom noise calculator to use. If None, a default filtered noise calculator is
                    computed based on the measurand type.
    """

    def __init__(
        self,
        ppath_file: Path,
        window: torch.Tensor,
        measurand: str | CompactStatProcess,
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
        super().__init__(ppath_file, window, measurand)
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
            if self.filter_module is not None:
                self.noise_calc = get_filtered_noise_calculator(self.measurand_str, self.filter_module)
            else:
                self.noise_calc = get_noise_calculator(self.measurand_str)

    def __str__(self) -> str:
        return f"Computes SNR of filtered {self.measurand_str} measurand using {str(self.noise_calc)}"

    def evaluate(self) -> float:
        # Two Paths: If measurand is string, generate new data via tof_config. Otherwise use internal data
        if isinstance(self.measurand, str):
            gen_config = yaml.safe_load(open("./experiments/tof_config.yaml", "r"))
            tof_temp_path = Path("./data/temp_tof_dataset.npz")
            generate_tof(self.ppath_file, gen_config, tof_temp_path)
            tof_dataset = np.load(tof_temp_path)
            tof_data = torch.tensor(tof_dataset["tof_dataset"], dtype=torch.float32)
            bin_edges = torch.tensor(tof_dataset["bin_edges"], dtype=torch.float32)
            moment_module = get_named_moment_module(self.measurand, tof_data, bin_edges)
        else:
            moment_module = self.measurand
            assert self.measurand.meta_data is not None, "Meta data must be provided in the measurand module"
            tof_data = self.measurand.tof_series
            bin_edges = self.measurand.bin_edges
        # Compute compact statistics
        compact_stats = moment_module(self.window)  # Shape: (num_timepoints,)
        noise_var = self.noise_calc.compute_noise(tof_data, bin_edges, self.window)  # Shape: (num_timepoints,)
        noise_var = noise_var.mean().item()
        assert noise_var > 0, "Computed noise variance must be greater than zero!"
        if self.filter_module is not None:
            compact_stats_reshaped = compact_stats.reshape(1, 1, -1)  # Reshape to (1, 1, signal_length) for filtering
            compact_stats = self.filter_module(compact_stats_reshaped).flatten()
        signal_energy = torch.sum(compact_stats**2).item()
        snr = signal_energy / noise_var
        self.final_metric = snr
        return self.final_metric


class FetalSNREvaluator(SNREvaluator):
    """
    Specialized SNREvaluator class for computing Fetal SNR with a given filter with filter_hw(in Hz)

    Note: This computes noise using FilteredNoiseCalculators internally.
    """

    def __init__(
        self, ppath_file: Path, window: torch.Tensor, measurand: str | CompactStatProcess, filter_hw: float = 0.3
    ):
        gen_config = yaml.safe_load(open("./experiments/tof_config.yaml", "r"))
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
        super().__init__(ppath_file, window, measurand, filter_module=fetal_comb_filter)


class PureFetalSNREvaluator(SNREvaluator):
    """
    Specialized version of SNREvaluator that assumes no maternal pulsations are present and thus no filtering is needed.
    The entire signal is Fetal Signal. Ignores internal measurand data.
    """

    def __init__(self, ppath_file: Path, window: torch.Tensor, measurand: str | CompactStatProcess):
        super().__init__(ppath_file, window, measurand, filter_module=None)

    def str(self) -> str:
        return "Computes Pure Fetal SNR when there is no maternal interference"

    def evaluate(self) -> float:
        gen_config = yaml.safe_load(open("./experiments/tof_config.yaml", "r"))
        temp_tof_path = Path("./data/temp_tof_dataset.npz")
        generate_tof(self.ppath_file, gen_config, temp_tof_path, pulse_maternal=False, pulse_fetal=True)
        tof_dataset = np.load(temp_tof_path)
        tof_data = torch.tensor(tof_dataset["tof_dataset"], dtype=torch.float32)
        bin_edges = torch.tensor(tof_dataset["bin_edges"], dtype=torch.float32)
        if isinstance(self.measurand, str):
            moment_module = get_named_moment_module(self.measurand, tof_data, bin_edges)
        else:
            moment_module = self.measurand
        # Compute compact statistics
        compact_stats = moment_module(self.window)  # Shape: (num_timepoints,)
        noise_var = self.noise_calc.compute_noise(tof_data, bin_edges, self.window)  # Shape: (num_timepoints,)
        noise_var = noise_var.mean().item()
        assert noise_var > 0, "Computed noise variance must be greater than zero!"
        signal_energy = torch.sum(compact_stats**2).item()
        snr = signal_energy / noise_var
        self.final_metric = snr
        return self.final_metric


class NormalizedFetalSNREvaluator(Evaluator):
    """
    Normalized Version of FetalSNREvaluator where the computed SNR is always between 0 and 1.

    This is done via computing the Best SNR
    """

    def __init__(
        self, ppath_file: Path, window: torch.Tensor, measurand: str | CompactStatProcess, filter_hw: float = 0.3
    ):
        super().__init__(ppath_file, window, measurand)
        self.fetal_snr_evaluator = FetalSNREvaluator(ppath_file, window, measurand, filter_hw)
        unit_window = torch.ones_like(window)
        unit_window /= unit_window.norm(p=2)
        self.best_snr_evaluator = PureFetalSNREvaluator(ppath_file, unit_window, measurand)
        worst_window = torch.zeros_like(window)  # Take the last two points to avoid weird noise issues
        worst_window[-2:] = 1 / sqrt(2)
        self.worst_snr_evaluator = PureFetalSNREvaluator(ppath_file, worst_window, measurand)

    def __str__(self) -> str:
        return "Computes Normalized Fetal SNR between 0 and 1"

    def evaluate(self) -> float:
        actual_snr = self.fetal_snr_evaluator.evaluate()
        best_snr = self.best_snr_evaluator.evaluate()
        worst_snr = self.worst_snr_evaluator.evaluate()
        normalized_snr = (actual_snr - worst_snr) / (best_snr - worst_snr)
        self.final_metric = normalized_snr
        return self.final_metric

class NormalizedFetalSensitivityEvaluator(Evaluator):
    """
    Normalized Version of FetalSensitivityEvaluator where the computed Sensitivity is always between 0 and 1.

    This is done via computing the Best Sensitivity
    """

    def __init__(
        self, ppath_file: Path, window: torch.Tensor, measurand: str | CompactStatProcess, filter_hw: float = 0.3
    ):
        super().__init__(ppath_file, window, measurand)
        self.fetal_sensitivity_evaluator = FetalSensitivityEvaluator(ppath_file, window, measurand, filter_hw)
        unit_window = torch.ones_like(window)
        unit_window /= unit_window.norm(p=2)
        self.best_sensitivity_evaluator = PureFetalSensitivityEvaluator(ppath_file, unit_window, measurand)
        worst_window = torch.zeros_like(window)  # Take the last two points to avoid weird noise issues
        worst_window[-2:] = 1 / sqrt(2)
        self.worst_sensitivity_evaluator = PureFetalSensitivityEvaluator(ppath_file, worst_window, measurand)

    def __str__(self) -> str:
        return "Computes Normalized Fetal Sensitivity between 0 and 1"

    def evaluate(self) -> float:
        actual_sensitivity = self.fetal_sensitivity_evaluator.evaluate()
        best_sensitivity = self.best_sensitivity_evaluator.evaluate()
        worst_sensitivity = self.worst_sensitivity_evaluator.evaluate()
        normalized_sensitivity = (actual_sensitivity - worst_sensitivity) / (best_sensitivity - worst_sensitivity)
        self.final_metric = normalized_sensitivity
        return self.final_metric