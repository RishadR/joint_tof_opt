"""
Code to compute the sensitivity of some measurand w.r.t. some window

Sensitivity is defined as the change in measurand per unit change in mu_a (in dL/M)
You can compare sensitivities across different windows to see which window gives better

For our case, the measurand is a mix between fetal and maternal hemoglobin changes - so sensitivities are computed
via a bandpass comb filter and an energy ratio metric. Which is further square rooted to get sensitivity in terms
of amplitude.

Process flow:
1. Read the ToF dataset from the given path
2. Use the given window to compute the measurand time series
3. Get the fetal_hb_series and maternal_hb_series from the dataset
4. Extract the fetal and maternal componets of the measurand using comb filters @FHR and MHR
5. Remove the DC from fetal_hb_series and maternal_hb_series
6. Compute Fetal Sensitivity as sqrt(energy ratio between measurand fetal component and fetal_hb_series)
7. Compute Maternal Sensitivity as sqrt(energy ratio between measurand maternal component and maternal_hb_series)
8. Return the two sensitivities
"""

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
    generate_tof
)


class FetalSensitivityNoInterferenceEvaluator(Evaluator):
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

        light_speeds = [
            float(speed) for speed in config["light_speeds"]
        ]  # in m/s for 4 layers

        # Load partial path data
        ppath_dataset = np.load(self.ppath_file)
        ppath = ppath_dataset["ppath"]  # Shape: (num_photons, num_layers + 1)
        bin_count = config["bin_count"]
        assert bin_count == len(
            self.window
        ), "Window length must match bin count in tof_config.yaml"
        fraction = config["weight_threshold_fraction"]
        filtered_ppath = (ppath[ppath[:, 0] == config["selected_sdd_index"]])[
            :, 1:
        ]  # Drop the sdd index column

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
            moment_calculator = get_named_moment_module(
                self.measurand, tof_series_tensor, bin_edges_tensor
            )
        else:
            moment_calculator = self.measurand

        measurand_values = moment_calculator.forward(self.window)
        measurand_values = measurand_values.detach().cpu().numpy()

        # Compute sensitivity
        delta_mu_a_fetal = (
            perturbed_model.prop[-1][0] - base_model.prop[-1][0]
        )  # Change in fetal mu_a in mm-1
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
            self.moment_module = get_named_moment_module(
                self.measurand, tof_series_tensor, bin_edges_tensor
            )
            sampling_rate = gen_config["sampling_rate"]
            maternal_f = gen_config["maternal_f"]
            fetal_f = gen_config["fetal_f"]
            maternal_hb_series = tof_dataset["maternal_hb_series"]
            fetal_hb_series = tof_dataset["fetal_hb_series"]
        else:
            assert (
                self.measurand.meta_data is not None
            ), "Meta data must be provided in the measurand module"
            assert (
                self.measurand.meta_data["sampling_rate"] is not None
            ), "Sampling rate must be in the meta data"
            assert (
                self.measurand.meta_data["maternal_f"] is not None
            ), "Maternal f must be in the meta data"
            assert (
                self.measurand.meta_data["fetal_f"] is not None
            ), "Fetal f must be in the meta data"
            assert (
                self.measurand.meta_data["maternal_hb_series"] is not None
            ), "Maternal hb series must be present"
            assert (
                self.measurand.meta_data["fetal_hb_series"] is not None
            ), "Fetal hb series must be present"
            self.moment_module = self.measurand
            tof_data = self.measurand.tof_series
            sampling_rate = self.measurand.meta_data["sampling_rate"]
            maternal_f = self.measurand.meta_data["maternal_f"]
            fetal_f = self.measurand.meta_data["fetal_f"]
            maternal_hb_series = self.measurand.meta_data["maternal_hb_series"]
            fetal_hb_series = self.measurand.meta_data["fetal_hb_series"]

        # Compute compact statistics
        compact_stats = self.moment_module(self.window)  # Shape: (num_timepoints,)
        compact_stats_reshaped = compact_stats.unsqueeze(0).unsqueeze(
            0
        )  # Shape: (1, 1, num_timepoints)

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
        maternal_hb_series_tensor = torch.tensor(
            maternal_hb_series, dtype=torch.float32
        )
        fetal_hb_series_tensor = torch.tensor(fetal_hb_series, dtype=torch.float32)

        # Compute sensitivities
        energy_ratio_metric = EnergyRatioMetric()
        fetal_sensitivity = torch.sqrt(
            energy_ratio_metric(fetal_filtered_signal, fetal_hb_series_tensor)
        ).item()
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
            self.moment_module = get_named_moment_module(
                self.measurand, tof_series_tensor, bin_edges_tensor
            )
            fetal_hb_series = tof_dataset["fetal_hb_series"]
        else:
            self.moment_module = self.measurand
            assert (
                self.measurand.meta_data is not None
            ), "Meta data must be provided in the measurand module"
            assert (
                self.measurand.meta_data["fetal_hb_series"] is not None
            ), "Fetal hb series must be in the meta data"
            tof_data = self.measurand.tof_series
            fetal_hb_series = self.measurand.meta_data["fetal_hb_series"]

        # Compute compact statistics
        compact_stats = self.moment_module(self.window)  # Shape: (num_timepoints,)
        assert (
            self.moment_module.meta_data is not None
        ), "Meta data must be provided in the measurand module"
        assert (
            self.moment_module.meta_data["fetal_f"] is not None
        ), "Fetal frequency must be in the meta data"
        assert (
            self.moment_module.meta_data["maternal_f"] is not None
        ), "Maternal frequency must be in the meta data"
        assert (
            self.moment_module.meta_data["sampling_rate"] is not None
        ), "Sampling rate must be in the meta data"
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
        self.filtered_signal = self.comb_filter(
            compact_stats.unsqueeze(0).unsqueeze(0)
        )  # Shape: (1, 1, num_timepoints)
        self.filtered_signal = (
            self.filtered_signal.squeeze()
        )  # Shape: (num_timepoints,)

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
        noise_func: (Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor] | None) = None,
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
            assert (
                noise_func is not None
            ), "Noise function must be provided for custom measurand modules"
            self.noise_func = noise_func
        else:
            self.noise_func = noise_func_table[measurand]

    def __str__(self) -> str:
        return "Computes Correlation x SNR between measurand and fetal or maternal hb changes"

    def evaluate(self) -> float:
        correlation = self._correlation_evaluator.evaluate()
        self.moment_module = self._correlation_evaluator.moment_module
        self.comb_filter = self._correlation_evaluator.comb_filter
        assert (
            self.moment_module is not None
        ), "Internal CorrelationEvaluator did not work properly!"
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
