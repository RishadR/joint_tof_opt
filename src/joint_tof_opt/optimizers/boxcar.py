"""
BoxCarOptimizer: Finds the optimal boxcar (rectangular) window via brute-force search over all
(left_idx, right_idx) combinations, rather than gradient-based optimization (see DIGSSOptimizer).
"""

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
from typing_extensions import override

from joint_tof_opt.compact_stat_process import get_named_moment_module
from joint_tof_opt.core import CompactStatProcess, NoiseCalculator, OptimizationExperiment, ToFData
from joint_tof_opt.metric_process import ContrastToNoiseMetric, EnergyRatioMetric
from joint_tof_opt.noise_calc import WindowSumNoiseCalculator
from joint_tof_opt.optimizers.specs import DEFAULT_SPECS_PATH, FilterType, load_optimizer_specs
from joint_tof_opt.signal_process import CombSeparator, FourierSeparator, PSAFESeparator

_BOXCAR_SPEC = load_optimizer_specs(DEFAULT_SPECS_PATH).boxcar

_Filter = CombSeparator | FourierSeparator | PSAFESeparator


class BoxCarOptimizer(OptimizationExperiment):
    """
    Brute-force boxcar (rectangular) window optimizer.

    Exhaustively tries every (left_idx, right_idx) pair (at least one bin wide), builds a rectangular window of ones
    between them (zeros elsewhere), and keeps the one that maximizes final_metric (selectivity * snr). right_idx is
    bounded by the last bin whose signal power is at least its mean noise variance; left_idx can start at bin 0.
    No early stopping, no regularization, no window smoothening/post-processing - the boxcar itself is already the
    final window.
    """

    def __init__(
        self,
        tof_data: ToFData,
        measurand: str | CompactStatProcess,
        noise_calc: NoiseCalculator | None = None,
        fetal_f: float | None = None,
        filter_hw: float = _BOXCAR_SPEC.filter_hw,
        normalize_reward: bool = _BOXCAR_SPEC.normalize_reward,
        filter_type: FilterType = _BOXCAR_SPEC.filter_type,
    ):
        """
        :param tof_data: ToFData instance to optimize on.
        :param measurand: The measurand to optimize for ("abs", "m1", "V") or custom module.
        :param noise_calc: Noise calculator - defaults to WindowSumNoiseCalculator
        :param fetal_f: Central frequency of fetal filter (in Hz). If None, extracted from dataset metadata.
        :param filter_hw: Half width of the comb/fourier filters (in Hz).
        :param normalize_reward: Whether to normalize SNR and selectivity by their single-bin maxima.
        :param filter_type: Type of filter to use (see DIGSSOptimizer for the options).
        """
        if isinstance(measurand, str):
            measurand = get_named_moment_module(measurand, tof_data)
        super().__init__(tof_data, measurand)

        self.noise_calc: NoiseCalculator = noise_calc if noise_calc is not None else WindowSumNoiseCalculator()
        self.filter_hw: float = filter_hw
        self.filter_type: FilterType = filter_type
        self.normalize_reward: bool = normalize_reward

        assert self.tof_data.meta_data is not None, "ToFData meta_data cannot be None"
        self.sampling_rate: float = self.tof_data.meta_data["sampling_rate"]
        self.fetal_f: float = fetal_f if fetal_f is not None else self.tof_data.meta_data["fetal_f"]
        self.maternal_f: float = self.tof_data.meta_data["maternal_f"]
        self.fetal_filter: _Filter
        self.maternal_filter: _Filter
        self.fetal_filter, self.maternal_filter = self._get_filters(filter_type)

        max_snr, max_selectivity = self._compute_max_values()
        self.max_snr: float = max_snr
        self.max_selectivity: float = max_selectivity

        self.num_bins: int = self.tof_data.tof_series.shape[1]
        time_points = self.tof_data.tof_series.shape[0]

        # Computing the right bound - power should always be greater than variance
        mean_frame = self.tof_data.tof_series.mean(dim=0)  # Shape: (num_bins,)
        signal_power = mean_frame**2  # Shape: (num_bins,)
        unity_window = torch.ones(self.num_bins, device=self.tof_data.bin_edges.device)
        total_noise_variance = self.noise_calc.compute_noise(
            self.tof_data, unity_window, sum_axis=0
        )  # Shape:(num_bins,)
        mean_noise_variance = total_noise_variance / time_points
        viable_bins = torch.where(signal_power >= mean_noise_variance)[0]
        right_most_bin = int(viable_bins[-1].item()) if viable_bins.numel() > 0 else -1
        assert right_most_bin >= 0, "No viable bins - every bin's signal power is below its noise variance."
        self.right_most_bin: int = right_most_bin

        self.window: torch.Tensor = unity_window
        self.training_curve_labels: list[str] = ["Energy Ratio", "Contrast-to-Noise", "Final Metric"]

    def _get_filters(self, filter_type: FilterType) -> tuple[_Filter, _Filter]:
        datapoint_count = int(self.tof_data.tof_series.shape[0])
        sampling_rate = self.sampling_rate
        if filter_type == "comb":
            fetal_filter = CombSeparator(
                fs=sampling_rate,
                f0=self.fetal_f,
                f1=2 * self.fetal_f,
                half_width=self.filter_hw,
                filter_length=datapoint_count // 2 + 1,
            )
            maternal_filter = CombSeparator(
                fs=sampling_rate,
                f0=self.maternal_f,
                f1=2 * self.maternal_f,
                half_width=self.filter_hw,
                filter_length=datapoint_count // 2 + 1,
            )
        elif filter_type == "fourier":
            fetal_filter = FourierSeparator(
                fs=sampling_rate,
                f0=self.fetal_f,
                f1=2 * self.fetal_f,
                half_width=self.filter_hw,
            )
            maternal_filter = FourierSeparator(
                fs=sampling_rate,
                f0=self.maternal_f,
                f1=2 * self.maternal_f,
                half_width=self.filter_hw,
            )
        elif filter_type == "psafe_same_width":
            fetal_filter = PSAFESeparator(fs=sampling_rate, center_freq=self.fetal_f, equate_length=True)
            maternal_filter = PSAFESeparator(fs=sampling_rate, center_freq=self.maternal_f, equate_length=True)
        elif filter_type == "psafe_true_width":
            fetal_filter = PSAFESeparator(fs=sampling_rate, center_freq=self.fetal_f, equate_length=False)
            maternal_filter = PSAFESeparator(fs=sampling_rate, center_freq=self.maternal_f, equate_length=False)
        elif filter_type == "comb_psafe_hybrid":
            fetal_filter = PSAFESeparator(fs=sampling_rate, center_freq=self.fetal_f, equate_length=True)
            maternal_filter = CombSeparator(
                fs=sampling_rate,
                f0=self.maternal_f,
                f1=2 * self.maternal_f,
                half_width=self.filter_hw,
                filter_length=datapoint_count // 2 + 1,
            )
        else:
            raise ValueError(f"Unknown filter_type: {filter_type}")  # pyright: ignore[reportUnreachable]
        return fetal_filter, maternal_filter

    def _filter_signals(self, window: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :return: (fetal_filtered_signal, maternal_filtered_signal) of the DC-removed compact statistic for `window`.
        """
        compact_stats = self.moment_module.forward(window)
        compact_stats = compact_stats - compact_stats.mean()
        compact_stats_reshaped = compact_stats.unsqueeze(0).unsqueeze(0)
        return self.fetal_filter.forward(compact_stats_reshaped), self.maternal_filter.forward(compact_stats_reshaped)

    def _compute_max_values(self) -> tuple[float, float]:
        """
        Compute the max SNR and max selectivity over single-bin (impulse) windows.

        :return: (max_snr, max_selectivity)
        """
        num_bins = self.tof_data.tof_series.shape[1]
        snr_calc = ContrastToNoiseMetric(noise_calc=self.noise_calc, tof_data=self.tof_data)
        selectivity_calc = EnergyRatioMetric()
        snr_list: list[float] = []
        selectivity_list: list[float] = []

        for i in range(num_bins):
            single_bin_window = torch.zeros(num_bins)
            single_bin_window[i] = 1.0
            fetal_filtered_signal, maternal_filtered_signal = self._filter_signals(single_bin_window)
            fetal_energy = torch.sum(fetal_filtered_signal**2)
            maternal_energy = torch.sum(maternal_filtered_signal**2)
            selectivity_list.append(selectivity_calc.forward(fetal_energy, maternal_energy).item())
            snr_list.append(snr_calc.forward(single_bin_window, fetal_filtered_signal).item())

        return max(snr_list), max(selectivity_list)

    @override
    def optimize(self):
        """
        Brute-force search over all boxcar windows (left_idx, right_idx) to maximize final_metric.
        """
        num_search_bins = self.right_most_bin + 1
        num_combos = num_search_bins * (num_search_bins + 1) // 2
        self.training_curves: npt.NDArray[np.float64] = np.zeros((num_combos, 3))

        best_metric = -np.inf
        best_window = self.window.clone()

        combo = 0
        with torch.no_grad():
            for left_idx in range(num_search_bins):
                for right_idx in range(left_idx, num_search_bins):
                    window = torch.zeros(self.num_bins, device=self.window.device)
                    window[left_idx : right_idx + 1] = 1.0

                    fetal_filtered_signal, maternal_filtered_signal = self._filter_signals(window)
                    fetal_energy = torch.sum(fetal_filtered_signal**2)
                    maternal_energy = torch.sum(maternal_filtered_signal**2)
                    baseline_noise_var = self.noise_calc.compute_noise(self.tof_data, window).sum()
                    baseline_noise_std = torch.sqrt(baseline_noise_var)
                    selectivity = torch.sqrt(fetal_energy / maternal_energy)
                    snr = torch.sqrt(fetal_energy) / baseline_noise_std

                    if self.normalize_reward:
                        snr = snr / float(self.max_snr)
                        selectivity = selectivity / float(self.max_selectivity)
                    final_metric = selectivity * snr

                    self.training_curves[combo, 0] = selectivity.item()
                    self.training_curves[combo, 1] = snr.item()
                    self.training_curves[combo, 2] = final_metric.item()
                    combo += 1

                    if final_metric.item() > best_metric:
                        best_metric = final_metric.item()
                        best_window = window.clone()

        self.window = best_window.detach()

    @override
    def __str__(self) -> str:
        return (
            f"BoxCarOptimizer(measurand={self.moment_module.__class__.__name__}, "
            f"filter_hw={self.filter_hw}"
            f"fetal_f={self.fetal_f}), type={self.filter_type}"
            f"normalize_reward={self.normalize_reward},"
            "normalization_scheme=unit_max"
        )

    @override
    def components(self) -> dict[str, nn.Module]:
        return {
            "fetal_filter": self.fetal_filter,
            "maternal_filter": self.maternal_filter,
            "measurand": self.moment_module,
        }
