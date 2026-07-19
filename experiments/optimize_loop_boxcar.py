"""
BoxCarOptimizer: Alternate version of DIGSSOptimizer that finds the optimal boxcar (rectangular) window via
brute-force search over all (left_idx, right_idx) combinations, rather than gradient-based optimization.
"""

import logging
from pathlib import Path

import numpy as np
import torch
import yaml

from joint_tof_opt import (
    AdditiveGaussianToFModifier,
    ToFData,
    WindowSumWithAdditiveGaussianNoiseCalculator,
    generate_tof,
)
from optimize_loop_paper import DIGSSOptimizer
from sensitivity_compute import AltPaperEvaluator3

logger = logging.getLogger(__name__)


class BoxCarOptimizer(DIGSSOptimizer):
    """
    Brute-force boxcar (rectangular) window optimizer.

    Same setup as DIGSSOptimizer (same __init__, filters, max SNR/selectivity bounds). Only optimize() differs:
    instead of gradient descent over exponentiated window weights, this exhaustively tries every (left_idx,
    right_idx) pair within the learnable window range, builds a rectangular window of ones between them, and keeps
    the one that maximizes the same final_metric (selectivity * snr) DIGSSOptimizer uses. No early stopping, no
    regularization, no window smoothening/post-processing - the boxcar itself is already the final window.
    """

    def optimize(self):
        """
        Brute-force search over all boxcar windows (left_idx, right_idx) to maximize final_metric.
        """
        num_learnable = self.learnable_component_exponents.numel()
        num_combos = num_learnable * (num_learnable + 1) // 2
        self.training_curves = np.zeros((num_combos, 3))

        best_metric = -np.inf
        best_window = self.window.clone()

        combo = 0
        with torch.no_grad():
            for left_idx in range(num_learnable):
                for right_idx in range(left_idx, num_learnable):
                    boxcar = torch.zeros(num_learnable, dtype=self.learnable_component_exponents.dtype)
                    boxcar[left_idx : right_idx + 1] = 1.0
                    window = torch.cat([self.fixed_left, boxcar, self.fixed_right], dim=0)
                    window_norm = self._win_norm_func(window)

                    compact_stats = self.moment_module(window_norm)
                    compact_stats = compact_stats - compact_stats.mean()
                    compact_stats_reshaped = compact_stats.unsqueeze(0).unsqueeze(0)
                    maternal_filtered_signal = self.maternal_filter(compact_stats_reshaped)
                    fetal_filtered_signal = self.fetal_filter(compact_stats_reshaped)

                    fetal_energy = torch.sum(fetal_filtered_signal**2)
                    maternal_energy = torch.sum(maternal_filtered_signal**2)
                    baseline_noise_var = self.noise_calc.compute_noise(self.tof_data, window_norm).sum()
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
                        best_window = window_norm.clone()

        self.window = best_window.detach()
        self.unprocessed_window = self.window.clone()

    def __str__(self) -> str:
        return (
            f"BoxCarOptimizer(measurand={self.moment_module.__class__.__name__}, "
            f"filter_hw={self.filter_hw}"
            f"fetal_f={self.fetal_f}), type={self.filter_type}"
            f"normalize_reward={self.normalize_reward},"
            f"normalization_scheme={self.normalization_scheme},"
            f"use_snr_left_bound={self.use_snr_left_bound}"
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    file_idx = 2
    measurand = "abs"
    ppath_file = Path(f"./data/experiment_{file_idx:04d}.npz")
    logger.info("Running BoxCar optimization loop for file: %04d.npz | Measurand: %s", file_idx, measurand)
    tof_dataset_path = Path("./data") / f"generated_tof_set_{ppath_file.stem}.npz"
    gen_config: dict = yaml.safe_load(open("./experiments/tof_config.yaml"))
    filter_hw = 0.01
    noise_var = 100.0
    generate_tof(ppath_file, gen_config, tof_dataset_path, True, True)
    tof_data = ToFData.from_npz(tof_dataset_path)
    modifier = AdditiveGaussianToFModifier(noise_var)
    modified_tof = modifier.modify(tof_data)
    noise_calc = WindowSumWithAdditiveGaussianNoiseCalculator(noise_var)
    experiment = BoxCarOptimizer(
        tof_data=modified_tof,
        measurand=measurand,
        noise_calc=noise_calc,
        fetal_f=gen_config["fetal_f"],
        normalize_reward=False,
        filter_hw=filter_hw,
        filter_type="psafe_same_width",
        normalization_scheme="unit_max",
    )
    experiment.optimize()

    optimized_window = experiment.window  # type: ignore
    result_curves = experiment.training_curves
    logger.info("Optimized Window: %s", optimized_window.numpy())
    logger.info("Best Final Metric: %s", result_curves[:, 2].max())
    logger.info("Total Combos Tried: %s", result_curves.shape[0])

    evaluator = AltPaperEvaluator3(ppath_file, optimized_window, measurand, gen_config, filter_hw)
    eval_results = evaluator.evaluate()
    logger.info("Evaluation Results: %s", eval_results)
    logger.info("Evaluator log: %s", evaluator.get_log())

    tof_dataset_path.unlink()  # Remove the generated ToF dataset to save space


if __name__ == "__main__":
    main()
