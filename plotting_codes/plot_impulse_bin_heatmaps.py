"""
Plot per-bin impulse-response metrics (SNR, Selectivity, Final Metric) as heatmaps across depths.

For every ppath file (one per dermis depth), each bin is individually turned "on" as a single-bin window and its
SNR, Selectivity, and Final Metric (Selectivity x SNR) are computed - the same brute-force per-bin scan
optimize_loop_paper.DIGSSOptimizer._compute_max_values() does, but reimplemented standalone here so this script
does not depend on that module.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from joint_tof_opt import (
    AdditiveGaussianToFModifier,
    PSAFESeparator,
    ToFData,
    WindowSumWithAdditiveGaussianNoiseCalculator,
    generate_tof,
    get_named_moment_module,
)
from joint_tof_opt.plotting import load_plot_config

NOISE_VAR = 100.0  # Fixed additive Gaussian noise variance
MEASURAND = "abs"


def compute_bin_metrics_for_file(ppath_file: Path, gen_config: dict) -> np.ndarray:
    """
    Turn each bin on individually and compute (SNR, Selectivity, Final Metric) for that single-bin window.

    :param ppath_file: Path to the partial path (.npz) file.
    :param gen_config: ToF dataset generation config (from tof_config.yaml).
    :return: Array of shape (num_bins, 3) with columns [SNR, Selectivity, Final Metric].
    """
    tof_dataset_path = Path("./data") / f"generated_tof_set_{ppath_file.stem}.npz"
    generate_tof(ppath_file, gen_config, tof_dataset_path, True, True)
    tof_data = ToFData.from_npz(tof_dataset_path)
    modifier = AdditiveGaussianToFModifier(NOISE_VAR)
    modified_tof = modifier.modify(tof_data)
    noise_calc = WindowSumWithAdditiveGaussianNoiseCalculator(NOISE_VAR)
    moment_module = get_named_moment_module(MEASURAND, modified_tof)

    sampling_rate = gen_config["sampling_rate"]
    fetal_filter = PSAFESeparator(sampling_rate, gen_config["fetal_f"], True)
    maternal_filter = PSAFESeparator(sampling_rate, gen_config["maternal_f"], True)

    num_bins = modified_tof.tof_series.shape[1]
    metrics = np.zeros((num_bins, 3))
    for i in range(num_bins):
        window = torch.zeros(num_bins)
        window[i] = 1.0
        noise_std = torch.sqrt(noise_calc.compute_noise(modified_tof, window).sum())
        compact_stats = moment_module(window)
        compact_stats = compact_stats - compact_stats.mean()
        compact_stats_reshaped = compact_stats.unsqueeze(0).unsqueeze(0)
        fetal_energy = torch.sum(fetal_filter(compact_stats_reshaped) ** 2)
        maternal_energy = torch.sum(maternal_filter(compact_stats_reshaped) ** 2)
        snr = (torch.sqrt(fetal_energy) / noise_std).item()
        selectivity = torch.sqrt(fetal_energy / maternal_energy).item()
        metrics[i] = [snr, selectivity, snr * selectivity]

    tof_dataset_path.unlink()
    return metrics


def plot_impulse_bin_heatmaps(
    snr_grid: np.ndarray,
    selectivity_grid: np.ndarray,
    final_metric_grid: np.ndarray,
    depths_mm: list[float],
    filename: str = "impulse_bin_heatmaps",
) -> None:
    """
    Plot SNR, Selectivity, and Final Metric heatmaps (depth x bin index) in a 2x2 grid (one cell left empty).

    :param snr_grid: Array of shape (num_depths, num_bins) with per-bin SNR values.
    :param selectivity_grid: Array of shape (num_depths, num_bins) with per-bin Selectivity values.
    :param final_metric_grid: Array of shape (num_depths, num_bins) with per-bin Final Metric values.
    :param depths_mm: Depth (mm) label for each row, in the same order as the grid rows.
    :param filename: Filename to save the plots. Saves to ./figures/{filename}.svg and ./figures/{filename}.pdf
    """
    load_plot_config()

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    plot_specs = [
        (axes[0, 0], snr_grid, "Normalized SNR"),
        (axes[0, 1], selectivity_grid, "Normalized Selectivity"),
        (axes[1, 0], final_metric_grid, "Normalized Reward Metric (Selectivity x SNR)"),
    ]
    for ax, grid, title in plot_specs:
        grid = grid / np.max(grid, axis=1, keepdims=True)  # Normalize per-row (per depth)
        im = ax.imshow(grid, origin="lower", aspect="auto", cmap="viridis")
        ax.set_xlabel("Active Bin Index")
        ax.set_ylabel("Depth (mm)")
        ax.set_yticks(range(len(depths_mm)))
        ax.set_yticklabels(depths_mm)
        ax.set_title(title)
        fig.colorbar(im, ax=ax)
    axes[1, 1].axis("off")
    plt.tight_layout()

    output_dir = Path("./figures/")
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{filename}.pdf"
    svg_path = output_dir / f"{filename}.svg"
    fig.savefig(pdf_path, format="pdf")
    fig.savefig(svg_path, format="svg")
    print(f"Saved PDF to: {pdf_path}")
    print(f"Saved SVG to: {svg_path}")
    plt.close()


def main() -> None:
    gen_config: dict = yaml.safe_load(open("./experiments/tof_config.yaml"))
    parameter_mapping: dict = yaml.safe_load(open("./data/parameter_mapping.json"))
    experiments = sorted(
        parameter_mapping["experiments"], key=lambda e: e["sweep_parameters"]["derm_thickness"]["value"]
    )

    depths_mm = []
    snr_rows, selectivity_rows, final_metric_rows = [], [], []
    for experiment in experiments:
        ppath_file = Path("./data") / experiment["filename"]
        print(f"Computing per-bin metrics for: {experiment['filename']}")
        metrics = compute_bin_metrics_for_file(ppath_file, gen_config)
        depths_mm.append(experiment["sweep_parameters"]["derm_thickness"]["value"] + 2)
        snr_rows.append(metrics[:, 0])
        selectivity_rows.append(metrics[:, 1])
        final_metric_rows.append(metrics[:, 2])

    plot_impulse_bin_heatmaps(
        np.stack(snr_rows), np.stack(selectivity_rows), np.stack(final_metric_rows), depths_mm
    )


if __name__ == "__main__":
    main()
