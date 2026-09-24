"""
Plot Selectivity vs. Fetal SNR for different noise variances (Unit Max).
Each curve is one noise level traversing fetal depths.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from joint_tof_opt.misc import noisy_results_path
from joint_tof_opt.plotting import legend_no_overlap, load_plot_config, log_samples, resolve_results_path


def main():
    """Generate Selectivity vs. Fetal SNR scatter plot grouped by noise level."""
    ANNOTATION_STEP = 3

    load_plot_config()

    with open(Path(__file__).parent.parent / "experiments" / "tof_config.yaml") as f:
        config = yaml.safe_load(f)
        total_photon_count = config.get("total_photon_count", 1e6)  # Default to 1e6 if not specified

    base_results_path = Path(__file__).parent.parent / "results" / "noise_sensitivity_comparison_results.yaml"
    results_path, inject_noise = resolve_results_path(base_results_path)
    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        return

    with open(results_path) as f:
        results = yaml.safe_load(f)

    # Group data: {noise_var: {depth_mm: {"snr": [], "selectivity": []}}}
    grouped_data = {}

    for _, exp_data in results.items():
        if not isinstance(exp_data, dict):
            continue

        depth = exp_data.get("Depth_mm")
        noise_var = exp_data.get("noise_variance")
        evaluator_log = exp_data.get("evaluator_log")

        if depth is None or noise_var is None or evaluator_log is None:
            continue

        depth_data = grouped_data.setdefault(noise_var, {}).setdefault(depth, {"snr": [], "selectivity": []})
        for log in log_samples(evaluator_log):
            fetal_ac_energy = log.get("fetal_ac_energy")
            maternal_ac_energy = log.get("maternal_ac_energy")
            baseline_noise_std = log.get("baseline_noise_std")
            if fetal_ac_energy is None or maternal_ac_energy is None or baseline_noise_std is None:
                continue

            depth_data["snr"].append(np.sqrt(fetal_ac_energy) / baseline_noise_std)
            depth_data["selectivity"].append(np.sqrt(fetal_ac_energy / maternal_ac_energy))

    fig, ax = plt.subplots()

    sorted_vars = sorted(grouped_data.keys())[::2]  # every other noise level

    for i, noise_var in enumerate(sorted_vars):
        depths = sorted(grouped_data[noise_var].keys())
        snr_means, snr_stds, sel_means, sel_stds, plot_depths = [], [], [], [], []

        for d in depths:
            snrs = grouped_data[noise_var][d]["snr"]
            sels = grouped_data[noise_var][d]["selectivity"]
            snr_means.append(np.mean(snrs))
            snr_stds.append(np.std(snrs))
            sel_means.append(np.mean(sels))
            sel_stds.append(np.std(sels))
            plot_depths.append(d)

        if noise_var == 0.0:
            label = "Noiseless"
        else:
            label = f"Normalized Noise $\\sigma$ : {np.sqrt(noise_var) / total_photon_count:.2e}"

        (line,) = ax.plot(snr_means, sel_means, label=label)
        color = line.get_color()

        theta = np.linspace(0, 2 * np.pi, 100)
        for j in range(len(snr_means)):
            dz_x = 0.434 * snr_stds[j] / snr_means[j]
            dz_y = 0.434 * sel_stds[j] / sel_means[j]
            x_pts = 10 ** (np.log10(snr_means[j]) + dz_x * np.cos(theta))
            y_pts = 10 ** (np.log10(sel_means[j]) + dz_y * np.sin(theta))
            ax.fill(x_pts, y_pts, color=color, alpha=0.15, edgecolor="none")

        # Annotate depths on the first (lowest noise) curve only
        if i == 0:
            for j, depth in enumerate(plot_depths):
                if j % ANNOTATION_STEP == 0:
                    ax.annotate(
                        f"{depth / 10:.1f} cm",
                        (snr_means[j], sel_means[j]),
                        textcoords="offset points",
                        xytext=(0, -10),
                        ha="left",
                        fontsize=7,
                        alpha=0.7,
                        bbox={"boxstyle": "round,pad=0.15", "fc": "lightgray", "ec": "none", "alpha": 0.7},
                    )

    ax.set_xlabel("Fetal SNR")
    ax.set_ylabel("Fetal Selectivity")
    ax.set_xscale("log")
    ax.set_yscale("log")
    legend_no_overlap(ax, "lower right")
    ax.grid(True, which="major")
    ax.grid(True, which="minor", alpha=0.3)
    ax.minorticks_on()

    figures_dir = Path(__file__).parent.parent / "figures"
    figures_dir.mkdir(exist_ok=True)

    fig.savefig(noisy_results_path(figures_dir / "noise_sensitivity_comparison2.pdf", inject_noise), format="pdf")
    fig.savefig(noisy_results_path(figures_dir / "noise_sensitivity_comparison2.svg", inject_noise), format="svg")

    print(f"Noise Selectivity vs. SNR plot saved to {figures_dir}")


if __name__ == "__main__":
    main()
