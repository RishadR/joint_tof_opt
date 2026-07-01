"""
Plot Selectivity vs. Fetal SNR for different noise variances (Unit Max).
Each curve is one noise level traversing fetal depths.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.patches import Ellipse

from joint_tof_opt.plotting import load_plot_config


def main():
    """Generate Selectivity vs. Fetal SNR scatter plot grouped by noise level."""
    ANNOTATION_STEP = 3
    TOTAL_PHOTONS = 1e9

    load_plot_config()

    results_path = Path(__file__).parent.parent / "results" / "noise_sensitivity_comparison_results.yaml"
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
        evaluator_log = exp_data.get("evaluator_log", {})

        fetal_ac_energy = evaluator_log.get("fetal_ac_energy")
        maternal_ac_energy = evaluator_log.get("maternal_ac_energy")
        baseline_noise_std = evaluator_log.get("baseline_noise_std")

        if any(v is None for v in [depth, noise_var, fetal_ac_energy, maternal_ac_energy, baseline_noise_std]):
            continue

        selectivity = np.sqrt(fetal_ac_energy / maternal_ac_energy)
        snr = np.sqrt(fetal_ac_energy) / baseline_noise_std

        if noise_var not in grouped_data:
            grouped_data[noise_var] = {}
        if depth not in grouped_data[noise_var]:
            grouped_data[noise_var][depth] = {"snr": [], "selectivity": []}
        grouped_data[noise_var][depth]["snr"].append(snr)
        grouped_data[noise_var][depth]["selectivity"].append(selectivity)

    fig, ax = plt.subplots(figsize=(6, 4))

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

        noise_pct = 100 * np.sqrt(noise_var) / TOTAL_PHOTONS
        label = f"{noise_pct:.2g}% noise"

        (line,) = ax.plot(snr_means, sel_means, label=label)
        color = line.get_color()

        for j in range(len(snr_means)):
            ellipse = Ellipse(
                (snr_means[j], sel_means[j]),
                width=2 * snr_stds[j],
                height=2 * sel_stds[j],
                facecolor=color,
                alpha=0.15,
                edgecolor="none",
            )
            ax.add_patch(ellipse)

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
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, which="major")
    ax.grid(True, which="minor", alpha=0.3)
    ax.minorticks_on()

    figures_dir = Path(__file__).parent.parent / "figures"
    figures_dir.mkdir(exist_ok=True)

    fig.savefig(figures_dir / "noise_sensitivity_comparison2.pdf", format="pdf")
    fig.savefig(figures_dir / "noise_sensitivity_comparison2.svg", format="svg")

    print(f"Noise Selectivity vs. SNR plot saved to {figures_dir}")


if __name__ == "__main__":
    main()
