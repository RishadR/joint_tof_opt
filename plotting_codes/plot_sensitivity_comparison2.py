"""
Plot Selectivity vs. SNR for different optimizers at various fetal depths.
Compares DIGSS, Liu et al., and CW methods.
Plots mean values with shaded "error balls" (ellipses) representing uncertainty.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from joint_tof_opt.plotting import load_plot_config


def main():
    """Generate Selectivity vs. SNR scatter plot with error balls."""
    # Configuration: Control which depth points to annotate (step size)
    ANNOTATION_STEP = 1     # Annotate every Nth point for all methods

    # Load matplotlib configuration
    load_plot_config()

    # Load sensitivity comparison results
    results_path = Path(__file__).parent.parent / "results" / "sensitivity_comparison_results.yaml"
    with open(results_path) as f:
        results = yaml.safe_load(f)

    # Group data: {label: {depth: {"snr": [], "selectivity": []}}}
    grouped_data = {}

    for _, exp_data in results.items():
        if not isinstance(exp_data, dict):
            continue

        depth = exp_data.get("Depth_mm")
        optimizer = exp_data.get("Optimizer", "")
        evaluator_log = exp_data.get("evaluator_log", {})

        # Extract required values from evaluator_log
        fetal_ac_energy = evaluator_log.get("fetal_ac_energy")
        maternal_ac_energy = evaluator_log.get("maternal_ac_energy")
        baseline_noise_std = evaluator_log.get("baseline_noise_std")

        if depth is None or fetal_ac_energy is None or maternal_ac_energy is None or baseline_noise_std is None:
            continue

        # Compute Selectivity and SNR
        selectivity = np.sqrt(fetal_ac_energy / maternal_ac_energy)
        snr = np.sqrt(fetal_ac_energy) / baseline_noise_std

        # Determine label
        optimizer_str = str(optimizer)
        label = None
        if optimizer_str.startswith("DIGSSOptimizer"):
            if "normalization_scheme=unit_sum" in optimizer_str:
                label = "DIGSS(Unit Sum)"
            elif "normalization_scheme=unit_max" in optimizer_str:
                label = "DIGSS"
        elif optimizer_str.startswith("LiuOptimizer"):
            label = "Spectral Boxcar$^{[27]}$"
        elif optimizer_str.startswith("AltLiuOptimizer"):
            label = "Alt. Boxcar"
        elif optimizer_str.startswith("DummyUnitWindowGenerator"):
            label = "CW"
        elif optimizer_str.startswith("BoxCarOptimizer"):
            label = "Brute Force Boxcar"
        if label:
            if label not in grouped_data:
                grouped_data[label] = {}
            if depth not in grouped_data[label]:
                grouped_data[label][depth] = {"snr": [], "selectivity": []}
            grouped_data[label][depth]["snr"].append(snr)
            grouped_data[label][depth]["selectivity"].append(selectivity)

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 4))

    labels_to_plot = ["DIGSS", "Spectral Boxcar$^{[27]}$", "Brute Force Boxcar", "CW"]
    # offsets = [-0.02, 0.00, +0.02]

    # labels_to_plot = ["DIGSS", "Boxcar$^{[27]}$", "CW"]

    for label in labels_to_plot:
        if label not in grouped_data:
            continue

        depths = sorted(grouped_data[label].keys())
        snr_means = []
        snr_stds = []
        sel_means = []
        sel_stds = []
        plot_depths = []

        for d in depths:
            snrs = grouped_data[label][d]["snr"]
            sels = grouped_data[label][d]["selectivity"]

            snr_means.append(np.mean(snrs))
            snr_stds.append(np.std(snrs))
            sel_means.append(np.mean(sels))
            sel_stds.append(np.std(sels))
            plot_depths.append(d)

        # Plot the mean line
        (line,) = ax.plot(snr_means, sel_means, label=label)
        color = line.get_color()

        theta = np.linspace(0, 2 * np.pi, 100)
        for i in range(len(snr_means)):
            dz_x = 0.434 * snr_stds[i] / snr_means[i]
            dz_y = 0.434 * sel_stds[i] / sel_means[i]
            x_pts = 10 ** (np.log10(snr_means[i]) + dz_x * np.cos(theta))
            y_pts = 10 ** (np.log10(sel_means[i]) + dz_y * np.sin(theta))
            ax.fill(x_pts, y_pts, color=color, alpha=0.15, edgecolor="none")

        # Add depth annotations on DIGSS only
        if label == labels_to_plot[-1]:  # Only annotate for the last label (Ideally CW - cleaner curve)
            for i, depth in enumerate(plot_depths):
                if i % ANNOTATION_STEP == 0:
                    depth_cm = depth / 10
                    ax.annotate(
                        f"{depth_cm:.1f} cm",
                        (snr_means[i], sel_means[i]),
                        textcoords="offset points",
                        xytext=(5, 0),
                        ha="left",
                        fontsize=7,
                        alpha=0.7,
                        bbox={"boxstyle": "round,pad=0.15", "fc": "lightgray", "ec": "none", "alpha": 0.7},
                    )

    # Configure axes
    ax.set_xlabel("Fetal SNR")
    ax.set_ylabel("Fetal Selectivity")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(loc="best")
    ax.grid(True, which="major")
    ax.grid(True, which="minor", alpha=0.3)
    ax.minorticks_on()

    # Save figure
    figures_dir = Path(__file__).parent.parent / "figures"
    figures_dir.mkdir(exist_ok=True)

    fig.savefig(figures_dir / "sensitivity_comparison2.pdf", format="pdf")
    fig.savefig(figures_dir / "sensitivity_comparison2.svg", format="svg")

    print(f"Selectivity vs. SNR plot saved to {figures_dir}")
    # plt.show()


if __name__ == "__main__":
    main()
