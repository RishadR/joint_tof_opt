"""
Plot Selectivity vs. SNR for different optimizers at various fetal depths.
Plots mean values with shaded "error balls" (ellipses) representing uncertainty.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from joint_tof_opt.misc import noisy_results_path
from joint_tof_opt.plotting import legend_no_overlap, load_plot_config, log_samples, resolve_results_path


def main():
    """Generate Selectivity vs. SNR scatter plot with error balls."""
    # Configuration: Control which depth points to annotate (step size)
    ANNOTATION_STEP = 1     # Annotate every Nth point for all methods

    # Load matplotlib configuration
    load_plot_config()

    # Load sensitivity comparison results (noisy/noiseless file picked via evaluator_specs.yaml)
    base_results_path = Path(__file__).parent.parent / "results" / "sensitivity_comparison_results.yaml"
    results_path, inject_noise = resolve_results_path(base_results_path)
    with open(results_path) as f:
        results = yaml.safe_load(f)

    # Group data: {label: {depth: {"snr": [], "selectivity": []}}}
    grouped_data = {}

    for _, exp_data in results.items():
        if not isinstance(exp_data, dict):
            continue

        depth = exp_data.get("Depth_mm")
        optimizer = exp_data.get("Optimizer", "")
        evaluator_log = exp_data.get("evaluator_log")

        if depth is None or evaluator_log is None:
            continue

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
        if not label:
            continue

        for log in log_samples(evaluator_log):
            fetal_ac_energy = log.get("fetal_ac_energy")
            maternal_ac_energy = log.get("maternal_ac_energy")
            baseline_noise_std = log.get("baseline_noise_std")
            if fetal_ac_energy is None or maternal_ac_energy is None or baseline_noise_std is None:
                continue

            selectivity = np.sqrt(fetal_ac_energy / maternal_ac_energy)
            snr = np.sqrt(fetal_ac_energy) / baseline_noise_std

            depth_data = grouped_data.setdefault(label, {}).setdefault(depth, {"snr": [], "selectivity": []})
            depth_data["snr"].append(snr)
            depth_data["selectivity"].append(selectivity)

    # Create figure
    fig, ax = plt.subplots()

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
    legend_no_overlap(ax, "lower right")
    ax.grid(True, which="major")
    ax.grid(True, which="minor", alpha=0.3)
    ax.minorticks_on()

    # Save figure
    figures_dir = Path(__file__).parent.parent / "figures"
    figures_dir.mkdir(exist_ok=True)

    fig.savefig(noisy_results_path(figures_dir / "sensitivity_comparison2.pdf", inject_noise), format="pdf")
    fig.savefig(noisy_results_path(figures_dir / "sensitivity_comparison2.svg", inject_noise), format="svg")

    print(f"Selectivity vs. SNR plot saved to {figures_dir}")
    # plt.show()


if __name__ == "__main__":
    main()
