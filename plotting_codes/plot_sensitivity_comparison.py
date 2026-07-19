"""
Plot Optimized Sensitivity vs. Fetal Depth for different optimizers.
Compares DIGSS, Liu et al., and CW methods.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from joint_tof_opt.plotting import load_plot_config


def main():
    """Generate sensitivity comparison plot."""
    # Load matplotlib configuration
    load_plot_config()

    # Load sensitivity comparison results
    results_path = Path(__file__).parent.parent / "results" / "sensitivity_comparison_results.yaml"
    with open(results_path) as f:
        results = yaml.safe_load(f)

    # Dictionary to group sensitivities: {label: {depth: [sens1, sens2, ...]}}
    grouped_data = {}

    for _, exp_data in results.items():
        if not isinstance(exp_data, dict):
            continue

        depth = round(float(exp_data.get("Depth_mm", 0.0)) / 10, 1)
        sensitivity = exp_data.get("Optimized_Sensitivity")
        optimizer = str(exp_data.get("Optimizer", ""))

        if depth is None or sensitivity is None:
            continue

        # Determine label
        label = None
        if optimizer.startswith("DIGSSOptimizer"):
            if "normalization_scheme=unit_sum" in optimizer:
                label = "DIGSS(Unit Sum)"
            elif "normalization_scheme=unit_max" in optimizer:
                label = "DIGSS"
        elif optimizer.startswith("LiuOptimizer"):
            label = "Spectral Boxcar$^{[27]}$"
        elif optimizer.startswith("AltLiuOptimizer"):
            label = "Alt. Boxcar"
        elif optimizer.startswith("DummyUnitWindowGenerator"):
            label = "CW"
        elif optimizer.startswith("BoxCarOptimizer"):
            label = "Brute Force Boxcar"
        if label:
            if label not in grouped_data:
                grouped_data[label] = {}
            if depth not in grouped_data[label]:
                grouped_data[label][depth] = []
            grouped_data[label][depth].append(sensitivity)

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 4))

    # Process and plot each group
    # labels_to_plot = ["DIGSS", "Boxcar$^{[27]}$", "CW"]
    # offsets = [-0.02, 0.00, +0.02]

    labels_to_plot = ["DIGSS", "Spectral Boxcar$^{[27]}$", "Brute Force Boxcar", "CW"]
    offsets = [-0.03, -0.01, 0.01, 0.03]

    for label, offset in zip(labels_to_plot, offsets, strict=True):
        depths = sorted(grouped_data[label].keys())
        means = []
        stds = []
        for d in depths:
            sens_list = grouped_data[label][d]
            means.append(np.mean(sens_list))
            stds.append(np.std(sens_list))
        means = np.array(means)
        stds = np.array(stds)
        # Plot with error bars
        dz = 0.434 * stds / means
        upper = means * (10**dz - 1)
        lower = means * (1 - 10**(-dz))
        ax.errorbar(np.array(depths) + offset, means, yerr=[lower, upper], label=label, capsize=3)

    # Configure axes
    ax.set_xlabel("Fetal Depth (cm)")
    ax.set_ylabel("Selectivity $\\times$ SNR")
    ax.set_yscale("log")
    ax.legend(loc="upper right")
    ax.grid(True, which="both", ls="-", alpha=0.5)

    # Save figure
    figures_dir = Path(__file__).parent.parent / "figures"
    figures_dir.mkdir(exist_ok=True)

    fig.savefig(figures_dir / "sensitivity_comparison.pdf", format="pdf")
    fig.savefig(figures_dir / "sensitivity_comparison.svg", format="svg")

    print(f"Sensitivity comparison plots saved to {figures_dir}")
    # plt.show()


if __name__ == "__main__":
    main()
