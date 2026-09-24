"""
Plot FoM vs. Fetal Depth for different optimizers. (Check src/optimizers)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from joint_tof_opt.misc import noisy_results_path
from joint_tof_opt.plotting import as_samples, legend_no_overlap, load_plot_config, resolve_results_path


def main():
    """Generate sensitivity comparison plot."""
    # Load matplotlib configuration
    load_plot_config()

    # Load sensitivity comparison results (noisy/noiseless file picked via evaluator_specs.yaml)
    base_results_path = Path(__file__).parent.parent / "results" / "sensitivity_comparison_results.yaml"
    results_path, inject_noise = resolve_results_path(base_results_path)
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
            grouped_data[label][depth].extend(as_samples(sensitivity))

    # Create figure
    fig, ax = plt.subplots()

    # Process and plot each group
    # labels_to_plot = ["DIGSS", "Boxcar$^{[27]}$", "CW"]
    # offsets = [-0.02, 0.00, +0.02]

    labels_to_plot = ["DIGSS", "Spectral Boxcar$^{[27]}$", "Brute Force Boxcar", "CW"]
    offsets = [-0.03, -0.01, 0.01, 0.03]

    for label, offset in zip(labels_to_plot, offsets, strict=True):
        depths = sorted(grouped_data[label].keys())
        means = np.array([np.mean(grouped_data[label][d]) for d in depths])

        yerr = None
        # Modify error bars to fit log-scale
        if inject_noise:
            stds = np.array([np.std(grouped_data[label][d]) for d in depths])
            dz = 0.434 * stds / means
            upper = means * (10**dz - 1)
            lower = means * (1 - 10**(-dz))
            yerr = [lower, upper]

        ax.errorbar(np.array(depths) + offset, means, yerr=yerr, label=label, capsize=3 if inject_noise else 0)

    # Configure axes
    ax.set_xlabel("Fetal Depth (cm)")
    ax.set_ylabel("Selectivity $\\times$ SNR")
    ax.set_yscale("log")
    legend_no_overlap(ax, "upper right")
    ax.grid(True, which="both", ls="-", alpha=0.5)

    # Save figure
    figures_dir = Path(__file__).parent.parent / "figures"
    figures_dir.mkdir(exist_ok=True)

    fig.savefig(noisy_results_path(figures_dir / "sensitivity_comparison.pdf", inject_noise), format="pdf")
    fig.savefig(noisy_results_path(figures_dir / "sensitivity_comparison.svg", inject_noise), format="svg")

    print(f"Sensitivity comparison plots saved to {figures_dir}")
    # plt.show()


if __name__ == "__main__":
    main()
