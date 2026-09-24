#!/usr/bin/env python3
"""
Plot Optimized Sensitivity vs. Fetal Depth for different SDD indices (Only for DIGSS)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from joint_tof_opt.misc import noisy_results_path
from joint_tof_opt.plotting import as_samples, legend_no_overlap, load_plot_config, resolve_results_path


def main():
    """Generate detector comparison plot."""
    # Load matplotlib configuration
    load_plot_config()

    # Load detector comparison results (noisy/noiseless file picked via evaluator_specs.yaml)
    base_results_path = Path(__file__).parent.parent / "results" / "detector_comparison_results.yaml"
    results_path, inject_noise = resolve_results_path(base_results_path)
    with open(results_path) as f:
        results = yaml.safe_load(f)

    # SDD distances in mm
    sdd_distances = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

    # Extract data for each SDD index (only DIGSS optimizer): {sdd_index: {depth_cm: [sens1, sens2, ...]}}
    sdd_data: dict[int, dict[float, list[float]]] = {}

    for exp_key, exp_data in results.items():
        if not isinstance(exp_data, dict):
            continue

        depth = exp_data.get("Depth_mm")
        sensitivity = exp_data.get("Optimized_Sensitivity")
        optimizer = exp_data.get("Optimizer", "")
        sdd_index = exp_data.get("SDD_Index")

        if depth is None or sensitivity is None or sdd_index is None:
            continue
        depth = round(depth / 10, 1)  # Convert to cm

        # Only process DIGSS optimizer
        if "DIGSSOptimizer" not in str(optimizer):
            continue

        sdd_data.setdefault(sdd_index, {}).setdefault(depth, []).extend(as_samples(sensitivity))

    # Create figure
    fig, ax = plt.subplots()

    # Plot each SDD index
    for idx, sdd_index in enumerate(sorted(sdd_data.keys())):
        # Too many options - let's only plot alternate ones to avoid clutter
        if idx not in [1, 2, 3, 4]:
            continue
        sdd_distance = sdd_distances[sdd_index - 1]  # SDD_Index is 1-based
        depths = sorted(sdd_data[sdd_index].keys())
        means = np.array([np.mean(sdd_data[sdd_index][d]) for d in depths])
        ax.plot(
            depths,
            means,
            linewidth=2,
            markersize=8,
            label=f"SDD = {round(sdd_distance / 10, 1)} cm",
        )

    # Configure axes
    ax.set_xlabel("Fetal Depth (cm)")
    ax.set_ylabel("Figure of Merit")
    ax.set_yscale("log")
    legend_no_overlap(ax, "upper right")
    ax.grid(True)
    # ax.set_ylim(top=1.3)

    # Save figure
    figures_dir = Path(__file__).parent.parent / "figures"
    figures_dir.mkdir(exist_ok=True)

    fig.savefig(noisy_results_path(figures_dir / "detector_comparison.pdf", inject_noise), format="pdf")
    fig.savefig(noisy_results_path(figures_dir / "detector_comparison.svg", inject_noise), format="svg")

    print(f"Detector comparison plots saved to {figures_dir}")


if __name__ == "__main__":
    main()
