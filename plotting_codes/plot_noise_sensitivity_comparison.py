"""
Plot Sensitivity vs. Fetal Depth for different noise variances (Unit Max).
Expresses noise as a percentage of total input photons.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from joint_tof_opt.plotting import load_plot_config


def main():
    """Generate noise sensitivity comparison plot."""
    # Load matplotlib configuration
    load_plot_config()

    # Load sensitivity comparison results
    results_path = Path(__file__).parent.parent / "results" / "noise_sensitivity_comparison_results.yaml"
    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        return

    with open(results_path) as f:
        results = yaml.safe_load(f)

    # Dictionary to group sensitivities: {noise_var: {depth: [sens1, sens2, ...]}}
    grouped_data = {}
    num_bins = 0

    for _, exp_data in results.items():
        if not isinstance(exp_data, dict):
            continue

        optimizer = str(exp_data.get("Optimizer", ""))
        # Filter for Unit Max only
        if "normalization_scheme=unit_max" not in optimizer:
            continue

        depth_mm = exp_data.get("Depth_mm")
        sensitivity = exp_data.get("Optimized_Sensitivity")
        noise_var = exp_data.get("noise_variance")
        window = exp_data.get("Optimized_Window", [])

        if depth_mm is None or sensitivity is None or noise_var is None:
            continue

        if num_bins == 0:
            num_bins = len(window)

        depth_cm = round(float(depth_mm) / 10.0, 1)

        if noise_var not in grouped_data:
            grouped_data[noise_var] = {}
        if depth_cm not in grouped_data[noise_var]:
            grouped_data[noise_var][depth_cm] = []
        grouped_data[noise_var][depth_cm].append(float(sensitivity))

    # Create figure
    fig, ax = plt.subplots(figsize=(7, 5))

    # Total input photon count as specified by user
    TOTAL_PHOTONS = 1e9

    # Sort variances for consistent plotting
    sorted_vars = sorted(grouped_data.keys())

    for noise_var in sorted_vars[:4]:
        depths = sorted(grouped_data[noise_var].keys())
        means = []
        stds = []
        for d in depths:
            sens_list = grouped_data[noise_var][d]
            means.append(np.mean(sens_list))
            stds.append(np.std(sens_list))

        # Calculate noise as percentage of input photon count
        # Total Noise Std Dev / Total Photons * 100
        noise_pct = 100 * np.sqrt(noise_var) / TOTAL_PHOTONS
        label = f"{noise_pct:.2g}% noise"

        means = np.array(means)
        stds = np.array(stds)
        dz = 0.434 * stds / means
        upper = means * (10**dz - 1)
        lower = means * (1 - 10**(-dz))
        ax.errorbar(depths, means, yerr=[lower, upper], label=label, capsize=3)

    # Configure axes
    ax.set_xlabel("Fetal Depth (cm)")
    ax.set_ylabel("Selectivity $\\times$ SNR")
    ax.set_yscale("log")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, which="both", ls="-", alpha=0.5)

    # Save figure
    figures_dir = Path(__file__).parent.parent / "figures"
    figures_dir.mkdir(exist_ok=True)

    fig.savefig(figures_dir / "noise_sensitivity_comparison.pdf", format="pdf")
    fig.savefig(figures_dir / "noise_sensitivity_comparison.svg", format="svg")

    print(f"Noise sensitivity comparison plots saved to {figures_dir}")


if __name__ == "__main__":
    main()
