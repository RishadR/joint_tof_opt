"""
Plot Sensitivity vs. Fetal Depth for different noise variances (Unit Max).
Expresses noise as a percentage of total input photons.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from joint_tof_opt.misc import figure_output_path
from joint_tof_opt.plotting import as_samples, legend_no_overlap, load_plot_config, resolve_results_path


def main():
    """Generate noise sensitivity comparison plot."""
    # Load matplotlib configuration
    load_plot_config()

    # Load total photon count from configuration
    with open(Path(__file__).parent.parent / "experiments" / "tof_config.yaml") as f:
        config = yaml.safe_load(f)
        total_photon_count = config.get("total_photon_count", 1e6)  # Default to 1e6 if not specified


    # Load sensitivity comparison results (noisy/noiseless file picked via evaluator_specs.yaml)
    base_results_path = Path(__file__).parent.parent / "results" / "noise_sensitivity_comparison_results.yaml"
    results_path, inject_noise = resolve_results_path(base_results_path)
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

        grouped_data.setdefault(noise_var, {}).setdefault(depth_cm, []).extend(as_samples(sensitivity))

    # Create figure
    fig, ax = plt.subplots()

    # Sort variances for consistent plotting
    sorted_vars = sorted(grouped_data.keys())

    for noise_var in sorted_vars[::2]:
        depths = sorted(grouped_data[noise_var].keys())
        means = []
        stds = []
        for d in depths:
            sens_list = grouped_data[noise_var][d]
            means.append(np.mean(sens_list))
            stds.append(np.std(sens_list))
        if noise_var == 0.0:
            label = "Noiseless"
        else:
            # label = f"Noise Var. : {noise_var:.0e}" # Use Variance as is
            label = f"Normalized Noise $\\sigma$ : {np.sqrt(noise_var) / total_photon_count:.2e}"

        means = np.array(means)
        stds = np.array(stds)
        dz = 0.434 * stds / means
        upper = means * (10**dz - 1)
        lower = means * (1 - 10 ** (-dz))
        ax.errorbar(depths, means, yerr=[lower, upper], label=label, capsize=3)

    # Configure axes
    ax.set_xlabel("Fetal Depth (cm)")
    ax.set_ylabel("Selectivity $\\times$ SNR")
    ax.set_yscale("log")
    legend_no_overlap(ax, "upper right")
    ax.grid(True, which="both", ls="-", alpha=0.5)

    # Save figure
    figures_dir = Path(__file__).parent.parent / "figures"
    figures_dir.mkdir(exist_ok=True)

    fig.savefig(figure_output_path(figures_dir / "noise_sensitivity_comparison.pdf", inject_noise), format="pdf")
    fig.savefig(figure_output_path(figures_dir / "noise_sensitivity_comparison.svg", inject_noise), format="svg")

    print(f"Noise sensitivity comparison plots saved to {figures_dir}")


if __name__ == "__main__":
    main()
