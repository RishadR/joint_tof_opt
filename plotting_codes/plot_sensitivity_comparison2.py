"""
Plot Selectivity vs. SNR for different optimizers at various fetal depths.
Compares DIGSS, Liu et al., and CW methods.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from joint_tof_opt.plotting import load_plot_config


def main():
    """Generate Selectivity vs. SNR scatter plot."""
    # Configuration: Control which depth points to annotate (step size)
    # Set to 1 to annotate all points, 2 for every other point, 3 for every third, etc.
    ANNOTATION_STEP = 3  # Annotate every Nth point for all methods

    # Load matplotlib configuration
    load_plot_config()

    # Load sensitivity comparison results
    results_path = (Path(__file__).parent.parent / "results" / "sensitivity_comparison_results.yaml")
    with open(results_path, "r") as f:
        results = yaml.safe_load(f)

    # Extract data for each optimizer
    digss_data = {"snr": [], "selectivity": [], "depths": []}
    liu_data_h1 = {"snr": [], "selectivity": [], "depths": []}
    liu_data_h2 = {"snr": [], "selectivity": [], "depths": []}
    alt_liu_data_h1 = {"snr": [], "selectivity": [], "depths": []}
    alt_liu_data_h2 = {"snr": [], "selectivity": [], "depths": []}
    cw_data = {"snr": [], "selectivity": [], "depths": []}

    for exp_key, exp_data in results.items():
        if not isinstance(exp_data, dict):
            continue

        depth = exp_data.get("Depth_mm")
        optimizer = exp_data.get("Optimizer", "")
        evaluator_log = exp_data.get("evaluator_log", {})

        # Extract required values from evaluator_log
        fetal_ac_energy = evaluator_log.get("fetal_ac_energy")
        maternal_ac_energy = evaluator_log.get("maternal_ac_energy")
        baseline_noise_std = evaluator_log.get("baseline_noise_std")

        if (
            depth is None
            or fetal_ac_energy is None
            or maternal_ac_energy is None
            or baseline_noise_std is None
        ):
            continue

        # Compute Selectivity and SNR
        selectivity = np.sqrt(fetal_ac_energy / maternal_ac_energy)
        snr = np.sqrt(fetal_ac_energy) / baseline_noise_std

        # Store data based on optimizer type
        if str(optimizer).startswith("DIGSSOptimizer"):
            digss_data["snr"].append(snr)
            digss_data["selectivity"].append(selectivity)
            digss_data["depths"].append(depth)
        elif str(optimizer).startswith("LiuOptimizer"):
            if "harmonics=1" in str(optimizer):
                liu_data_h1["snr"].append(snr)
                liu_data_h1["selectivity"].append(selectivity)
                liu_data_h1["depths"].append(depth)
            elif "harmonics=2" in str(optimizer):
                liu_data_h2["snr"].append(snr)
                liu_data_h2["selectivity"].append(selectivity)
                liu_data_h2["depths"].append(depth)
        elif str(optimizer).startswith("AltLiuOptimizer"):
            if "harmonics=1" in str(optimizer):
                alt_liu_data_h1["snr"].append(snr)
                alt_liu_data_h1["selectivity"].append(selectivity)
                alt_liu_data_h1["depths"].append(depth)
            elif "harmonics=2" in str(optimizer):
                alt_liu_data_h2["snr"].append(snr)
                alt_liu_data_h2["selectivity"].append(selectivity)
                alt_liu_data_h2["depths"].append(depth)
        elif str(optimizer).startswith("DummyUnitWindowGenerator"):
            cw_data["snr"].append(snr)
            cw_data["selectivity"].append(selectivity)
            cw_data["depths"].append(depth)

    # Filter data to only include every Nth point
    def filter_data(data, step):
        """Filter data dictionary to only include every Nth point."""
        filtered = {key: [] for key in data.keys()}
        for i in range(0, len(data["snr"]), step):
            for key in data.keys():
                filtered[key].append(data[key][i])
        return filtered

    def plot_method_with_annotations(ax, data, label):
        """Plot a method's data with depth annotations."""
        if not data["snr"]:
            return

        ax.plot(data["snr"], data["selectivity"], label=label)
        # Add depth annotations
        for i, depth in enumerate(data["depths"]):
            depth_cm = depth / 10  # Convert mm to cm
            ax.annotate(
                f"{depth_cm:.1f} cm",
                (data["snr"][i], data["selectivity"][i]),
                textcoords="offset points",
                xytext=(5, 5),
                ha="left",
                fontsize=7,
                alpha=0.7,
                bbox=dict(
                    boxstyle="round,pad=0.15", fc="lightgray", ec="none", alpha=0.7
                ),
            )

    digss_filtered = filter_data(digss_data, ANNOTATION_STEP)
    liu_filtered = filter_data(liu_data_h1, ANNOTATION_STEP)
    cw_filtered = filter_data(cw_data, ANNOTATION_STEP)

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 4))

    # Plot each optimizer with lines connecting points
    plot_method_with_annotations(ax, digss_filtered, "DIGSS")
    plot_method_with_annotations(ax, liu_filtered, "Boxcar$^{[27]}$")
    plot_method_with_annotations(ax, cw_filtered, "CW")

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
