"""
Plot DIGSS sensitivity vs. fetal depth for unit_sum and unit_max normalization,
and percent improvement of unit_max over unit_sum.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from joint_tof_opt.plotting import load_plot_config


def main():
    """Generate sensitivity comparison plot for DIGSS normalization schemes."""
    # Load matplotlib configuration
    load_plot_config()

    # Load sensitivity comparison results
    results_path = Path(__file__).parent.parent / "results" / "sensitivity_comparison_results.yaml"
    with open(results_path, "r") as f:
        results = yaml.safe_load(f)

    # Collect DIGSS sensitivity values by depth for each normalization scheme
    unit_sum_by_depth = {}
    unit_max_by_depth = {}

    for _, exp_data in results.items():
        if not isinstance(exp_data, dict):
            continue

        depth_mm = exp_data.get("Depth_mm")
        sensitivity = exp_data.get("Optimized_Sensitivity")
        optimizer_str = str(exp_data.get("Optimizer", ""))

        if depth_mm is None or sensitivity is None:
            continue

        depth_cm = round(float(depth_mm) / 10.0, 1)

        if optimizer_str.startswith("DIGSSOptimizer"):
            if "normalization_scheme=unit_sum" in optimizer_str:
                unit_sum_by_depth[depth_cm] = float(sensitivity)
            elif "normalization_scheme=unit_max" in optimizer_str:
                unit_max_by_depth[depth_cm] = float(sensitivity)

    # Depths where both schemes are available
    common_depths = sorted(set(unit_sum_by_depth.keys()) & set(unit_max_by_depth.keys()))

    unit_sum_sens = [unit_sum_by_depth[d] for d in common_depths]
    unit_max_sens = [unit_max_by_depth[d] for d in common_depths]

    # Percent improvement of unit_max over unit_sum
    pct_improvement = [
        100.0 * (umax - usum) / usum if usum != 0 else np.nan
        for usum, umax in zip(unit_sum_sens, unit_max_sens)
    ]

    # Create figure and twin y-axis
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax2 = ax1.twinx()

    # Left axis: sensitivity curves
    line1, = ax1.plot(common_depths, unit_sum_sens, marker="o", label="DIGSS(Unit Sum)")
    line2, = ax1.plot(common_depths, unit_max_sens, marker="s", label="DIGSS(Unit Max)")

    # Right axis: percent improvement curve
    # Continue from the same global style cycle used by ax1 (instead of restarting on ax2).
    style_index = len(ax1.lines)
    cycle_values = plt.rcParams["axes.prop_cycle"].by_key()
    line3_style = {
        key: values[style_index % len(values)]
        for key, values in cycle_values.items()
        if values
    }

    line3, = ax2.plot(
        common_depths,
        pct_improvement,
        label="% Improvement (Unit Max over Unit Sum)",
        **line3_style,
    )

    # Configure axes
    ax1.set_xlabel("Fetal Depth (cm)")
    ax1.set_ylabel("Selectivity $\\times$ SNR")
    ax1.set_yscale("log")
    ax1.grid(True)

    ax2.set_ylabel("Improvement (%)")
    ax2.set_ylim(bottom=50, top=170)

    # Combined legend
    lines = [line1, line2, line3]
    labels = [str(ln.get_label()) for ln in lines]
    ax1.legend(lines, labels, loc="best")

    # Save figure
    figures_dir = Path(__file__).parent.parent / "figures"
    figures_dir.mkdir(exist_ok=True)

    fig.savefig(figures_dir / "sensitivity_comparison3.pdf", format="pdf")
    fig.savefig(figures_dir / "sensitivity_comparison3.svg", format="svg")

    print(f"Sensitivity comparison 3 plots saved to {figures_dir}")


if __name__ == "__main__":
    main()
