#!/usr/bin/env python3
"""
Plot selectivity vs. fetal SNR for different SDD indices.
"""

import yaml
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from cycler import cycler


def main():
    """Generate detector comparison plot."""
    # Load matplotlib configuration
    config_path = Path(__file__).parent / "plot_config.yaml"
    with open(config_path, "r") as f:
        plot_config = yaml.safe_load(f)
        custom_cycler = (
            cycler(color=plot_config["plotting"]["colors"])
            +
            # Turning off line styles - makes it too messy
            #  cycler(linestyle=plot_config['plotting']['line_styles']) +
            cycler(marker=plot_config["plotting"]["markers"])
        )
        plt.rcParams["axes.prop_cycle"] = custom_cycler
        plot_config.pop("plotting", None)  # Remove custom plotting config from rcParams
        plt.rcParams.update(plot_config)

    # Load detector comparison results
    results_path = Path(__file__).parent.parent / "results" / "detector_comparison_results.yaml"
    with open(results_path, "r") as f:
        results = yaml.safe_load(f)

    # SDD distances in cm
    sdd_distances = [2, 1.3, 2.4, 3.5, 4.6, 5.6, 6.7, 7.8, 8.9]

    # Extract data for each SDD index (only DIGSS optimizer)
    sdd_data = {}

    for exp_key, exp_data in results.items():
        if not isinstance(exp_data, dict):
            continue

        depth = exp_data.get("Depth_mm")
        optimizer = exp_data.get("Optimizer", "")
        sdd_index = exp_data.get("SDD_Index")
        evaluator_log = exp_data.get("evaluator_log", {})

        if depth is None or sdd_index is None:
            continue

        # Only process DIGSS optimizer
        if not str(optimizer).startswith("DIGSS"):
            continue

        if not isinstance(evaluator_log, dict):
            continue

        baseline_noise_std = evaluator_log.get("baseline_noise_std")
        fetal_ac_energy = evaluator_log.get("fetal_ac_energy")
        maternal_ac_amp = evaluator_log.get("maternal_ac_amp")

        if baseline_noise_std is None or fetal_ac_energy is None or maternal_ac_amp is None:
            continue

        if baseline_noise_std <= 0 or maternal_ac_amp <= 0 or fetal_ac_energy < 0:
            continue

        fetal_ac_amp = np.sqrt(fetal_ac_energy)
        selectivity = fetal_ac_amp / maternal_ac_amp
        fetal_snr = fetal_ac_amp / baseline_noise_std

        # Skip the NaNs - I dont want to deal with them right now
        if np.isnan(selectivity) or np.isnan(fetal_snr) or np.isinf(selectivity) or np.isinf(fetal_snr):
            continue

        fetal_depth_cm = round(depth / 10.0, 1)

        if sdd_index not in sdd_data:
            sdd_data[sdd_index] = {"fetal_snr": [], "selectivity": [], "depth_cm": []}

        sdd_data[sdd_index]["fetal_snr"].append(fetal_snr)
        sdd_data[sdd_index]["selectivity"].append(selectivity)
        sdd_data[sdd_index]["depth_cm"].append(fetal_depth_cm)

    # Sort data by fetal depth (cm) and create arrays
    for sdd_index in sdd_data:
        if sdd_data[sdd_index]["depth_cm"]:
            sorted_indices = np.argsort(sdd_data[sdd_index]["depth_cm"])
            sdd_data[sdd_index]["fetal_snr"] = np.array(sdd_data[sdd_index]["fetal_snr"])[sorted_indices]
            sdd_data[sdd_index]["selectivity"] = np.array(sdd_data[sdd_index]["selectivity"])[sorted_indices]
            sdd_data[sdd_index]["depth_cm"] = np.array(sdd_data[sdd_index]["depth_cm"])[sorted_indices]

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 4))

    # I have way too make Fetal Depths. Only plotting alternate ones
    for sdd_index in sorted(sdd_data.keys()):
        sdd_data[sdd_index]["fetal_snr"] = sdd_data[sdd_index]["fetal_snr"][1::2]
        sdd_data[sdd_index]["selectivity"] = sdd_data[sdd_index]["selectivity"][1::2]
        sdd_data[sdd_index]["depth_cm"] = sdd_data[sdd_index]["depth_cm"][1::2]

    # Let's only plot the third and fifth SDD indices
    for sdd_index in sorted(sdd_data.keys()):
        if sdd_index not in [2, 3, 4]:
            continue
        sdd_distance = sdd_distances[sdd_index - 1]  # SDD_Index is 1-based
        fetal_snr = sdd_data[sdd_index]["fetal_snr"]
        selectivity = sdd_data[sdd_index]["selectivity"]
        depth_cm = sdd_data[sdd_index]["depth_cm"]

        (line,) = ax.plot(fetal_snr, selectivity, linewidth=2, markersize=8, label=f"SDD = {sdd_distance} cm")
        line_color = line.get_color()

        for idx, (x, y, d_cm) in enumerate(zip(fetal_snr, selectivity, depth_cm)):
            x_offset = 8
            y_offset = 8
            ax.annotate(
                f"{d_cm:.1f} cm",
                xy=(x, y),
                xytext=(x_offset, y_offset),
                textcoords="offset points",
                fontsize=8,
                ha="right",
                va="top",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.5),
            )

    # Configure axes
    ax.set_xlabel("Fetal SNR")
    ax.set_ylabel("Selectivity")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    # ax.set_ylim(top=1.3)

    # Save figure
    figures_dir = Path(__file__).parent.parent / "figures"
    figures_dir.mkdir(exist_ok=True)

    fig.savefig(figures_dir / "detector_comparison2.pdf", format="pdf")
    fig.savefig(figures_dir / "detector_comparison2.svg", format="svg")

    print(f"Detector comparison plots saved to {figures_dir}")
    plt.show()


if __name__ == "__main__":
    main()
