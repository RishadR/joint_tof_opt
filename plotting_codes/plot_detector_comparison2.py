#!/usr/bin/env python3
"""
Plot selectivity vs. fetal SNR for different SDD indices.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import yaml
from adjustText import adjust_text
from cycler import cycler
from joint_tof_opt.plotting import load_plot_config


def main():
    """Generate detector comparison plot."""
    # Load matplotlib configuration
    load_plot_config()

    # Load detector comparison results
    results_path = Path(__file__).parent.parent / "results" / "detector_comparison_results.yaml"
    with open(results_path, "r") as f:
        results = yaml.safe_load(f)

    # SDD distances in cm
    sdd_distances = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

    # Extract data for each SDD index (only DIGSS optimizer)
    sdd_data = {}

    for _, exp_data in results.items():
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
    annotation_points = []

    for sdd_index in sorted(sdd_data.keys()):
        if sdd_index not in [2, 3, 4]:
            continue
        sdd_distance = sdd_distances[sdd_index - 1]  # SDD_Index is 1-based
        fetal_snr = sdd_data[sdd_index]["fetal_snr"]
        selectivity = sdd_data[sdd_index]["selectivity"]
        depth_cm = sdd_data[sdd_index]["depth_cm"]

        ax.plot(fetal_snr, selectivity, linewidth=2, markersize=8, label=f"SDD = {sdd_distance} cm")

        for x, y, d_cm in zip(fetal_snr, selectivity, depth_cm, strict=True):
            annotation_points.append((x, y, d_cm, sdd_index))

    # Configure axes
    ax.set_xlabel("Fetal SNR")
    ax.set_ylabel("Selectivity")
    ax.set_yscale("log")
    ax.set_xscale("log")

    # Add a bit of breathing room and place labels with bounded offsets.
    ax.margins(x=0.12, y=0.18)

    x_anchor = []
    y_anchor = []
    text_artists = []

    for x, y, d_cm, sdd_index in annotation_points:
        # Tiny deterministic offsets avoid identical starting positions.
        x_scale = 1.012 + 0.002 * (sdd_index - 2)
        y_scale = 1.01 + 0.002 * (sdd_index - 2)
        x_text = x * x_scale
        y_text = y * y_scale

        text_artists.append(
            ax.text(
                x_text,
                y_text,
                f"{d_cm:.1f} cm",
                fontsize=8,
                ha="left",
                va="bottom",
                clip_on=True,
                bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.5},
            )
        )
        x_anchor.append(x)
        y_anchor.append(y)

    fig.canvas.draw()
    adjust_text(
        text_artists,
        x=x_anchor,
        y=y_anchor,
        ax=ax,
        ensure_inside_axes=True,
        expand_axes=False,
        only_move={"text": "xy", "static": "xy", "explode": "xy", "pull": "xy"},
        force_text=(0.12, 0.2),
        force_static=(0.08, 0.14),
        force_pull=(0.02, 0.04),
        arrowprops={"arrowstyle": "-", "color": "0.5", "lw": 0.45, "alpha": 0.55},
    )

    ax.legend(loc="lower right")
    ax.grid(True)
    # ax.set_ylim(top=1.3)

    # Save figure
    figures_dir = Path(__file__).parent.parent / "figures"
    figures_dir.mkdir(exist_ok=True)

    fig.savefig(figures_dir / "detector_comparison2.pdf", format="pdf")
    fig.savefig(figures_dir / "detector_comparison2.svg", format="svg")

    print(f"Detector comparison plots saved to {figures_dir}")


if __name__ == "__main__":
    main()
