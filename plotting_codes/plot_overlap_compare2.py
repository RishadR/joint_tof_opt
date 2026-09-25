from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from joint_tof_opt.misc import figure_output_path
from joint_tof_opt.plotting import as_samples, legend_no_overlap, load_plot_config, resolve_results_path


def _combo_label(filter_type: str, filter_hw: float) -> str:
    if filter_type == "psafe_same_width":
        return "TSA"
    return f"{filter_type} (HW={filter_hw:g})"


def _mean(value: float | list[float]) -> float:
    return float(np.mean(as_samples(value)))


def main(
    input_yaml: Path | None = None,
    output_base: Path | None = None,
) -> None:
    # Load matplotlib configuration (same implementation as plot_detector_comparison.py)
    load_plot_config()
    fig_size_x = plt.rcParams.get("figure.figsize", [6, 4])[0]
    fig_size_y = plt.rcParams.get("figure.figsize", [6, 4])[1]

    # Resolve the noisy/noiseless input file (and matching output prefix) via evaluator_specs.yaml.
    resolved_input, inject_noise = resolve_results_path(input_yaml or Path("./results/overlap_results2.yaml"))
    output_base = figure_output_path(output_base or Path("./figures/overlap_compare2"), inject_noise)

    with open(resolved_input, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    grouped_s1 = defaultdict(list)
    grouped_s2 = defaultdict(list)
    grouped_diff = defaultdict(list)

    for _, entry in data.items():
        depth = float(entry["Depth_mm"])
        hw = float(entry["Filter_HW"])
        ftype = str(entry["Filter_Type"])
        s1 = _mean(entry["Sensitivity1"])
        s2 = _mean(entry["Sensitivity2"])
        grouped_s1[(ftype, hw)].append((depth, s1))
        grouped_s2[(ftype, hw)].append((depth, s2))
        grouped_diff[(ftype, hw)].append((depth, abs(s1 - s2)))

    fig, axes = plt.subplots(1, 2, sharex=True, sharey=False, figsize=(fig_size_x * 1.5, fig_size_y))

    for (ftype, hw) in sorted(grouped_s1.keys(), key=lambda k: (k[0], k[1])):
        label = _combo_label(ftype, hw)

        points1 = sorted(grouped_s1[(ftype, hw)], key=lambda t: t[0])
        x1 = [p[0] for p in points1]
        y1 = [p[1] for p in points1]
        axes[0].plot(x1, y1, label=label)

        points2 = sorted(grouped_s2[(ftype, hw)], key=lambda t: t[0])
        x2 = [p[0] for p in points2]
        y2 = [p[1] for p in points2]
        axes[1].plot(x2, y2, label=label)

    axes[0].set_xlabel("Fetal Depth (mm)")
    axes[0].set_ylabel("Figure of Merit(FoM)")
    legend_no_overlap(axes[0], "upper right", title="Filter Setup")
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Fetal Depth (mm)")
    axes[1].set_ylabel("Reward Metric")
    legend_no_overlap(axes[1], "upper right", title="Filter Setup")
    axes[1].grid(True, alpha=0.3)

    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".pdf"), format="pdf")
    fig.savefig(output_base.with_suffix(".svg"), format="svg")

    fig_alt, ax_alt = plt.subplots(1, 1, sharex=True, sharey=False)

    for (ftype, hw) in sorted(grouped_diff.keys(), key=lambda k: (k[0], k[1])):
        label = _combo_label(ftype, hw)
        points = sorted(grouped_diff[(ftype, hw)], key=lambda t: t[0])
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        ax_alt.plot(x, y, label=label)

    ax_alt.set_xlabel("Fetal Depth (mm)")
    ax_alt.set_ylabel("|FoM - Reward Metric|")
    legend_no_overlap(ax_alt, "upper right", title="Filter Setup")
    ax_alt.grid(True, alpha=0.3)

    output_alt_base = output_base.with_name(f"{output_base.name}_alt")
    fig_alt.tight_layout()
    fig_alt.savefig(output_alt_base.with_suffix(".pdf"), format="pdf")
    fig_alt.savefig(output_alt_base.with_suffix(".svg"), format="svg")


if __name__ == "__main__":
    main()
