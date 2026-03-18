"""
What happens when Fetal and Maternal HR are overlapping/close? How do the two evaluators compare?
"""
from pathlib import Path
from collections import defaultdict

import yaml
import matplotlib.pyplot as plt
from cycler import cycler

from joint_tof_opt.plotting import load_plot_config


def _combo_label(filter_type: str, filter_hw: float) -> str:
    if filter_type == "psafe_same_width":
        return "TSA"
    return f"{filter_type} (HW={filter_hw:g})"


def main(
    input_yaml: Path = Path("./results/overlap_results.yaml"),
    output_base: Path = Path("./figures/overlap_compare"),
) -> None:
    # Load matplotlib configuration (same implementation as plot_detector_comparison.py)
    load_plot_config()
    with open(input_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    grouped_s1 = defaultdict(list)
    grouped_s2 = defaultdict(list)
    grouped_diff = defaultdict(list)

    for _, entry in data.items():
        sep = float(entry["Separation_Hz"])
        hw = float(entry["Filter_HW"])
        ftype = str(entry["Filter_Type"])
        s1 = float(entry["Sensitivity1"])   # FoM 
        s2 = float(entry["Sensitivity2"])   # Reward Metric
        # s2 = float(entry["Optimizer Best Metric"])
        # s2 = float(entry["Optimizer Best Selectivity"])
        # s2 = float(entry["Optimizer Best SNR"])
        grouped_s1[(ftype, hw)].append((sep, s1))
        grouped_s2[(ftype, hw)].append((sep, s2))
        grouped_diff[(ftype, hw)].append((sep, abs(s1 - s2)))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True, sharey=False)

    for (ftype, hw) in sorted(grouped_s1.keys(), key=lambda k: (k[0], k[1])):
        label = _combo_label(ftype, hw)

        points1 = sorted(grouped_s1[(ftype, hw)], key=lambda t: t[0])
        x1 = [p[0] for p in points1]
        y1 = [p[1] for p in points1]
        axes[0].plot(x1, y1, linewidth=2, markersize=8, label=label)

        points2 = sorted(grouped_s2[(ftype, hw)], key=lambda t: t[0])
        x2 = [p[0] for p in points2]
        y2 = [p[1] for p in points2]
        axes[1].plot(x2, y2, linewidth=2, markersize=8, label=label)

    axes[0].set_xlabel("Fetal Fundemental & Maternal 2nd Harmonic Separation (Hz)")
    axes[0].set_ylabel("Figure of Merit(FoM)")
    axes[0].legend(title="Filter Setup")
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Fetal Fundemental & Maternal 2nd Harmonic Separation (Hz)")
    axes[1].set_ylabel("Reward Metric")
    axes[1].legend(title="Filter Setup")
    axes[1].grid(True, alpha=0.3)

    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".pdf"), format="pdf")
    fig.savefig(output_base.with_suffix(".svg"), format="svg")

    fig_alt, ax_alt = plt.subplots(1, 1, figsize=(6, 4), sharex=True, sharey=False)
    # Normalize the differences by the max FoM value across all points to get a relative difference
    max_fom = max(max(y1 for _, y1 in points) for points in grouped_s1.values())
    for (ftype, hw) in sorted(grouped_diff.keys(), key=lambda k: (k[0], k[1])):
        label = _combo_label(ftype, hw)
        points = sorted(grouped_diff[(ftype, hw)], key=lambda t: t[0])
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        y = [diff / max_fom for diff in y]  # Normalize by max FoM to get relative difference
        ax_alt.plot(x, y, linewidth=2, markersize=8, label=label)

    ax_alt.set_xlabel("Fetal Fundemental & Maternal 2nd Harmonic Separation (Hz)")
    ax_alt.set_ylabel("Normalized |FoM - Reward Metric|")
    ax_alt.legend(title="Filter Setup")
    ax_alt.grid(True)

    output_alt_base = output_base.with_name(f"{output_base.name}_alt")
    fig_alt.tight_layout()
    fig_alt.savefig(output_alt_base.with_suffix(".pdf"), format="pdf")
    fig_alt.savefig(output_alt_base.with_suffix(".svg"), format="svg")


if __name__ == "__main__":
    main()
