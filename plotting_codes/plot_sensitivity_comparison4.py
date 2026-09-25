"""Plot optimized windows vs. bin-center time for selected optimizers and depths."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from joint_tof_opt.misc import figure_output_path
from joint_tof_opt.plotting import legend_no_overlap, load_plot_config, resolve_results_path


def main():
    load_plot_config()
    base_results_path = Path(__file__).parent.parent / "results" / "sensitivity_comparison_results.yaml"
    results_path, inject_noise = resolve_results_path(base_results_path)
    with open(results_path) as f:
        results = yaml.safe_load(f)

    targets = {
        # "DIGSS(Unit Sum)": lambda o: (
        #     o.startswith("DIGSSOptimizer") and "normalization_scheme=unit_sum" in o
        # ),
        "DIGSS": lambda o: (
            o.startswith("DIGSSOptimizer") and "normalization_scheme=unit_max" in o
        ),
        "Spectral Boxcar": lambda o: o.startswith("LiuOptimizer") and "harmonics=2" in o,
    }
    windows = {k: [] for k in targets}

    for exp in results.values():
        if not isinstance(exp, dict):
            continue
        depth = exp.get("Depth_mm")
        win = exp.get("Optimized_Window")
        edges = exp.get("Bin_Edges")
        opt = str(exp.get("Optimizer", ""))
        if depth is None or win is None or edges is None:
            continue
        centers_ns = ((np.asarray(edges[:-1]) + np.asarray(edges[1:])) * 0.5 * 1e9).tolist()
        for name, match in targets.items():
            if match(opt):
                windows[name].append((float(depth), win, centers_ns))
                break

    fig, axes = plt.subplots(1, len(targets), sharey=True)
    for ax, (name, rows) in zip(axes, windows.items(), strict=True):
        seen_depths = {}
        for depth, win, centers_ns in rows:
            seen_depths.setdefault(depth, (depth, win, centers_ns))
        for depth, win, centers_ns in sorted(seen_depths.values(), key=lambda x: x[0])[:4]:
            n = min(len(win), len(centers_ns))
            ax.plot(centers_ns[:n], win[:n], label=f"{depth / 10:.1f} cm")
        ax.set_title(name)
        ax.set_xlabel("Bin Center (ns)")
        ax.grid(True)
        legend_no_overlap(ax, "lower right", title="Depth")

    axes[0].set_ylabel("Optimized Window")
    fig.tight_layout()

    out = Path(__file__).parent.parent / "figures"
    out.mkdir(exist_ok=True)
    pdf_path = figure_output_path(out / "sensitivity_comparison4.pdf", inject_noise)
    svg_path = figure_output_path(out / "sensitivity_comparison4.svg", inject_noise)
    fig.savefig(pdf_path, format="pdf")
    fig.savefig(svg_path, format="svg")
    print(f"Saved: {pdf_path} and {svg_path}")


if __name__ == "__main__":
    main()
