"""Plot optimized windows vs. bin-center time for selected optimizers and depths."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from joint_tof_opt.plotting import load_plot_config


def main():
    load_plot_config()
    results_path = Path(__file__).parent.parent / "results" / "sensitivity_comparison_results.yaml"
    with open(results_path) as f:
        results = yaml.safe_load(f)

    targets = {
        "DIGSS(Unit Sum)": lambda o: (
            o.startswith("DIGSSOptimizer") and "normalization_scheme=unit_sum" in o
        ),
        "DIGSS(Unit Max)": lambda o: (
            o.startswith("DIGSSOptimizer") and "normalization_scheme=unit_max" in o
        ),
        "BOxcar": lambda o: o.startswith("LiuOptimizer") and "harmonics=1" in o,
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

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    for ax, (name, rows) in zip(axes, windows.items(), strict=True):
        for depth, win, centers_ns in sorted(rows, key=lambda x: x[0])[:4]:
            n = min(len(win), len(centers_ns))
            ax.plot(centers_ns[:n], win[:n], label=f"{depth / 10:.1f} cm")
        ax.set_title(name)
        ax.set_xlabel("Bin Center (ns)")
        ax.grid(True)
        ax.legend(title="Depth")

    axes[0].set_ylabel("Optimized Window")
    fig.tight_layout()

    out = Path(__file__).parent.parent / "figures"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "sensitivity_comparison4.pdf", format="pdf")
    fig.savefig(out / "sensitivity_comparison4.svg", format="svg")
    print(f"Saved: {out / 'sensitivity_comparison4.pdf'} and {out / 'sensitivity_comparison4.svg'}")


if __name__ == "__main__":
    main()
