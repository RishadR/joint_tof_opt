"""
Plot Sensitivity vs. Fetal Depth for the four ablation variants of DIGSSOptimizer.
One line per variant: (use_window_post_process, use_snr_left_bound) ∈ {T,F}².
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from joint_tof_opt.plotting import load_plot_config

VARIANTS = [
    (True,  True,  "Flat-Top Projection + SNR bound", 10, +0.02),
    (False, True,  "No Flat-Top Projection",           3, +0.01),
    (True,  False, "No SNR left bound",                3, -0.01),
    (False, False, "Neither",                          3, -0.02),
]


def _matches(optimizer_str: str, wpp: bool, slb: bool) -> bool:
    return (
        f"use_window_post_process={wpp}" in optimizer_str
        and f"use_snr_left_bound={slb}" in optimizer_str
    )


def main():
    load_plot_config()

    results_path = Path(__file__).parent.parent / "results" / "ablation_results.yaml"
    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        return

    with open(results_path) as f:
        results = yaml.safe_load(f)

    NOISE_VAR = 1000.0

    grouped_data: list[dict[float, list[float]]] = [{} for _ in VARIANTS]

    for _, exp_data in results.items():
        if not isinstance(exp_data, dict):
            continue
        if exp_data.get("noise_variance") != NOISE_VAR:
            continue
        depth_mm = exp_data.get("Depth_mm")
        sensitivity = exp_data.get("Optimized_Sensitivity")
        optimizer = str(exp_data.get("Optimizer", ""))
        if depth_mm is None or sensitivity is None:
            continue
        depth_cm = round(float(depth_mm) / 10.0, 1)
        for idx, (wpp, slb, _, _, _) in enumerate(VARIANTS):
            if _matches(optimizer, wpp, slb):
                grouped_data[idx].setdefault(depth_cm, []).append(float(sensitivity))
                break

    fig, ax = plt.subplots(figsize=(7, 5))

    for (_, _, label, zorder, offset), data in zip(VARIANTS, grouped_data, strict=True):
        depths = sorted(data.keys())
        means = np.array([np.mean(data[d]) for d in depths])
        stds  = np.array([np.std(data[d])  for d in depths])
        dz = 0.434 * stds / means  # half-width in log10 space
        upper = means * (10**dz - 1)
        lower = means * (1 - 10**(-dz))
        ax.errorbar(np.array(depths) + offset, means, yerr=[lower, upper], label=label, capsize=3, zorder=zorder)

    ax.set_xlabel("Fetal Depth (cm)")
    ax.set_ylabel("Selectivity $\\times$ SNR")
    ax.set_yscale("log")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, which="both", ls="-", alpha=0.5)

    fig.tight_layout()

    figures_dir = Path(__file__).parent.parent / "figures"
    figures_dir.mkdir(exist_ok=True)
    fig.savefig(figures_dir / "ablation_study.pdf", format="pdf")
    fig.savefig(figures_dir / "ablation_study.svg", format="svg")
    print(f"Ablation study plot saved to {figures_dir}")


if __name__ == "__main__":
    main()
