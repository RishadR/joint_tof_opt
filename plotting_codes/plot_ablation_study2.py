"""
Plot Sensitivity vs. Noise Variance for the four ablation variants at a fixed fetal depth.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from joint_tof_opt.plotting import load_plot_config

VARIANTS = [
    (True, True, "Flat-Top Projection + SNR bound", 4, +0.02),
    (False, True, "No Flat-Top Projection", 5, +0.01),
    (True, False, "No SNR left bound", 3, -0.01),
    (False, False, "Neither", 3, -0.02),
]

FIXED_DEPTH_MM = 14.0


def _matches(optimizer_str: str, wpp: bool, slb: bool) -> bool:
    return f"use_window_post_process={wpp}" in optimizer_str and f"use_snr_left_bound={slb}" in optimizer_str


def main():
    load_plot_config()

    results_path = Path(__file__).parent.parent / "results" / "ablation_results.yaml"
    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        return

    with open(results_path) as f:
        results = yaml.safe_load(f)

    # grouped_data[variant_idx][noise_var] = [sensitivity, ...]
    grouped_data: list[dict[float, list[float]]] = [{} for _ in VARIANTS]

    for _, exp_data in results.items():
        if not isinstance(exp_data, dict):
            continue
        if exp_data.get("Depth_mm") != FIXED_DEPTH_MM:
            continue
        noise_var = exp_data.get("noise_variance")
        sensitivity = exp_data.get("Optimized_Sensitivity")
        optimizer = str(exp_data.get("Optimizer", ""))
        if noise_var is None or sensitivity is None:
            continue
        for idx, (wpp, slb, _, _, _) in enumerate(VARIANTS):
            if _matches(optimizer, wpp, slb):
                grouped_data[idx].setdefault(float(noise_var), []).append(float(sensitivity))
                break

    # Split layout: narrow left panel for noiseless, wide right panel for log noise axis
    fig, (ax_nl, ax_log) = plt.subplots(
        1,
        2,
        figsize=(8, 5),
        gridspec_kw={"width_ratios": [1, 4]},
        sharey=True,
    )
    fig.subplots_adjust(wspace=0)

    log_ticks = set()
    for (_, _, label, zorder, _), data in zip(VARIANTS, grouped_data, strict=True):
        noise_vars = sorted(data.keys())
        noiseless_vals, log_x_vals, log_noise_vars = [], [], []
        for v in noise_vars:
            if v == 0.0:
                noiseless_vals.append(v)
            else:
                log_x_vals.append(v)
                log_noise_vars.append(v)
                log_ticks.add(v)

        def _errbars(vals_list):
            means = np.array([np.mean(data[v]) for v in vals_list])
            stds = np.array([np.std(data[v]) for v in vals_list])
            dz = 0.434 * stds / means
            return means, means * (10**dz - 1), means * (1 - 10 ** (-dz))

        if noiseless_vals:
            m, u, lo = _errbars(noiseless_vals)
            ax_nl.errorbar([0], m, yerr=[lo, u], capsize=3, zorder=zorder, label=label)

        if log_noise_vars:
            m, u, lo = _errbars(log_noise_vars)
            ax_log.errorbar(log_x_vals, m, yerr=[lo, u], capsize=3, zorder=zorder, label=label)

    # Left panel: noiseless point
    ax_nl.set_xticks([0])
    ax_nl.set_xticklabels(["Noiseless"])
    ax_nl.set_xlim(-0.5, 0.5)
    ax_nl.set_ylabel("Selectivity $\\times$ SNR")
    ax_nl.set_yscale("log")
    ax_nl.grid(True, which="both", ls="-", alpha=0.5)
    ax_nl.spines["right"].set_visible(False)

    # Dashed divider between panels
    ax_nl.axvline(x=0.5, color="gray", linestyle="--", linewidth=0.8, clip_on=False)

    # Right panel: log noise axis
    ax_log.set_xscale("log")
    sorted_ticks = sorted(log_ticks)
    ax_log.set_xticks(sorted_ticks)
    ax_log.set_xticklabels([f"{t:.4g}" for t in sorted_ticks])
    ax_log.set_xlabel("Noise Variance")
    ax_log.grid(True, which="both", ls="-", alpha=0.5)
    ax_log.spines["left"].set_visible(False)
    ax_log.tick_params(left=False)
    ax_log.legend(loc="best", fontsize=8)
    fig.tight_layout()

    figures_dir = Path(__file__).parent.parent / "figures"
    figures_dir.mkdir(exist_ok=True)
    fig.savefig(figures_dir / "ablation_study2.pdf", format="pdf")
    fig.savefig(figures_dir / "ablation_study2.svg", format="svg")
    print(f"Ablation study 2 plot saved to {figures_dir}")


if __name__ == "__main__":
    main()
