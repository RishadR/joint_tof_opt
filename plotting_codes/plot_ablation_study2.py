"""
Plot Sensitivity vs. Noise Standard Deviation for the four ablation variants at a fixed fetal depth.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from joint_tof_opt.misc import noisy_results_path
from joint_tof_opt.plotting import as_samples, legend_no_overlap, load_plot_config, resolve_results_path

VARIANTS = [
    (True,  True,  "Flat-Top Projection + Window Smoothen", 10, +0.02),
    (False, True,  "No Flat-Top Projection",           3, +0.01),
    (True,  False, "No Window Smoothen",                3, -0.01),
    (False, False, "Neither",                          3, -0.02),
]

FIXED_DEPTH_MM = 14.0

# ablation_study.py sweeps these noise standard deviations (results are stored as variance = std**2).
NOISE_STDS = [0.0, 5.0, 10.0, 15.0]
CANONICAL_VARIANCES = {std**2 for std in NOISE_STDS}


def _matches(optimizer_str: str, wpp: bool, slb: bool) -> bool:
    return f"use_window_post_process={wpp}" in optimizer_str and f"window_smoothening={slb}" in optimizer_str


def main():
    load_plot_config()

    base_results_path = Path(__file__).parent.parent / "results" / "ablation_results.yaml"
    results_path, inject_noise = resolve_results_path(base_results_path)
    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        return

    with open(results_path) as f:
        results = yaml.safe_load(f)

    # grouped_data[variant_idx][noise_std] = [sensitivity, ...]
    grouped_data: list[dict[float, list[float]]] = [{} for _ in VARIANTS]

    for _, exp_data in results.items():
        if not isinstance(exp_data, dict):
            continue
        if exp_data.get("Depth_mm") != FIXED_DEPTH_MM:
            continue
        noise_var = exp_data.get("noise_variance")
        sensitivity = exp_data.get("Optimized_Sensitivity")
        optimizer = str(exp_data.get("Optimizer", ""))
        if noise_var is None or sensitivity is None or noise_var not in CANONICAL_VARIANCES:
            continue
        noise_std = float(noise_var) ** 0.5
        for idx, (wpp, slb, _, _, _) in enumerate(VARIANTS):
            if _matches(optimizer, wpp, slb):
                grouped_data[idx].setdefault(noise_std, []).extend(as_samples(sensitivity))
                break

    # Split layout: narrow left panel for noiseless, wide right panel for log noise axis
    fig, (ax_nl, ax_log) = plt.subplots(
        1,
        2,
        gridspec_kw={"width_ratios": [1, 4]},
        sharey=True,
    )
    fig.subplots_adjust(wspace=0)

    log_ticks = set()
    for (_, _, label, zorder, _), data in zip(VARIANTS, grouped_data, strict=True):
        noise_stds = sorted(data.keys())
        noiseless_vals, log_x_vals, log_noise_vars = [], [], []
        for v in noise_stds:
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
    ax_log.set_xlabel("Noise Standard Deviation")
    ax_log.grid(True, which="both", ls="-", alpha=0.5)
    ax_log.spines["left"].set_visible(False)
    ax_log.tick_params(left=False)
    legend_no_overlap(ax_log, "upper right")
    fig.tight_layout()

    figures_dir = Path(__file__).parent.parent / "figures"
    figures_dir.mkdir(exist_ok=True)
    pdf_path = noisy_results_path(figures_dir / "ablation_study2.pdf", inject_noise)
    svg_path = noisy_results_path(figures_dir / "ablation_study2.svg", inject_noise)
    fig.savefig(pdf_path, format="pdf")
    fig.savefig(svg_path, format="svg")
    print(f"Ablation study plot saved to {pdf_path}")
    print(f"Ablation study plot saved to {svg_path}")


if __name__ == "__main__":
    main()
