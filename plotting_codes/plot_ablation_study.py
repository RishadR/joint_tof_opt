"""
Plot Sensitivity vs. Fetal Depth for the four ablation variants of DIGSSOptimizer.
One line per variant: (use_window_post_process, use_snr_left_bound) ∈ {T,F}².
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from joint_tof_opt import load_evaluator_specs
from joint_tof_opt.misc import figure_output_path
from joint_tof_opt.plotting import as_samples, legend_no_overlap, load_plot_config, resolve_results_path

# [is–flat_top?, has_snr_bound?, plot_z_oder, plot_horizontal_offset]
VARIANTS = [
    (True,  True,  "Flat-Top Projection + Window Smoothen", 10, +0.02),
    (False, True,  "No Flat-Top Projection",           3, +0.01),
    (True,  False, "No Window Smoothen",                3, -0.01),
    (False, False, "Neither",                          3, -0.02),
]


def _matches(optimizer_str: str, wpp: bool, slb: bool) -> bool:
    return (
        f"use_window_post_process={wpp}" in optimizer_str
        and f"window_smoothening={slb}" in optimizer_str
    )


def main():
    load_plot_config()

    base_results_path = Path(__file__).parent.parent / "results" / "ablation_results.yaml"
    results_path, inject_noise = resolve_results_path(base_results_path)
    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        return

    with open(results_path) as f:
        results = yaml.safe_load(f)

    # Fixed noise level to slice on: the currently tuned variance from evaluator_specs.yaml.
    eval_spec = load_evaluator_specs(Path("./experiments/evaluator_specs.yaml"))
    noise_var = eval_spec.instrument_noise_variance

    grouped_data: list[dict[float, list[float]]] = [{} for _ in VARIANTS]

    for _, exp_data in results.items():
        if not isinstance(exp_data, dict):
            continue
        if exp_data.get("noise_variance") != noise_var:
            continue
        depth_mm = exp_data.get("Depth_mm")
        sensitivity = exp_data.get("Optimized_Sensitivity")
        optimizer = str(exp_data.get("Optimizer", ""))
        if depth_mm is None or sensitivity is None:
            continue
        depth_cm = round(float(depth_mm) / 10.0, 1)
        for idx, (wpp, slb, _, _, _) in enumerate(VARIANTS):
            if _matches(optimizer, wpp, slb):
                grouped_data[idx].setdefault(depth_cm, []).extend(as_samples(sensitivity))
                break

    fig, ax = plt.subplots()

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
    legend_no_overlap(ax, "upper right")
    ax.grid(True, which="both", ls="-", alpha=0.5)

    fig.tight_layout()

    figures_dir = Path(__file__).parent.parent / "figures"
    figures_dir.mkdir(exist_ok=True)
    pdf_path = figure_output_path(figures_dir / "ablation_study.pdf", inject_noise)
    svg_path = figure_output_path(figures_dir / "ablation_study.svg", inject_noise)
    fig.savefig(pdf_path, format="pdf")
    fig.savefig(svg_path, format="svg")
    print(f"Ablation study plot saved to {pdf_path}")
    print(f"Ablation study plot saved to {svg_path}")


if __name__ == "__main__":
    main()
