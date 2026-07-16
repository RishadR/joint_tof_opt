"""
Plot the raw (unprocessed) vs post-processed flat-top optimized window from the DIGSS optimization loop.
"""

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from joint_tof_opt import (
    AdditiveGaussianToFModifier,
    ToFData,
    WindowSumWithAdditiveGaussianNoiseCalculator,
    generate_tof,
)
from joint_tof_opt.plotting import load_plot_config

# ponytail: sys.path hack to reuse DIGSSOptimizer from experiments/ instead of duplicating its ~150 lines here
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from optimize_loop_paper import DIGSSOptimizer  # noqa: E402

logger = logging.getLogger(__name__)


def run_experiment(file_idx: int = 5) -> tuple[DIGSSOptimizer, np.ndarray]:
    """Reproduce the single-experiment setup from optimize_loop_paper.main() and return the fitted experiment."""
    measurand = "abs"
    ppath_file = Path(f"./data/experiment_{file_idx:04d}.npz")
    tof_dataset_path = Path("./data") / f"generated_tof_set_{ppath_file.stem}.npz"
    gen_config: dict = yaml.safe_load(open("./experiments/tof_config.yaml"))
    noise_var = 100000.0
    generate_tof(ppath_file, gen_config, tof_dataset_path, True, True)
    tof_data = ToFData.from_npz(tof_dataset_path)
    modifier = AdditiveGaussianToFModifier(noise_var)
    modified_tof = modifier.modify(tof_data)
    noise_calc = WindowSumWithAdditiveGaussianNoiseCalculator(noise_var)
    experiment = DIGSSOptimizer(
        tof_data=modified_tof,
        measurand=measurand,
        noise_calc=noise_calc,
        fetal_f=gen_config["fetal_f"],
        normalize_reward=False,
        lr=0.1,
        filter_hw=0.01,
        patience=50,
        reg_type="l2",
        reg_weight=0.000,
        filter_type="psafe_same_width",
        normalization_scheme="unit_max",
    )
    experiment.optimize()
    tof_dataset_path.unlink()  # Remove the generated ToF dataset to save space
    return experiment, modified_tof.bin_edges.numpy()


def plot_flat_top_example(
    window: torch.Tensor,
    unprocessed_window: torch.Tensor,
    bin_edges: np.ndarray,
    fig_size: tuple[int, int] = (6, 2),
    filename: str = "flat_top_example",
) -> None:
    """
    Plot the post-processed window against the raw (unprocessed) window and save to a file.

    :param window: Final, post-processed window tensor.
    :param unprocessed_window: Raw normalized window tensor before smoothening/post-processing.
    :param bin_edges: Bin edges tensor.
    :param fig_size: Figure size for the plot. Defaults to (6, 4).
    :param filename: Filename to save the plots. Saves to ./figures/{filename}.svg and ./figures/{filename}.pdf
    """
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_centers_ns = np.round(bin_centers * 1e9, decimals=2)  # Convert to nanoseconds for readability

    load_plot_config()

    fig, ax = plt.subplots(figsize=fig_size)
    ax.plot(bin_centers_ns, unprocessed_window.detach().cpu().numpy(), marker="o", label="Unprocessed Window")
    ax.plot(bin_centers_ns, window.detach().cpu().numpy(), marker="o", label="Flat-Top Window")
    ax.set_xlabel("Bin Center (ns)")
    ax.set_ylabel("Bin Weight\n($W_b$)")
    ax.set_title("Flat-Top Window Post-Processing")
    ax.legend()
    plt.tight_layout()

    output_dir = Path("./figures/")
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{filename}.pdf"
    svg_path = output_dir / f"{filename}.svg"
    fig.savefig(pdf_path, format="pdf")
    fig.savefig(svg_path, format="svg")
    print(f"Saved PDF to: {pdf_path}")
    print(f"Saved SVG to: {svg_path}")

    plt.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    experiment, bin_edges = run_experiment()
    plot_flat_top_example(experiment.window, experiment.unprocessed_window, bin_edges)


if __name__ == "__main__":
    main()
