"""
Plot a sample time-of-flight (TOF) spectrum using matplotlib.
"""
from typing import Literal
import numpy as np
import matplotlib.pyplot as plt
import yaml
from pathlib import Path
import re
from scipy.interpolate import UnivariateSpline
from joint_tof_opt.tof_batch_process import compute_tof_data_series, ToFData


def load_plot_config():
    """Load matplotlib configuration from YAML file."""
    config_path = "./plotting_codes/plot_config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_color_cycle(config) -> list[str]:
    # Extract colors from the prop_cycle
    colors_str = config["axes.prop_cycle"]
    # Parse the cycler string to extract hex colors
    colors = [color.strip("'\"") for color in colors_str.split("color")[1].split("[")[1].split("]")[0].split(",")]
    colors = [c.strip() for c in colors]
    # Add '#' prefix to make them valid hex colors for matplotlib
    colors = ['#' + c if not c.startswith('#') else c for c in colors]
    return colors


def load_gen_config():
    config_path = "./experiments/tof_config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def plot_sample_tof(ppath: Path, plot_type: Literal["distribution", "density"]):
    """
    Plot a sample time-of-flight histogram.

    Parameters:
    -----------
    ppath : str or Path
        Path to the .npz to the partial path table
    plot_type : str
        Type of plot: 'distribution' or 'density'
    """
    gen_config = load_gen_config()
    gen_config["datapoint_count"] = 2  # Only generate 2 ToFs for quick loading

    # Load data
    tof_data = compute_tof_data_series(ppath, gen_config, True, True)

    # Get first row (first time point)
    tof_histogram = tof_data.tof_series.numpy()[0, :]

    # Convert from seconds to nanoseconds
    bin_edges_ns = tof_data.bin_edges.numpy() * 1e9

    # Calculate bin centers for the line plot
    bin_centers = (bin_edges_ns[:-1] + bin_edges_ns[1:]) / 2

    # Calculate bin widths
    bin_widths = np.diff(bin_edges_ns)

    # Normalize to density if requested
    if plot_type == "density":
        # Density: histogram / (sum * bin_width)
        y_values = tof_histogram / (np.sum(tof_histogram) * bin_widths)
        ylabel = "Probability Density"
    else:
        # Distribution: raw counts
        y_values = tof_histogram
        ylabel = "Count"
    
    # Create smooth interpolated curve
    spline = UnivariateSpline(bin_centers, y_values, s=0.5, k=3)
    x_smooth = np.linspace(bin_centers[0], bin_centers[-1], 300)
    y_smooth = spline(x_smooth)

    # Load plot configuration
    plot_config = load_plot_config()

    # Apply rcParams
    plt.rcParams.update(plot_config)
    # Get first color from the color cycle
    bar_color = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]
    line_color = plt.rcParams['axes.prop_cycle'].by_key()['color'][1]

    # Create figure
    fig, ax = plt.subplots(figsize=(4, 3))

    # Plot bars with black edge
    ax.bar(
        bin_centers,
        y_values,
        width=bin_widths,
        color=bar_color,
        edgecolor="black",
        linewidth=0.5,
        alpha=0.7,
        label="Histogram",
    )

    # Plot smooth line connecting bar centers
    ax.plot(
        x_smooth,
        y_smooth,
        color=line_color,
        linewidth=2,
    )

    # Labels and title
    ax.set_xlabel("Time of Flight (ns)")
    ax.set_ylabel(ylabel)
    ax.set_title("Sample Time-of-Flight Distribution")
    # ax.legend()
    ax.grid(True, alpha=0.1)

    # Create output directory if it doesn't exist
    output_dir = Path("./figures/")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save figures
    base_name = f"sample_tof_{plot_type}"
    pdf_path = output_dir / f"{base_name}.pdf"
    svg_path = output_dir / f"{base_name}.svg"

    fig.savefig(pdf_path, format="pdf")
    fig.savefig(svg_path, format="svg")

    print(f"Saved PDF to: {pdf_path}")
    print(f"Saved SVG to: {svg_path}")

    plt.close()


def main(data_path=Path("data/experiment_0000.npz"), plot_type: Literal["distribution", "density"]="distribution"):
    """Main function to run the plotting script.

    Parameters:
    -----------
    data_path : Path
        Path to the data file
    output_dir : Path
        Output directory for figures
    plot_type : Literal["distribution", "density"]
        Plot type: 'distribution' (raw counts) or 'density' (normalized)
    """
    plot_sample_tof(data_path, plot_type)


if __name__ == "__main__":
    main()
