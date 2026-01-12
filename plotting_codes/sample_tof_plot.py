"""
Plot a sample time-of-flight (TOF) spectrum using matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt
import yaml
from pathlib import Path
import re


def load_plot_config(config_path):
    """Load matplotlib configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def plot_sample_tof(data_path, plot_type="distribution"):
    """
    Plot a sample time-of-flight histogram.

    Parameters:
    -----------
    data_path : str or Path
        Path to the .npz file containing the TOF data
    plot_type : str
        Type of plot: 'distribution' or 'density'
    """
    # Load data
    data = np.load(data_path)
    tof_dataset = data["tof_dataset"]
    bin_edges = data["bin_edges"]

    # Get first row (first time point)
    tof_histogram = tof_dataset[5, :]

    # Convert from seconds to nanoseconds
    bin_edges_ns = bin_edges * 1e9

    # Calculate bin centers for the line plot
    bin_centers = (bin_edges_ns[:-1] + bin_edges_ns[1:]) / 2

    # Calculate bin widths
    bin_widths = np.diff(bin_edges_ns)

    # Normalize to density if requested
    if plot_type == "density":
        # Density: histogram / (sum * bin_width)
        y_values = tof_histogram / (np.sum(tof_histogram) * bin_widths)
        ylabel = "Probability Density (ns$^{-1}$)"
    else:
        # Distribution: raw counts
        y_values = tof_histogram
        ylabel = "Count"

    # Load plot configuration
    plot_config = load_plot_config("./plotting_codes/plot_config.yaml")

    # Apply rcParams
    plt.rcParams.update(plot_config)

    # Get color from the color cycle
    if "axes.prop_cycle" in plot_config:
        # Extract colors from the cycler
        prop_cycle_str = plot_config["axes.prop_cycle"]
        # Parse the cycler string to extract colors
        colors_match = re.search(r"\[(.*?)\]", prop_cycle_str)
        if colors_match:
            colors_str = colors_match.group(1)
            colors = [c.strip().strip("'\"") for c in colors_str.split(",")]
            # Add '#' prefix if not present
            colors = ["#" + c if not c.startswith("#") else c for c in colors]
            bar_color = colors[0]  # Use first color
            line_color = colors[1]  # Use second color for line
        else:
            bar_color = "#7fc97f"
            line_color = "#beaed4"
    else:
        bar_color = "#7fc97f"
        line_color = "#beaed4"

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 4))

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

    # Plot line connecting bar centers
    ax.plot(
        bin_centers,
        y_values,
        color=line_color,
        linewidth=2,
        marker="o",
        markersize=4,
        label="Center line",
    )

    # Labels and title
    ax.set_xlabel("Time of Flight (ns)")
    ax.set_ylabel(ylabel)
    ax.set_title("Sample Time-of-Flight Distribution")
    # ax.legend()
    ax.grid(True, alpha=0.3)

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


def main(
    data_path="./data/generated_tof_set.npz", plot_type="distribution"
):
    """Main function to run the plotting script.

    Parameters:
    -----------
    data_path : str
        Path to the data file
    output_dir : str
        Output directory for figures
    plot_type : str
        Plot type: 'distribution' (raw counts) or 'density' (normalized)
    """
    plot_sample_tof(data_path, plot_type)


if __name__ == "__main__":
    main()
