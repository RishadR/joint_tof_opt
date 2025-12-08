"""
Visualization for sensitivity comparison results with optimized window display.

Layout:
- Left pane: Optimized window vs bin center
- Right top pane: Optimized & Vanilla Sensitivity vs Depth for 'abs' measurand (dual y-axes with Improvement %)
- Right bottom pane: Optimized & Vanilla Sensitivity vs Depth for 'V' measurand (dual y-axes with Improvement %)
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
import re

# Get Plot config
config_file = "./experiments/plot_config.yaml"
with open(config_file, "r") as f:
    plot_config = yaml.safe_load(f)
plt.rcParams.update(plot_config)

# Load optimized windows and timebin edges
window_to_plot = "abs_depth_10"  # Example: 'abs_depth_10' for measurand 'abs' and depth 10mm
windows_data = np.load("./results/optimized_windows.npz")
timebin_edges_data = np.load("./results/timebin_edges.npz")
optimized_window = windows_data[window_to_plot]
bin_edges = timebin_edges_data[window_to_plot]
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
bin_centers_ns = bin_centers * 1e9  # Convert to nanoseconds
bin_centers_ns = np.round(bin_centers_ns, decimals=2)  # Makes plotting cleaner

# Load sensitivity results
df_results = pd.read_csv("./results/sensitivity_comparison_results.csv")
measurands_to_plot = ["abs", "V"]
df_results_filtered = df_results[df_results["Measurand"].isin(measurands_to_plot)]

# Create figure with custom grid layout
fig = plt.figure(figsize=(12, 4))
gs = gridspec.GridSpec(2, 2, figure=fig, width_ratios=[1, 1], hspace=0.35, wspace=0.15)

# Left pane: single plot spanning both rows
ax_left = fig.add_subplot(gs[:, 0])

# Right panes: two vertical plots
ax_right_top = fig.add_subplot(gs[0, 1])
ax_right_bottom = fig.add_subplot(gs[1, 1])

# ============ LEFT PANE: Optimized Window vs Bin Center ============
ax_left.plot(bin_centers_ns, optimized_window, marker="o", linewidth=2, markersize=5)
ax_left.set_xlabel("Quantized Distribution of Time-of-Flight(DTOF)\nBin Centers (ns)", fontsize=12)
ax_left.set_ylabel("Window Value", fontsize=12)
ax_left.set_title(f"(a) Example of a DIGSS-Optimized Window", fontsize=14)
ax_left.grid(True, alpha=0.3)

# ============ RIGHT TOP PANE: 'abs' measurand ============
df_abs = df_results_filtered[df_results_filtered["Measurand"] == "abs"].sort_values("Depth")
ax_right_top.sharex(ax_right_bottom)
# Extract colors from plot_config
prop_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
if len(prop_cycle) >= 4:
    colors = prop_cycle
else:
    # Fallback colors
    colors = ["#000000", "#000000", "#386cb0", "#fdc086", "#f0027f"]  # fallback

color_optimized = colors[2]
color_vanilla = colors[3]
color_improvement = colors[4]

# Left y-axis: Sensitivities
ax_right_top.plot(
    df_abs["Depth"],
    df_abs["Optimized Sensitivity"],
    marker="s",
    label="DIGSS-Optimized Sensitivity",
    linewidth=2,
    markersize=6,
    color=color_optimized,
)
ax_right_top.plot(
    df_abs["Depth"],
    df_abs["Vanilla Sensitivity"],
    marker="^",
    label="Non-Timegated Sensitivity",
    linewidth=2,
    markersize=6,
    color=color_vanilla,
)
ax_right_top.set_ylabel("Sensitivity, $\\frac{\\Delta N_{tot}}{\\Delta \\mu_{a,fetal}}$", fontsize=12, color="black")
ax_right_top.tick_params(axis="y", labelcolor="black")
ax_right_top.tick_params(axis="x", labelbottom=False)  # Hide x-axis labels for top subplot
ax_right_top.set_title("(b) Measurand: Photon Count ($N_{tot}$)", fontsize=14)
ax_right_top.grid(True, alpha=0.3)


# Right y-axis: Improvement %
ax_right_top_twin = ax_right_top.twinx()
ax_right_top_twin.plot(
    df_abs["Depth"],
    df_abs["Improvement"],
    marker="D",
    label="Improvement %",
    linewidth=2,
    markersize=6,
    color=color_improvement,
    linestyle="--",
)
ax_right_top_twin.set_ylabel("Improvement (%)", fontsize=12, color=color_improvement)
ax_right_top_twin.set_ylim(bottom=0)
current_top = ax_right_top_twin.get_ylim()[1]
ax_right_top_twin.set_ylim(top=current_top * 1.2)
ax_right_top_twin.tick_params(axis="y", labelcolor=color_improvement)
# Combined legend
lines1, labels1 = ax_right_top.get_legend_handles_labels()
lines2, labels2 = ax_right_top_twin.get_legend_handles_labels()
ax_right_top.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=10)

# ============ RIGHT BOTTOM PANE: 'V' measurand ============
df_V = df_results_filtered[df_results_filtered["Measurand"] == "V"].sort_values("Depth")

# Left y-axis: Sensitivities
ax_right_bottom.plot(
    df_V["Depth"],
    df_V["Optimized Sensitivity"],
    marker="s",
    label="DIGSS-Optimized Sensitivity",
    linewidth=2,
    markersize=6,
    color=color_optimized,
)
ax_right_bottom.plot(
    df_V["Depth"],
    df_V["Vanilla Sensitivity"],
    marker="^",
    label="Non-Timegated Sensitivity",
    linewidth=2,
    markersize=6,
    color=color_vanilla,
)
ax_right_bottom.set_xlabel("Fetal Depth (mm)", fontsize=12)
ax_right_bottom.set_ylabel("Sensitivity, $\\frac{\\Delta V}{\\Delta \\mu_{a,fetal}}$", fontsize=12, color="black")
ax_right_bottom.tick_params(axis="y", labelcolor="black")
ax_right_bottom.set_title("(c) Measurand: Photon Arrival Time Variance($V$)", fontsize=14)
ax_right_bottom.grid(True, alpha=0.3)

# Right y-axis: Improvement %
ax_right_bottom_twin = ax_right_bottom.twinx()
ax_right_bottom_twin.plot(
    df_V["Depth"],
    df_V["Improvement"],
    marker="D",
    label="Improvement %",
    linewidth=2,
    markersize=6,
    color=color_improvement,
    linestyle="--",
)
ax_right_bottom_twin.set_ylabel("Improvement (%)", fontsize=12, color=color_improvement)
ax_right_bottom_twin.set_ylim(bottom=0)
current_top = ax_right_bottom_twin.get_ylim()[1]
ax_right_bottom_twin.set_ylim(top=current_top * 1.5)
ax_right_bottom_twin.tick_params(axis="y", labelcolor=color_improvement)

# No legend for bottom plot to avoid clutter

# Save figures
output_path = "./figures/sensitivity_comparison"
Path("./figures").mkdir(exist_ok=True)
plt.savefig(f"{output_path}.svg", format="svg")
plt.savefig(f"{output_path}.pdf", format="pdf")
print(f"Plots saved to {output_path}.svg and {output_path}.pdf")

plt.show()
