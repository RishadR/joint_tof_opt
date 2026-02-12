"""
Visualize photon paths through a slab with varying numbers of sections and depths - figure for time gating paper.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import yaml


def _random_increments(total, count, rng):
    weights = rng.uniform(0.2, 1.0, size=count)
    return total * weights / weights.sum()


def generate_photon_path(num_sections, depth, rng, start, dx_half):
    if num_sections % 2 != 0:
        raise ValueError("num_sections must be even.")

    half = num_sections // 2
    first_dx = _random_increments(dx_half, half, rng)
    first_dy = _random_increments(depth, half, rng)
    second_dx = _random_increments(dx_half, half, rng)
    second_dy = -_random_increments(depth, half, rng)

    dx = np.concatenate([first_dx, second_dx])
    dy = np.concatenate([first_dy, second_dy])
    steps = np.column_stack([dx, dy])
    points = np.vstack([start, start + np.cumsum(steps, axis=0)])

    lengths = np.linalg.norm(steps, axis=1)
    return points, lengths.sum()


def main():
    # Load configs
    config_path = "./plotting_codes/plot_config.yaml"
    with open(config_path, "r") as f:
        config : dict= yaml.safe_load(f)
    config.pop("plotting", None)
    plt.rcParams.update(config)
    
    rng = np.random.default_rng(42)

    slab_width = 10.0
    slab_height = 4.0
    left_point = np.array([1.0, 0.0])
    right_point = np.array([slab_width - 1.0, 0.0])
    target_dx = right_point[0] - left_point[0]

    section_counts = [4, 6, 8, 10, 12]
    depths = [0.3, 1.2, 2.1, 3.0, 4.0]
    paths = []
    lengths = []

    for num_sections, depth in zip(section_counts, depths):
        points, total_len = generate_photon_path(
            num_sections,
            depth,
            rng,
            left_point,
            target_dx / 2.0,
        )
        paths.append(points)
        lengths.append(total_len)

    min_len = min(lengths)
    max_len = max(lengths)
    if max_len - min_len < 1e-6:
        colors = [(0.2, 0.2, 0.8)] * len(lengths)
    else:
        colors = []
        for total_len in lengths:
            t = (total_len - min_len) / (max_len - min_len)
            colors.append((t, 0.1, 1.0 - t))

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(8, 4), facecolor="white")
    ax_left.set_facecolor("white")
    ax_right.set_facecolor("white")

    slab = Rectangle(
        (0, 0),
        slab_width,
        slab_height,
        facecolor="#eef3ff",
        edgecolor="#b3c0dd",
        linewidth=1.5,
    )
    ax_left.add_patch(slab)

    for points, color, num_sections in zip(paths, colors, section_counts):
        ax_left.plot(
            points[:, 0],
            points[:, 1],
            color=color,
            linewidth=2.2,
        )
        for start_pt, end_pt in zip(points[:-1], points[1:]):
            ax_left.annotate(
                "",
                xy=(end_pt[0], end_pt[1]),
                xytext=(start_pt[0], start_pt[1]),
                arrowprops={
                    "arrowstyle": "->",
                    "color": color,
                    "lw": 1.4,
                    "shrinkA": 0,
                    "shrinkB": 0,
                },
            )

    ax_left.scatter(
        [left_point[0], right_point[0]],
        [left_point[1], right_point[1]],
        color="black",
        s=45,
        zorder=5,
    )
    ax_left.text(left_point[0] - 0.2, left_point[1] - 0.2, "Source", fontsize=12)
    ax_left.text(right_point[0] + 0.1, right_point[1] - 0.2, "Detector", fontsize=12)

    ax_left.set_xlim(-0.5, slab_width + 0.5)
    ax_left.set_ylim(slab_height + 0.6, -0.6)
    ax_left.set_aspect("equal", adjustable="box")
    ax_left.axis("off")

    times = np.array(lengths) * 0.2
    weights = np.exp(-np.array(lengths))
    order = np.argsort(times)
    times = times[order]
    weights = weights[order]
    colors_sorted = [colors[i] for i in order]

    if len(times) > 1:
        span = times.max() - times.min()
        width = 0.6 * span / len(times)
    else:
        width = 0.1

    ax_right.bar(times, weights, width=width, color=colors_sorted, edgecolor="none")
    ax_right.set_xlabel("Arrival time")
    ax_right.set_ylabel("Log(Packet Intensity)")
    ax_right.set_title("Time-of-flight distribution")
    ax_right.set_yscale("log")
    # Remove the tick labels but keep the ticks themselves for visual clarity
    ax_right.set_xticklabels([])
    ax_right.set_yticklabels([])

    plt.tight_layout()
    plt.savefig("./figures/time_gating_visual.pdf", bbox_inches="tight")
    plt.savefig("./figures/time_gating_visual.svg", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
