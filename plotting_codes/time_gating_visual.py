"""
Visualize photon paths through a slab with varying numbers of sections and depths - figure for time gating paper.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from joint_tof_opt.plotting import load_plot_config


def _random_increments(total, count, rng, allow_negative=False):
    if allow_negative:
        mask = rng.random(size=count) < 0.1
        weights = np.empty(count)
        weights[mask] = rng.uniform(-0.2, -0.1, size=mask.sum())
        weights[~mask] = rng.uniform(0.2, 1.0, size=(~mask).sum())
    else:
        weights = rng.uniform(0.2, 1.0, size=count)
    return total * weights / weights.sum()


def generate_photon_path(num_sections, depth, rng, start, dx_half):
    if num_sections % 2 != 0:
        raise ValueError("num_sections must be even.")

    half_len = num_sections // 2
    quarter_len = half_len // 2
    three_quarter_len = 3 * quarter_len

    # Randomly choose where we reach the max-depth from these choices
    reach_max_sections = np.random.choice([quarter_len, half_len, three_quarter_len])
    remaining_index = num_sections - reach_max_sections
    first_dx = _random_increments(dx_half, reach_max_sections, rng, allow_negative=True)
    first_dy = _random_increments(depth, reach_max_sections, rng)
    second_dx = _random_increments(dx_half, remaining_index, rng, allow_negative=True)
    second_dy = -_random_increments(depth, remaining_index, rng)

    dx = np.concatenate([first_dx, second_dx])
    dy = np.concatenate([first_dy, second_dy])
    steps = np.column_stack([dx, dy])
    points = np.vstack([start, start + np.cumsum(steps, axis=0)])

    lengths = np.linalg.norm(steps, axis=1)
    return points, lengths.sum()


def main():
    # plt.xkcd(scale=0.6, length=0.6, randomness=0.2)
    # Load configs
    load_plot_config()

    rng = np.random.default_rng(32)

    slab_width = 10.0
    slab_height = 5.5
    left_point = np.array([1.0, 0.0])
    right_point = np.array([slab_width - 1.0, 0.0])
    target_dx = right_point[0] - left_point[0]

    section_counts = [8, 12, 16, 20]
    depths = [1.5, 3.0, 4.0, 5.0]
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

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(8, 3), facecolor="white")
    ax_left.set_facecolor("white")
    ax_right.set_facecolor("white")

    slab = Rectangle(
        (0, 0),
        slab_width,
        slab_height,
        facecolor="#ffebeb",
        edgecolor="#ddb3b3",
        linewidth=1.5,
    )
    ax_left.add_patch(slab)

    for points, color, num_sections in zip(paths, colors, section_counts):
        ax_left.plot(
            points[:, 0],
            points[:, 1],
            color=color,
            linewidth=1.8,
            linestyle="-",
            marker="",
        )
        for start_pt, end_pt in zip(points[:-1], points[1:]):
            ax_left.annotate(
                "",
                xy=(end_pt[0], end_pt[1]),
                xytext=(start_pt[0], start_pt[1]),
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": color,
                    "lw": 0.5,
                    "shrinkA": 0,
                    "shrinkB": 0,
                },
            )

    ax_left.scatter(
        [left_point[0]],
        [left_point[1]],
        color="black",
        s=45,
        zorder=6,
    )
    detector_width = 0.6
    detector_height = 0.3
    detector = FancyBboxPatch(
        (right_point[0] - detector_width / 2, right_point[1] - detector_height / 2),
        detector_width,
        detector_height,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        facecolor="gray",
        edgecolor="black",
        linewidth=0.8,
        zorder=6,
    )
    ax_left.add_patch(detector)
    ax_left.text(left_point[0] - 0.2, left_point[1] - 0.3, "Source", fontsize=16)
    ax_left.text(right_point[0] - 1.2, right_point[1] - 0.3, "Detector", fontsize=16)

    ax_left.set_xlim(-0.5, slab_width + 0.5)
    # ax_left.set_ylim(slab_height + 0.6, -0.6)
    ax_left.set_ylim(slab_height, -0.6)
    # ax_left.set_aspect("equal", adjustable="datalim")
    ax_left.axis("off")
    ax_left.text(0.5, -0.15, "a)", transform=ax_left.transAxes, fontsize=16, ha="center", va="top")

    times = np.array(lengths) * 0.2
    max_penetration_depths = np.array([points[:, 1].max() for points in paths])
    order = np.argsort(times)
    times = times[order]
    max_penetration_depths = max_penetration_depths[order]
    colors_sorted = [colors[i] for i in order]

    if len(times) > 1:
        span = times.max() - times.min()
        width = 0.6 * span / len(times)
    else:
        width = 0.1

    ax_right.bar(
        times,
        max_penetration_depths,
        width=width,
        color=colors_sorted,
        edgecolor="black",
    )
    ax_right.set_xlabel("Arrival time")
    ax_right.set_ylabel("Max Penetration Depth")
    plt.grid(False)
    # Remove the tick labels but keep the ticks themselves for visual clarity
    ax_right.set_xticklabels([])
    ax_right.set_yticklabels([])
    ax_right.text(0.5, -0.15, "b)", transform=ax_right.transAxes, fontsize=16, ha="center", va="top")

    plt.tight_layout()

    plt.savefig("./figures/time_gating_visual.pdf", bbox_inches="tight")
    plt.savefig("./figures/time_gating_visual.svg", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
