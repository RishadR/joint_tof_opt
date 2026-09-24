"""
Core plotting utilities
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from cycler import cycler

from joint_tof_opt.evaluators.specs import load_evaluator_specs
from joint_tof_opt.misc import noisy_results_path

config_path = Path(__file__).parent / "plot_config.yaml"


def load_plot_config():
    """Load matplotlib configuration from YAML file."""
    with open(config_path, "r") as f:
        plot_config = yaml.safe_load(f)
        custom_cycler = (
            cycler(color=plot_config["plotting"]["colors"])
            + cycler(marker=plot_config["plotting"]["markers"])
            + cycler(linestyle=plot_config["plotting"]["line_styles"])
        )
        plt.rcParams["axes.prop_cycle"] = custom_cycler
        plot_config.pop("plotting", None)
        plt.rcParams.update(plot_config)


def resolve_results_path(base_path: Path) -> tuple[Path, bool]:
    """
    Resolve a results YAML path against the current evaluator_specs.yaml's inject_noise setting, so
    plots automatically read whichever of the noisy/noiseless result files the experiment scripts
    actually produced. Returns (resolved_path, inject_noise).
    """
    eval_spec = load_evaluator_specs(Path("./experiments/evaluator_specs.yaml"))
    return noisy_results_path(base_path, eval_spec.inject_noise), eval_spec.inject_noise


def legend_no_overlap(ax, loc: str, margin: float = 0.08, max_expand: int = 12, max_bisect: int = 12, **kwargs):
    """
    Place a legend in a fixed corner ("upper right" / "lower right" / etc.) and grow the
    y-axis limits just enough that the legend's bounding box doesn't cover any plotted line
    or errorbar. Works in data or log scale; caller still picks the corner.

    A wide legend (long labels) can overhang points far from its anchor corner, so instead
    of a fixed small step we exponentially expand until clear, then bisect back down to the
    minimal bound (plus `margin`) so the axis isn't padded more than necessary.
    """
    legend = ax.legend(loc=loc, **kwargs)
    fig = ax.figure
    grow_upper = "upper" in loc
    is_log = ax.get_yscale() == "log"

    def overlaps() -> bool:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        lbox = legend.get_window_extent(renderer)

        for other in fig.axes:  # includes twin axes sharing this figure
            # lines, errorbars (collections) and fills (patches, e.g. ax.fill error ellipses)
            for artist in (*other.lines, *other.collections, *other.patches):
                bbox = artist.get_window_extent(renderer)
                if bbox.width > 0 and bbox.height > 0 and lbox.overlaps(bbox):
                    return True
        return False

    if not overlaps():
        return legend

    lo0, hi0 = ax.get_ylim()
    span = hi0 - lo0
    stuck_bound = hi0 if grow_upper else lo0

    def set_bound(b):
        ax.set_ylim(lo0, b) if grow_upper else ax.set_ylim(b, hi0)

    # exponential search for a bound that's clearly overlap-free
    clear_bound = stuck_bound
    for i in range(max_expand):
        step = 2 ** (i + 1)
        clear_bound = (hi0 * step if grow_upper else lo0 / step) if is_log else \
            (hi0 + span * step if grow_upper else lo0 - span * step)
        set_bound(clear_bound)
        if not overlaps():
            break

    # bisect back down to the minimal bound that still clears
    for _ in range(max_bisect):
        mid = (10 ** ((np.log10(stuck_bound) + np.log10(clear_bound)) / 2)) if is_log \
            else (stuck_bound + clear_bound) / 2
        set_bound(mid)
        if overlaps():
            stuck_bound = mid
        else:
            clear_bound = mid

    final = clear_bound * (1 + margin) if is_log and grow_upper else \
        clear_bound / (1 + margin) if is_log else \
        clear_bound + span * margin if grow_upper else \
        clear_bound - span * margin
    set_bound(final)

    return legend


def as_samples(value: float | list[float]) -> list[float]:
    """Normalize an Optimized_Sensitivity value (scalar for a single run, list for repeats) to a list."""
    return value if isinstance(value, list) else [value]


def log_samples(value: dict | list[dict]) -> list[dict]:
    """Normalize an evaluator_log value (a dict for a single run, list of dicts for repeats) to a list."""
    return value if isinstance(value, list) else [value]
