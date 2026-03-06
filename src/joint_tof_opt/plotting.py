"""
Core plotting utilities
"""

from pathlib import Path

import matplotlib.pyplot as plt
import yaml
from cycler import cycler

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
