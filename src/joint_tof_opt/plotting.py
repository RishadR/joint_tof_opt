"""
Core plotting utilities
"""

from pathlib import Path
from cycler import cycler
import yaml
import matplotlib.pyplot as plt

# Load matplotlib configuration (same implementation as plot_detector_comparison.py)
config_path = Path(__file__).parent.parent.parent / "plotting_codes" / "plot_config.yaml"


def load_plot_config():
    """Load matplotlib configuration from YAML file."""
    with open(config_path, "r") as f:
        plot_config = yaml.safe_load(f)
        custom_cycler = cycler(color=plot_config["plotting"]["colors"]) + cycler(
            marker=plot_config["plotting"]["markers"]
        )
        plt.rcParams["axes.prop_cycle"] = custom_cycler
        plot_config.pop("plotting", None)
        plt.rcParams.update(plot_config)
