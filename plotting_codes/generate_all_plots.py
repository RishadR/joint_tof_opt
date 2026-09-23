"""
Master script to generate all paper figures.
Calls all individual plotting scripts.
"""

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for batch processing

from .plot_ablation_study import main as plot_ablation_study
from .plot_ablation_study2 import main as plot_ablation_study2
from .plot_detector_comparison import main as plot_detector
from .plot_detector_comparison2 import main as plot_detector_comparison2
from .plot_false_fetal_f import main as plot_false_f
from .plot_noise_sensitivity_comparison import main as plot_noise_sensitivity
from .plot_noise_sensitivity_comparison2 import main as plot_noise_sensitivity2
from .plot_overlap_compare import main as plot_overlap_compare
from .plot_overlap_compare2 import main as plot_overlap_compare2
from .plot_sample_tof import main as plot_sample_tof
from .plot_sensitivity_comparison import main as plot_sensitivity
from .plot_sensitivity_comparison2 import main as plot_sensitivity2
from .plot_sensitivity_comparison4 import main as plot_sensitivity4


def main():
    """Generate all plots for the paper."""
    print("=" * 60)
    print("Generating all plots for the paper")
    print("=" * 60)

    print("\n[1/9] Generating sensitivity comparison plot...")
    plot_sensitivity()

    print("\n[2/9] Generating detector comparison plot...")
    plot_detector()

    print("\n[3/9] Generating false fetal frequency comparison plot...")
    plot_false_f()

    print("\n[4/9] Generating sample time-of-flight plot...")
    plot_sample_tof(plot_type="distribution")
    plot_sample_tof(plot_type="density")

    print("\n[5/9] Generating overlap comparison plot...")
    plot_overlap_compare()

    print("\n[6/9] Generating overlap comparison plot (variant)...")
    plot_overlap_compare2()

    print("\n[7/9] Generating detector comparison plot (variant)...")
    plot_detector_comparison2()

    print("\n[8/9] Generating sensitivity comparison plot (variant)...")
    plot_sensitivity2()

    print("\n[9/9] Generating optimized-window sensitivity comparison plot...")
    plot_sensitivity4()

    print("\n[10/9] Generating noise sensitivity comparison plot...")
    plot_noise_sensitivity()
    print("\n[11/9] Generating noise sensitivity comparison plot (variant)...")
    plot_noise_sensitivity2()
    print("\n[12/9] Generating ablation study plot...")
    plot_ablation_study()
    print("\n[13/9] Generating ablation study plot (variant)...")
    plot_ablation_study2()

    print("\n" + "=" * 60)
    print("All plots generated successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
