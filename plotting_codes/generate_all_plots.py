#!/usr/bin/env python3
"""
Master script to generate all paper figures.
Calls all individual plotting scripts.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for batch processing

from plot_sensitivity_comparison import main as plot_sensitivity
from plot_detector_comparison import main as plot_detector
from plot_false_fetal_f import main as plot_false_f
from plot_sample_tof import main as plot_sample_tof
from plot_overlap_compare import main as plot_overlap_compare
from plot_overlap_compare2 import main as plot_overlap_compare2
from plot_detector_comparison2 import main as plot_detector_comparison2


def main():
    """Generate all plots for the paper."""
    print("=" * 60)
    print("Generating all plots for the paper")
    print("=" * 60)

    print("\n[1/7] Generating sensitivity comparison plot...")
    plot_sensitivity()

    print("\n[2/7] Generating detector comparison plot...")
    plot_detector()

    print("\n[3/7] Generating false fetal frequency comparison plot...")
    plot_false_f()

    print("\n[4/7] Generating sample time-of-flight plot...")
    plot_sample_tof(plot_type="distribution")
    plot_sample_tof(plot_type="density")

    print("\n[5/7] Generating overlap comparison plot...")
    plot_overlap_compare()

    print("\n[6/7] Generating overlap comparison plot (variant)...")
    plot_overlap_compare2()
    
    print("\n[7/7] Generating detector comparison plot (variant)...")
    plot_detector_comparison2()

    print("\n" + "=" * 60)
    print("All plots generated successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
