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


def main():
    """Generate all plots for the paper."""
    print("=" * 60)
    print("Generating all plots for the paper")
    print("=" * 60)
    
    print("\n[1/3] Generating sensitivity comparison plot...")
    plot_sensitivity()
    
    print("\n[2/3] Generating detector comparison plot...")
    plot_detector()
    
    print("\n[3/3] Generating false fetal frequency comparison plot...")
    plot_false_f()
    
    print("\n" + "=" * 60)
    print("All plots generated successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
