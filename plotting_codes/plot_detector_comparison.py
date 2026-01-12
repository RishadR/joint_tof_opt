#!/usr/bin/env python3
"""
Plot Optimized Sensitivity vs. Fetal Depth for different SDD indices.
Only for DIGSS optimizer.
"""

import yaml
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def main():
    """Generate detector comparison plot."""
    # Load matplotlib configuration
    config_path = Path(__file__).parent / 'plot_config.yaml'
    with open(config_path, 'r') as f:
        plot_config = yaml.safe_load(f)
        plt.rcParams.update(plot_config)

    # Load detector comparison results
    results_path = Path(__file__).parent.parent / 'results' / 'detector_comparison_results.yaml'
    with open(results_path, 'r') as f:
        results = yaml.safe_load(f)

    # SDD distances in mm
    sdd_distances = [2, 13, 24, 35, 46, 56, 67, 78, 89]

    # Extract data for each SDD index (only DIGSS optimizer)
    sdd_data = {}

    for exp_key, exp_data in results.items():
        if not isinstance(exp_data, dict):
            continue
        
        depth = exp_data.get('Depth_mm')
        sensitivity = exp_data.get('Optimized_Sensitivity')
        optimizer = exp_data.get('Optimizer', '')
        sdd_index = exp_data.get('SDD_Index')
        
        if depth is None or sensitivity is None or sdd_index is None:
            continue
        
        # Only process DIGSS optimizer
        if 'DIGSSOptimizer' not in str(optimizer):
            continue
        
        if sdd_index not in sdd_data:
            sdd_data[sdd_index] = {'depths': [], 'sensitivities': []}
        
        sdd_data[sdd_index]['depths'].append(depth)
        sdd_data[sdd_index]['sensitivities'].append(sensitivity)

    # Sort data by depth and create arrays
    for sdd_index in sdd_data:
        if sdd_data[sdd_index]['depths']:
            sorted_indices = np.argsort(sdd_data[sdd_index]['depths'])
            sdd_data[sdd_index]['depths'] = np.array(sdd_data[sdd_index]['depths'])[sorted_indices]
            sdd_data[sdd_index]['sensitivities'] = np.array(sdd_data[sdd_index]['sensitivities'])[sorted_indices]

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 4))

    # Plot each SDD index
    for sdd_index in sorted(sdd_data.keys()):
        sdd_distance = sdd_distances[sdd_index - 1]  # SDD_Index is 1-based
        ax.plot(sdd_data[sdd_index]['depths'], sdd_data[sdd_index]['sensitivities'], 
                marker='o', linewidth=2, markersize=8, 
                label=f'SDD = {sdd_distance} mm')

    # Configure axes
    ax.set_xlabel('Fetal Depth (mm)')
    ax.set_ylabel('SNR x Selectivity')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(top=1.3)

    # Save figure
    figures_dir = Path(__file__).parent.parent / 'figures'
    figures_dir.mkdir(exist_ok=True)

    fig.savefig(figures_dir / 'detector_comparison.pdf', format='pdf')
    fig.savefig(figures_dir / 'detector_comparison.svg', format='svg')

    print(f"Detector comparison plots saved to {figures_dir}")
    plt.show()


if __name__ == "__main__":
    main()
