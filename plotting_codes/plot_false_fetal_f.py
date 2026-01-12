#!/usr/bin/env python3
"""
Plot performance drop in DIGSS when wrong fetal heartrate frequency is used.
Shows Optimized Sensitivity vs. Fetal Depth for different error levels.
"""

import yaml
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def main():
    """Generate false fetal frequency comparison plot."""
    # Load matplotlib configuration
    config_path = Path(__file__).parent / 'plot_config.yaml'
    with open(config_path, 'r') as f:
        plot_config = yaml.safe_load(f)
        plt.rcParams.update(plot_config)

    # Load false fetal frequency results
    results_path = Path(__file__).parent.parent / 'results' / 'false_fetal_f_results.yaml'
    with open(results_path, 'r') as f:
        results = yaml.safe_load(f)

    # Extract data for each error level
    error_data = {}

    for exp_key, exp_data in results.items():
        if not isinstance(exp_data, dict):
            continue
        
        depth = exp_data.get('Depth_mm')
        sensitivity = exp_data.get('Optimized_Sensitivity')
        percent_error = exp_data.get('Percent_Error')
        
        if depth is None or sensitivity is None or percent_error is None:
            continue
        
        if percent_error not in error_data:
            error_data[percent_error] = {'depths': [], 'sensitivities': []}
        
        error_data[percent_error]['depths'].append(depth)
        error_data[percent_error]['sensitivities'].append(sensitivity)

    # Sort data by depth
    for error_level in error_data:
        if error_data[error_level]['depths']:
            sorted_indices = np.argsort(error_data[error_level]['depths'])
            error_data[error_level]['depths'] = np.array(error_data[error_level]['depths'])[sorted_indices]
            error_data[error_level]['sensitivities'] = np.array(error_data[error_level]['sensitivities'])[sorted_indices]

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 4))

    # Plot each error level
    for error_level in sorted(error_data.keys()):
        error_percent = error_level * 100
        # Let's only plot the 10s multiple for clarity
        if error_percent % 10 == 0: 
            ax.plot(error_data[error_level]['depths'], error_data[error_level]['sensitivities'], 
                marker='o', linewidth=2, markersize=8, 
                label=f'{error_percent:.0f}% Error')

    # Configure axes
    ax.set_xlabel('Fetal Depth (mm)')
    ax.set_ylabel('SNR x Selectivity')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Save figure
    figures_dir = Path(__file__).parent.parent / 'figures'
    figures_dir.mkdir(exist_ok=True)

    fig.savefig(figures_dir / 'false_fetal_f_comparison.pdf', format='pdf')
    fig.savefig(figures_dir / 'false_fetal_f_comparison.svg', format='svg')

    print(f"False fetal frequency comparison plots saved to {figures_dir}")
    plt.show()


if __name__ == "__main__":
    main()
