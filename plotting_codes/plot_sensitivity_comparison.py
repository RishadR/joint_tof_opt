"""
Plot Optimized Sensitivity vs. Fetal Depth for different optimizers.
Compares DIGSS, Liu et al., and CW methods.
"""

import yaml
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from joint_tof_opt.plotting import load_plot_config

def main():
    """Generate sensitivity comparison plot."""
    # Load matplotlib configuration
    load_plot_config()

    # Load sensitivity comparison results
    results_path = Path(__file__).parent.parent / 'results' / 'sensitivity_comparison_results.yaml'
    with open(results_path, 'r') as f:
        results = yaml.safe_load(f)

    # Extract data for each optimizer
    digss_data = {'depths': [], 'sensitivities': []}
    liu_data_h1 = {'depths': [], 'sensitivities': []}
    liu_data_h2 = {'depths': [], 'sensitivities': []}
    alt_liu_data_h1 = {'depths': [], 'sensitivities': []}
    alt_liu_data_h2 = {'depths': [], 'sensitivities': []}
    cw_data = {'depths': [], 'sensitivities': []}

    for exp_key, exp_data in results.items():
        if not isinstance(exp_data, dict):
            continue
        
        depth = round(float(exp_data.get('Depth_mm', 0.0))/10, 1) 
        sensitivity = exp_data.get('Optimized_Sensitivity')
        optimizer = exp_data.get('Optimizer', '')
        
        if depth is None or sensitivity is None:
            continue
        
        if str(optimizer).startswith('DIGSSOptimizer'):
            digss_data['depths'].append(depth)
            digss_data['sensitivities'].append(sensitivity)
        elif str(optimizer).startswith('LiuOptimizer'):
            if 'harmonics=1' in str(optimizer):
                liu_data_h1['depths'].append(depth)
                liu_data_h1['sensitivities'].append(sensitivity)
            elif 'harmonics=2' in str(optimizer):
                liu_data_h2['depths'].append(depth) 
                liu_data_h2['sensitivities'].append(sensitivity)
        elif str(optimizer).startswith('AltLiuOptimizer'):
            if 'harmonics=1' in str(optimizer):
                alt_liu_data_h1['depths'].append(depth)
                alt_liu_data_h1['sensitivities'].append(sensitivity)
            elif 'harmonics=2' in str(optimizer):
                alt_liu_data_h2['depths'].append(depth) 
                alt_liu_data_h2['sensitivities'].append(sensitivity)
        elif str(optimizer).startswith('DummyUnitWindowGenerator'):
            cw_data['depths'].append(depth)
            cw_data['sensitivities'].append(sensitivity)

    # Sort data by depth
    for data in [digss_data, liu_data_h1, liu_data_h2, alt_liu_data_h1, alt_liu_data_h2, cw_data]:
        if data['depths']:
            sorted_indices = np.argsort(data['depths'])
            data['depths'] = np.array(data['depths'])[sorted_indices].tolist()
            data['sensitivities'] = np.array(data['sensitivities'])[sorted_indices].tolist()

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 4))

    # Plot each optimizer
    ax.plot(digss_data['depths'], digss_data['sensitivities'], label='DIGSS')
    ax.plot(liu_data_h1['depths'], liu_data_h1['sensitivities'], label='Boxcar$^{[27]}$')
    # ax.plot(alt_liu_data_h2['depths'], alt_liu_data_h2['sensitivities'], label='Modified Boxcar$^{[27]}$')
    ax.plot(cw_data['depths'], cw_data['sensitivities'], label='CW')

    # Configure axes
    ax.set_xlabel('Fetal Depth (cm)')
    ax.set_ylabel('Selectivity $\\times$ SNR')
    ax.set_yscale('log')
    ax.legend(loc='upper right')
    # ax.set_ylim(top=0.02)
    ax.grid(True)

    # Save figure
    figures_dir = Path(__file__).parent.parent / 'figures'
    figures_dir.mkdir(exist_ok=True)

    fig.savefig(figures_dir / 'sensitivity_comparison.pdf', format='pdf')
    fig.savefig(figures_dir / 'sensitivity_comparison.svg', format='svg')

    print(f"Sensitivity comparison plots saved to {figures_dir}")
    # plt.show()


if __name__ == "__main__":
    main()
