#!/usr/bin/env python3
"""
Plot Optimized Sensitivity vs. Frequency Separation for DIGSSOptimizer.
Compares different filter_hw values at a fixed fetal depth.
"""

import yaml
import matplotlib.pyplot as plt
import numpy as np
import re
from pathlib import Path
from cycler import cycler


def main(target_depth: int = 6):
    """Generate filter halfwidth comparison plot."""
    # Load matplotlib configuration
    config_path = Path(__file__).parent / 'plot_config.yaml'
    with open(config_path, 'r') as f:
        plot_config = yaml.safe_load(f)
        custom_cycler = (cycler(color=plot_config['plotting']['colors']) +
                         # Turning off line styles - makes it too messy
                #  cycler(linestyle=plot_config['plotting']['line_styles']) +
                 cycler(marker=plot_config['plotting']['markers']))
        plt.rcParams['axes.prop_cycle'] = custom_cycler
        plot_config.pop('plotting', None)  # Remove custom plotting config from rcParams
        plt.rcParams.update(plot_config)

    # Load overlap comparison results
    results_path = Path(__file__).parent.parent / 'results' / 'overlap_comparison_results2.yaml'
    with open(results_path, 'r') as f:
        results = yaml.safe_load(f)

    # Store data grouped by filter_hw value
    filter_hw_data = {}

    for exp_key, exp_data in results.items():
        if not isinstance(exp_data, dict):
            continue
        
        depth = exp_data.get('Depth_mm')
        sensitivity = exp_data.get('Optimized_Sensitivity')
        evaluator_log = exp_data.get('evaluator_log', {})
        baseline_noise_std = evaluator_log.get('baseline_noise_std', 1.0)
        fetal_ac_energy = evaluator_log.get('fetal_ac_energy', 1.0)
        maternal_ac_amp = evaluator_log.get('maternal_ac_amplitude', 1.0)
        separation = exp_data.get('Separation_Hz')
        optimizer = exp_data.get('Optimizer', '')
        
        # Only process DIGSSOptimizer experiments at the target depth
        if not str(optimizer).startswith('DIGSSOptimizer'):
            continue
        if depth != target_depth or sensitivity is None:
            continue
        
        # Extract filter_hw from optimizer string using regex
        match = re.search(r'filter_hw=([\d.]+)', str(optimizer))
        if not match:
            continue
        
        filter_hw = float(match.group(1))
        
        # Initialize data dict for this filter_hw if not exists
        if filter_hw not in filter_hw_data:
            filter_hw_data[filter_hw] = {
                'filter_hw_values': [],
                'sensitivities': [],
                'depths': [],
                'baseline_noise_std': [],
                'fetal_ac_energy': [],
                'maternal_ac_amp': [],
                'separations': []
            }
        
        # Append data
        filter_hw_data[filter_hw]['filter_hw_values'].append(filter_hw)
        filter_hw_data[filter_hw]['sensitivities'].append(sensitivity)
        filter_hw_data[filter_hw]['depths'].append(depth)
        filter_hw_data[filter_hw]['baseline_noise_std'].append(baseline_noise_std)
        filter_hw_data[filter_hw]['fetal_ac_energy'].append(fetal_ac_energy)
        filter_hw_data[filter_hw]['maternal_ac_amp'].append(maternal_ac_amp)
        filter_hw_data[filter_hw]['separations'].append(separation)

    # Sort each filter_hw's data by separation
    for filter_hw in filter_hw_data:
        data = filter_hw_data[filter_hw]
        sorted_indices = np.argsort(data['separations'])
        data['separations'] = np.array(data['separations'])[sorted_indices].tolist()
        data['sensitivities'] = np.array(data['sensitivities'])[sorted_indices].tolist()
        data['depths'] = np.array(data['depths'])[sorted_indices].tolist()
        data['baseline_noise_std'] = np.array(data['baseline_noise_std'])[sorted_indices].tolist()
        data['fetal_ac_energy'] = np.array(data['fetal_ac_energy'])[sorted_indices].tolist()
        data['maternal_ac_amp'] = np.array(data['maternal_ac_amp'])[sorted_indices].tolist()

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 4))

    # Plot each filter_hw as a separate line
    for filter_hw in sorted(filter_hw_data.keys()):
        data = filter_hw_data[filter_hw]
        y_to_plot = np.array(data['fetal_ac_energy']) ** (1/2) / np.array(data['maternal_ac_amp'])
        if data['separations']:
            ax.plot(data['separations'], y_to_plot,
                    linewidth=2, markersize=8,
                    label=f'filter_hw={filter_hw} Hz')

    # Configure axes
    ax.set_xlabel('Frequency Separation (Hz)')
    ax.set_ylabel('Figure of Merit')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Save figure
    figures_dir = Path(__file__).parent.parent / 'figures'
    figures_dir.mkdir(exist_ok=True)

    fig.savefig(figures_dir / 'overlap_comparison2.pdf', format='pdf')
    fig.savefig(figures_dir / 'overlap_comparison2.svg', format='svg')

    print(f"Filter halfwidth comparison plots saved to {figures_dir}")
    plt.show()


if __name__ == "__main__":
    main(target_depth=20)
