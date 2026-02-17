"""
Plot Optimized Sensitivity vs. Frequency Separation for different optimizers.
Compares DIGSS, Liu et al., Modified Liu et al., and CW methods at a fixed depth.
"""

import yaml
from cycler import cycler
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def main(target_depth: int = 6):
    """Generate overlap comparison plot."""
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
    results_path = Path(__file__).parent.parent / 'results' / 'overlap_comparison_results.yaml'
    with open(results_path, 'r') as f:
        results = yaml.safe_load(f)

    digss_data = {
        'separations': [], 'sensitivities': [], 'depths': [],
        'baseline_noise_std': [], 'fetal_ac_energy': [], 'maternal_ac_amp': []
    }
    liu_data = {
        'separations': [], 'sensitivities': [], 'depths': [],
        'baseline_noise_std': [], 'fetal_ac_energy': [], 'maternal_ac_amp': []
    }
    alt_liu_data = {
        'separations': [], 'sensitivities': [], 'depths': [],
        'baseline_noise_std': [], 'fetal_ac_energy': [], 'maternal_ac_amp': []
    }
    cw_data = {
        'separations': [], 'sensitivities': [], 'depths': [],
        'baseline_noise_std': [], 'fetal_ac_energy': [], 'maternal_ac_amp': []
    }

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
        
        # Only process experiments at the target depth
        if depth != target_depth or sensitivity is None or separation is None:
            continue
        
        # Categorize by optimizer
        data_dict_to_use = None
        if str(optimizer).startswith('DIGSSOptimizer'):
            data_dict_to_use = digss_data
        elif str(optimizer).startswith('LiuOptimizer'):
            data_dict_to_use = liu_data
        elif str(optimizer).startswith('AltLiuOptimizer'):
            data_dict_to_use = alt_liu_data
        elif str(optimizer).startswith('DummyUnitWindowGenerator'):
            data_dict_to_use = cw_data
        
        if data_dict_to_use is not None:
            data_dict_to_use['separations'].append(separation)
            data_dict_to_use['sensitivities'].append(sensitivity)
            data_dict_to_use['depths'].append(depth)
            data_dict_to_use['baseline_noise_std'].append(baseline_noise_std)
            data_dict_to_use['fetal_ac_energy'].append(fetal_ac_energy)
            data_dict_to_use['maternal_ac_amp'].append(maternal_ac_amp)

    # Sort data by separation
    for data in [digss_data, liu_data, alt_liu_data, cw_data]:
        if data['separations']:
            sorted_indices = np.argsort(data['separations'])
            data['separations'] = np.array(data['separations'])[sorted_indices].tolist()
            data['sensitivities'] = np.array(data['sensitivities'])[sorted_indices].tolist()
            data['depths'] = np.array(data['depths'])[sorted_indices].tolist()
            data['baseline_noise_std'] = np.array(data['baseline_noise_std'])[sorted_indices].tolist()
            data['fetal_ac_energy'] = np.array(data['fetal_ac_energy'])[sorted_indices].tolist()
            data['maternal_ac_amp'] = np.array(data['maternal_ac_amp'])[sorted_indices].tolist()
            data['selectivity'] = (np.array(data['fetal_ac_energy']) / np.array(data['maternal_ac_amp'])**2).tolist()

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 4))

    # Plot each optimizer
    y_key = 'sensitivities'  # Change this key to plot different metrics
    if digss_data['separations']:
        ax.plot(digss_data['separations'], digss_data[y_key], 
                linewidth=2, markersize=8, label='DIGSS')
    # if liu_data['separations']:
    #     ax.plot(liu_data['separations'], liu_data[y_key], 
    #             linewidth=2, markersize=8, label='Liu et al.')
    # if alt_liu_data['separations']:
    #     ax.plot(alt_liu_data['separations'], alt_liu_data[y_key], 
    #             linewidth=2, markersize=8, label='Modified Liu et al.')
    # if cw_data['separations']:
    #     ax.plot(cw_data['separations'], cw_data[y_key], 
    #             linewidth=2, markersize=8, label='CW')

    # Configure axes
    ax.set_xlabel('Frequency Separation (Hz)')
    ax.set_ylabel('Figure of Merit (' + y_key + ')')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    # Save figure
    figures_dir = Path(__file__).parent.parent / 'figures'
    figures_dir.mkdir(exist_ok=True)

    fig.savefig(figures_dir / 'overlap_comparison.pdf', format='pdf')
    fig.savefig(figures_dir / 'overlap_comparison.svg', format='svg')

    print(f"Overlap comparison plots saved to {figures_dir}")
    plt.show()


if __name__ == "__main__":
    main(target_depth=20)
