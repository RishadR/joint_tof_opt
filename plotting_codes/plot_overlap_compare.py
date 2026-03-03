"""
Plot Optimized Reward-to-FoM ratio vs separation for DIGSS results.
"""
import yaml
from cycler import cycler
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np
from pathlib import Path


def main():
    """Generate DIGSS Optimized Reward-to-FoM ratio vs separation plot."""
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
        'ratio': [],
        'separations': []
    }

    for _, exp_data in results.items():
        if not isinstance(exp_data, dict):
            continue
        
        separation = round(float(exp_data.get('Separation_Hz', 0.0)), 1)
        fom = exp_data.get('Optimized_Sensitivity1')
        reward = exp_data.get('Optimized_Sensitivity2')
        optimizer = exp_data.get('Optimizer', '')

        if fom is None or reward is None or separation is None:
            continue

        if fom == 0:
            continue

        if str(optimizer).startswith('DIGSSOptimizer'):
            digss_data['ratio'].append(reward / fom)
            digss_data['separations'].append(separation)

    if digss_data['ratio']:
        sorted_indices = np.argsort(digss_data['separations'])
        digss_data['ratio'] = np.array(digss_data['ratio'])[sorted_indices].tolist()
        digss_data['separations'] = np.array(digss_data['separations'])[sorted_indices].tolist()

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 4))

    # Plot DIGSS points
    if digss_data['ratio']:
        ax.scatter(
            digss_data['separations'],
            digss_data['ratio'],
            s=50,
        )

    ax.axhline(1.0, linestyle='--', linewidth=1.2)

    # Configure axes
    ax.set_xlabel('Separation (Hz)')
    ax.set_ylabel('Optimized Reward / FoM')
    # ax.set_yscale('log')
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, which='major', alpha=0.3)
    ax.grid(True, which='minor', alpha=0.15)

    # Save figure
    figures_dir = Path(__file__).parent.parent / 'figures'
    figures_dir.mkdir(exist_ok=True)

    fig.savefig(figures_dir / 'overlap_comparison.pdf', format='pdf')
    fig.savefig(figures_dir / 'overlap_comparison.svg', format='svg')

    print(f"Overlap comparison plots saved to {figures_dir}")


if __name__ == "__main__":
    main()
