#!/usr/bin/env python3
"""Plot DIGSS figure of merit vs. fetal frequency error percentage."""

from pathlib import Path
from cycler import cycler
import matplotlib.pyplot as plt
import yaml


def main(depth=10):
    """Generate false fetal frequency comparison plot for a given depth."""
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

    # Load false fetal frequency results
    results_path = Path(__file__).parent.parent / 'results' / 'false_fetal_f_results.yaml'
    with open(results_path, 'r') as f:
        results = yaml.safe_load(f)

    # Collect sensitivities by optimizer type and error percentage at a given depth.
    error_data = {
        'comb': {},
        'psafe_same_width': {},
        'liu': {},
        'cw': {},
    }

    for exp_data in results.values():
        if not isinstance(exp_data, dict):
            continue

        if exp_data.get('Depth_mm') != depth:
            continue

        optimizer = exp_data.get('Optimizer', '')
        sensitivity = exp_data.get('Optimized_Sensitivity')
        percent_error = exp_data.get('Percent_Error')

        if sensitivity is None or percent_error is None:
            continue

        if 'type=comb' in optimizer and 'DIGSSOptimizer' in optimizer:
            optimizer_key = 'comb'
        elif 'type=psafe_same_width' in optimizer and 'DIGSSOptimizer' in optimizer:
            optimizer_key = 'psafe_same_width'
        elif 'LiuOptimizer' in optimizer:
            optimizer_key = 'liu'
        elif 'DummyUnitWindowGenerator' in optimizer:
            optimizer_key = 'cw'
        else:
            continue

        error_data[optimizer_key][percent_error] = sensitivity

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 4))

    for optimizer_key, label in [
        ('comb', 'DIGSS w Comb Separator'),
        ('psafe_same_width', 'DIGSS w PSAFE Separator'),
        ('liu', 'Liu Optimizer'),
        ('cw', 'CW'),
    ]:
        if not error_data[optimizer_key]:
            continue

        percent_errors = sorted(error_data[optimizer_key].keys())
        sensitivities = [error_data[optimizer_key][error] for error in percent_errors]
        ax.plot(
            [error * 100 for error in percent_errors],
            sensitivities,
            linewidth=2,
            markersize=6,
            label=label,
        )

    # Configure axes
    ax.set_xlabel('Error Percentage (%)')
    ax.set_ylabel('Figure of Merit')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    xticks = sorted({error * 100 for data in error_data.values() for error in data.keys()})
    if xticks:
        ax.set_xticks(xticks)
        ax.set_xticklabels([f'{tick:.0f}%' for tick in xticks])

    # Save figure
    figures_dir = Path(__file__).parent.parent / 'figures'
    figures_dir.mkdir(exist_ok=True)

    fig.savefig(figures_dir / 'false_fetal_f_comparison.pdf', format='pdf')
    fig.savefig(figures_dir / 'false_fetal_f_comparison.svg', format='svg')

    print(f"False fetal frequency comparison plots saved to {figures_dir}")
    plt.show()


if __name__ == "__main__":
    main(20)
