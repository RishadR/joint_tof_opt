"""Plot FoM vs. fetal frequency error for comb filter_hw=0.01."""

from pathlib import Path
import re
import matplotlib.pyplot as plt
import yaml
from joint_tof_opt.plotting import load_plot_config

def main(depth=10):
    """Generate false fetal frequency comparison plot for a given depth."""
    # Load matplotlib configuration
    load_plot_config()

    # Load false fetal frequency results
    results_path = Path(__file__).parent.parent / 'results' / 'false_fetal_f_results2.yaml'
    with open(results_path, 'r') as f:
        results = yaml.safe_load(f)

    # Collect FoM by error level (Hz) for comb filter with filter_hw=0.01.
    error_data = {}

    def _parse_optimizer(optimizer_str):
        if 'DIGSSOptimizer' not in optimizer_str:
            return None, None

        type_match = re.search(r'type=([a-zA-Z0-9_]+)', optimizer_str)
        if not type_match:
            return None, None

        optimizer_type = type_match.group(1)
        filter_match = re.search(r'filter_hw=([0-9]*\.?[0-9]+)', optimizer_str)
        filter_hw = float(filter_match.group(1)) if filter_match else None
        return optimizer_type, filter_hw

    for exp_data in results.values():
        if not isinstance(exp_data, dict):
            continue

        if exp_data.get('Depth_mm') != depth:
            continue

        optimizer = exp_data.get('Optimizer', '')
        # Compute error in Hz as difference between errored and true fetal frequency
        errored_fetal_f = exp_data.get('Errored_Fetal_F_Hz')
        true_fetal_f = exp_data.get('True_Fetal_F_Hz')
        fom = exp_data.get('Optimized_Sensitivity')

        if errored_fetal_f is None or true_fetal_f is None or fom is None:
            continue

        error_hz = errored_fetal_f - true_fetal_f

        optimizer_type, filter_hw = _parse_optimizer(optimizer)
        # Only keep comb type with filter_hw=0.01
        if optimizer_type != 'comb' or filter_hw is None or abs(filter_hw - 0.01) > 1e-12:
            continue

        if error_hz not in error_data:
            error_data[error_hz] = []
        error_data[error_hz].append(fom)

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 4))

    # Build one curve: x=error (Hz), y=average FoM at that error.
    error_values = sorted(error_data.keys())
    fom_values = [sum(error_data[err]) / len(error_data[err]) for err in error_values]
    fom_values = [fom / max(fom_values) for fom in fom_values]  # Normalize FoM to max value
    ax.plot(error_values, fom_values)

    # Configure axes
    ax.set_xlabel('Fetal Frequency Error (Hz)')
    # ax.set_ylabel('FoM = Sensitivity $\\times$ Selectivity')
    ax.set_ylabel('Normalized FoM')
    
    ax.grid(True)

    ax.text(
        0.02,
        0.97,
        'True FHR: 2.5Hz\nMHR $2^{\\text{nd}}$ Harmonic: 2.0Hz',
        transform=ax.transAxes,
        ha='left',
        va='top',
        fontsize=12,
        bbox={
            'facecolor': 'white',
            'edgecolor': 'none',
            'linewidth': 0,
            'boxstyle': 'square,pad=0.3',
        },
    )

    # Save figure
    figures_dir = Path(__file__).parent.parent / 'figures'
    figures_dir.mkdir(exist_ok=True)

    fig.savefig(figures_dir / 'false_fetal_f_comparison2.pdf', format='pdf')
    fig.savefig(figures_dir / 'false_fetal_f_comparison2.svg', format='svg')

    print(f"False fetal frequency comparison plots saved to {figures_dir}")
    # plt.show()


if __name__ == "__main__":
    main(6)
