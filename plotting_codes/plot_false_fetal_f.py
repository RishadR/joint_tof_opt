"""Plot separated fetal energy vs. fetal frequency error for comb filter_hw=0.01."""

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
    results_path = Path(__file__).parent.parent / 'results' / 'false_fetal_f_results.yaml'
    with open(results_path, 'r') as f:
        results = yaml.safe_load(f)

    # Collect fetal energy by error level (Hz) for comb filter with filter_hw=0.01.
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

    def _get_energy(exp_data, energy_kind):
        """Extract fetal/maternal energy from supported result layouts."""
        evaluator_log = exp_data.get('evaluator_log', {})

        if energy_kind == 'fetal':
            candidates = [
                exp_data.get('Optimizer(Fetal Energy)'),
                exp_data.get('Optimizer_Fetal_Energy'),
                exp_data.get('fetal_ac_energy'),
                evaluator_log.get('fetal_ac_energy'),
            ]
        else:
            candidates = [
                exp_data.get('Optimizer(Maternal Energy)'),
                exp_data.get('Optimizer_Maternal_Energy'),
                exp_data.get('maternal_ac_energy'),
                evaluator_log.get('maternal_ac_energy'),
            ]

        for value in candidates:
            if value is not None:
                return value
        return None

    for exp_data in results.values():
        if not isinstance(exp_data, dict):
            continue

        if exp_data.get('Depth_mm') != depth:
            continue

        optimizer = exp_data.get('Optimizer', '')
        fetal_energy = _get_energy(exp_data, 'fetal')

        # Compute error in Hz as difference between errored and true fetal frequency
        errored_fetal_f = exp_data.get('Errored_Fetal_F_Hz')
        true_fetal_f = exp_data.get('True_Fetal_F_Hz')

        if fetal_energy is None or errored_fetal_f is None or true_fetal_f is None:
            continue

        error_hz = errored_fetal_f - true_fetal_f

        optimizer_type, filter_hw = _parse_optimizer(optimizer)
        # Only keep comb type with filter_hw=0.01
        if optimizer_type != 'comb' or filter_hw is None or abs(filter_hw - 0.01) > 1e-12:
            continue

        if error_hz not in error_data:
            error_data[error_hz] = []
        error_data[error_hz].append(fetal_energy)

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5))

    # Build one curve: x=error (Hz), y=average fetal energy at that error.
    error_values = sorted(error_data.keys())
    fetal_energy_values = [sum(error_data[err]) / len(error_data[err]) for err in error_values]
    ax.plot(error_values, fetal_energy_values)

    # Configure axes
    ax.set_xlabel('Fetal Frequency Error (Hz)')
    ax.set_ylabel('Separated Fetal Energy')
    ax.grid(True, alpha=0.3)

    # Save figure
    figures_dir = Path(__file__).parent.parent / 'figures'
    figures_dir.mkdir(exist_ok=True)

    fig.savefig(figures_dir / 'false_fetal_f_comparison.pdf', format='pdf')
    fig.savefig(figures_dir / 'false_fetal_f_comparison.svg', format='svg')

    print(f"False fetal frequency comparison plots saved to {figures_dir}")
    # plt.show()


if __name__ == "__main__":
    main(6)
