#!/usr/bin/env python3
"""
Plot Optimized Sensitivity vs. Fetal Depth for different optimizers.
Compares DIGSS, Liu et al., and CW methods.
"""

import yaml
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load matplotlib configuration
config_path = Path(__file__).parent / 'plot_config.yaml'
with open(config_path, 'r') as f:
    plot_config = yaml.safe_load(f)
    plt.rcParams.update(plot_config)

# Load sensitivity comparison results
results_path = Path(__file__).parent.parent / 'results' / 'sensitivity_comparison_results.yaml'
with open(results_path, 'r') as f:
    results = yaml.safe_load(f)

# Extract data for each optimizer
digss_data = {'depths': [], 'sensitivities': []}
liu_data_h1 = {'depths': [], 'sensitivities': []}
liu_data_h2 = {'depths': [], 'sensitivities': []}
cw_data = {'depths': [], 'sensitivities': []}

for exp_key, exp_data in results.items():
    if not isinstance(exp_data, dict):
        continue
    
    depth = exp_data.get('Depth_mm')
    sensitivity = exp_data.get('Optimized_Sensitivity')
    optimizer = exp_data.get('Optimizer', '')
    
    if depth is None or sensitivity is None:
        continue
    
    if 'DIGSSOptimizer' in str(optimizer):
        digss_data['depths'].append(depth)
        digss_data['sensitivities'].append(sensitivity)
    elif 'LiuOptimizer' in str(optimizer):
        if 'harmonics=1' in str(optimizer):
            liu_data_h1['depths'].append(depth)
            liu_data_h1['sensitivities'].append(sensitivity)
        elif 'harmonics=2' in str(optimizer):
            liu_data_h2['depths'].append(depth) 
            liu_data_h2['sensitivities'].append(sensitivity)
    elif 'DummyUnitWindowGenerator' in str(optimizer):
        cw_data['depths'].append(depth)
        cw_data['sensitivities'].append(sensitivity)

# Sort data by depth
for data in [digss_data, liu_data_h1, liu_data_h2, cw_data]:
    if data['depths']:
        sorted_indices = np.argsort(data['depths'])
        data['depths'] = np.array(data['depths'])[sorted_indices].tolist()
        data['sensitivities'] = np.array(data['sensitivities'])[sorted_indices].tolist()

# Create figure
fig, ax = plt.subplots(figsize=(8, 6))

# Plot each optimizer
ax.plot(digss_data['depths'], digss_data['sensitivities'], 
        marker='o', linewidth=2, markersize=8, label='DIGSS')
ax.plot(liu_data_h1['depths'], liu_data_h1['sensitivities'], 
        marker='s', linewidth=2, markersize=8, label='Liu et al. (Single Harmonic)')
ax.plot(liu_data_h2['depths'], liu_data_h2['sensitivities'], 
        marker='s', linewidth=2, markersize=8, label='Liu et al. (Both Harmonics)')
ax.plot(cw_data['depths'], cw_data['sensitivities'], 
        marker='^', linewidth=2, markersize=8, label='CW')

# Configure axes
ax.set_xlabel('Fetal Depth (mm)')
ax.set_ylabel('Optimized Sensitivity')
ax.legend()
ax.grid(True, alpha=0.3)

# Save figure
figures_dir = Path(__file__).parent.parent / 'figures'
figures_dir.mkdir(exist_ok=True)

fig.savefig(figures_dir / 'sensitivity_comparison.pdf', format='pdf')
fig.savefig(figures_dir / 'sensitivity_comparison.svg', format='svg')

print(f"Plots saved to {figures_dir}")
plt.show()
