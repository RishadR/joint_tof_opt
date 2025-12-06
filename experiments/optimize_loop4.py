"""
Fourth implementation of the optimization that optimizes noise * contrast

Process Flow:
1. Load ./data/generated_tof_set.npz containing a 2D array of ToF series
2. Compute First Order Moment with a parameterized window
3. Apply Sinc Comb Filter to extract fetal signal using a known frequency
4. Apply Sinc Comb Filter to extract maternal signal using a known frequency
5. Compute the Energy Ratio Metric between filtered fetal and filtered maternal signals
6. Compute the contrast-to-noise metric for the fetal signal
7. Final Metric is the product of Energy Ratio and Contrast-to-Noise
8. Optimize the window parameters to maximize the final metric
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from joint_tof_opt.compact_stat_process import NthOrderMoment, NthOrderCenteredMoment
from joint_tof_opt.signal_process import CombSeparator
from joint_tof_opt.metric_process import EnergyRatioMetric, ContrastToNoiseMetric
from joint_tof_opt import compute_noise_m1, compute_noise_variance, compute_noise_window_sum


# Load the generated ToF dataset
data = np.load("./data/generated_tof_set.npz")
tof_series = data["tof_dataset"]  # Shape: (num_timepoints, num_bins)
bin_edges = data["bin_edges"]  # Shape: (num_bins + 1,)
sampling_rate = data["sampling_rate"]   # Sampling rate in Hz
fetal_f = data["fetal_f"]  # Fetal heartbeat frequency in Hz
maternal_f = data["maternal_f"]  # Maternal heartbeat frequency in Hz
time_axis = data["time_axis"]  # Time axis

# Convert to torch tensors
tof_series_tensor = torch.tensor(tof_series, dtype=torch.float32)
bin_edges_tensor = torch.tensor(bin_edges, dtype=torch.float32)
num_timepoints, num_bins = tof_series_tensor.shape
# window_exponents = nn.Parameter(torch.ones(num_bins, dtype=torch.float32, requires_grad=True))
window_exponents = nn.Parameter(torch.ones(num_bins, dtype=torch.float32, requires_grad=True))


# Initialize modules
# moment_calculator = NthOrderMoment(tof_series_tensor, bin_edges_tensor, order=1)
# noise_func = compute_noise_m1

moment_calculator = NthOrderCenteredMoment(tof_series_tensor, bin_edges_tensor, order=2)
noise_func = compute_noise_variance


fetal_comb_filter = CombSeparator(
    fs=sampling_rate,
    f0=fetal_f,
    f1=2 * fetal_f,
    half_width=0.3,
    filter_length=len(time_axis) // 2,  # TODO: Choose appropriate filter length later
)
maternal_comb_filter = CombSeparator(
    fs=sampling_rate,
    f0=maternal_f,
    f1=2 * maternal_f,
    half_width=0.3,
    filter_length=len(time_axis) // 2,  # TODO: Choose appropriate filter length later
)

energy_ratio_metric = EnergyRatioMetric()
contrast_to_noise_metric = ContrastToNoiseMetric(noise_func, tof_series_tensor, bin_edges_tensor, False)
# contrast_to_noise_metric = ContrastToNoiseMetric(noise_func, tof_series_tensor, bin_edges_tensor, True)

# Optimization setup
optimizer = optim.Adam([window_exponents], lr=0.01)
num_epochs = 2000
metric_array = []
for epoch in range(num_epochs):
    optimizer.zero_grad()
    
    # Reparameterize window using exponentiation to ensure positivity & normalize
    window = torch.exp(window_exponents)
    # window = window_exponents
    
    
    window_normalized = window / (torch.norm(window) + 1e-20)
    compact_stats = moment_calculator(window_normalized)  # Shape: (num_timepoints,)

    compact_stats_reshaped = compact_stats.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, num_timepoints)
    fetal_filtered_signal = fetal_comb_filter(compact_stats_reshaped)
    maternal_filtered_signal = maternal_comb_filter(compact_stats_reshaped)
    
    selectivity_metric = torch.sqrt(energy_ratio_metric(fetal_filtered_signal, maternal_filtered_signal))
    # Compute the contrast-to-noise pre-filter
    contrast_value = contrast_to_noise_metric(window_normalized, fetal_filtered_signal)
    # contrast_value = 40 + contrast_value)
    final_metric = selectivity_metric * contrast_value
    loss = -final_metric
    loss.backward()
    optimizer.step()

    metric_array.append([contrast_value.item(), selectivity_metric.item(), final_metric.item()])
    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch + 1}/{num_epochs}"
            f"Selectivity: {selectivity_metric.item():.6f}, "
            f"Contrast: {contrast_value.item():.6f}, "
            f"Final Metric: {final_metric.item():.6f}")

print("Optimization complete.")
window_normalized = torch.exp(window_exponents)
window_normalized = window_normalized / (torch.norm(window_normalized) + 1e-20)
print("Final Window :", window_normalized.detach().numpy())


# Plot loss curve & window
metric_array = np.array(metric_array)
plt.subplots(1, 2, figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(metric_array[:, 0], label="Contrast")
plt.plot(metric_array[:, 1], label="Selectivity")
plt.plot(metric_array[:, 2], label="Final Metric")
plt.xlabel("Epoch")
plt.ylabel("Metric Value")
plt.legend()
plt.title("Metric Values Over Epochs")


plt.subplot(1, 2, 2)
plt.plot(window_normalized.detach().numpy(), marker='o')
plt.xlabel("Bin Index")
plt.ylabel("Window Value")
plt.title("Optimized Window")
plt.tight_layout()
plt.show()

plt.savefig("./figures/optimize_loop4_results.png")

