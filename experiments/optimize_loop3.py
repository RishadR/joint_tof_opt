"""
Second implementation of the optimization.

Process Flow:
1. Load ./data/generated_tof_set.npz containing a 2D array of ToF series
2. Compute First Order Moment with a parameterized window
3. Apply Sinc Comb Filter to extract fetal signal using a known frequency 
5. Compute th e Energy Ratio Metric between the fetal filtered and the true unwindowed signal
6. Optimize the window parameters to maximize the metric 
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from joint_tof_opt.compact_stat_process import NthOrderMoment, NthOrderCenteredMoment
from joint_tof_opt.signal_process import CombSeparator
from joint_tof_opt.metric_process import EnergyRatioMetric

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
window_exponents = nn.Parameter(torch.ones(num_bins, dtype=torch.float32, requires_grad=True))
# Initialize modules
# moment_calculator = NthOrderMoment(tof_series_tensor, bin_edges_tensor, order=1)
moment_calculator = NthOrderCenteredMoment(tof_series_tensor, bin_edges_tensor, order=2)
fetal_comb_filter = CombSeparator(
    fs=sampling_rate,
    f0=fetal_f,
    f1=2 * fetal_f,
    half_width=0.1,
    filter_length=len(time_axis) // 2,  # TODO: Choose appropriate filter length later
)


energy_ratio_metric = EnergyRatioMetric()

# Optimization setup
optimizer = optim.Adam([window_exponents], lr=0.01)
num_epochs = 1000
loss_array = []
for epoch in range(num_epochs):
    optimizer.zero_grad()
    
    # Reparameterize window using exponentiation to ensure positivity
    window = torch.exp(window_exponents)
    # Normalize window to unit energy
    window_normalized = window / (torch.norm(window) + 1e-20)
    
    # Step 1: Compute first order moment with current window
    compact_stats = moment_calculator(window_normalized)  # Shape: (num_timepoints,)
    original_energy = torch.norm(compact_stats)

    # Step 2: Apply comb filter to extract fetal signal
    compact_stats_reshaped = compact_stats.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, num_timepoints)
    fetal_filtered_signal = fetal_comb_filter(compact_stats_reshaped)
    
    # Step 3: Compute energy ratio metric
    metric_value = energy_ratio_metric(fetal_filtered_signal, original_energy)
    # We want to maximize the metric, so minimize the negative
    loss = -metric_value
    loss.backward()
    optimizer.step()

    loss_array.append(loss.item())
    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Metric: {metric_value.item():.6f}")

print("Optimization complete.")
window_normalized = torch.exp(window_exponents)
window_normalized = window_normalized / (torch.norm(window_normalized) + 1e-20)
print("Final Window :", window_normalized.detach().numpy())
