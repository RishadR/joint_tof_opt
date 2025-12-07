"""
Core modules for joint TOF optimization to make life easier.
"""

import torch
import torch.nn as nn
from joint_tof_opt.compact_stat_process import NthOrderMoment, NthOrderCenteredMoment, WindowedSum
from joint_tof_opt.noise_calc import compute_noise_m1, compute_noise_variance, compute_noise_window_sum
from typing import Literal


# Single source of truth: define moment configurations once
MOMENT_CONFIGS = {
    "abs": lambda tof, edges: WindowedSum(tof, edges),
    "m1": lambda tof, edges: NthOrderMoment(tof, edges, order=1),
    "V": lambda tof, edges: NthOrderCenteredMoment(tof, edges, order=2),
}

named_moment_types = list(MOMENT_CONFIGS.keys())


def get_named_moment_module(
    moment_type: str,
    tof_series_tensor: torch.Tensor,
    bin_edges_tensor: torch.Tensor,
) -> nn.Module:
    if moment_type not in MOMENT_CONFIGS:
        raise ValueError(f"Invalid moment type: {moment_type}")

    config = MOMENT_CONFIGS[moment_type]
    return config(tof_series_tensor, bin_edges_tensor)


noise_func_table = {
    "abs": compute_noise_window_sum,
    "m1": compute_noise_m1,
    "V": compute_noise_variance,
}
