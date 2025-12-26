from joint_tof_opt.core import OptimizationExperiment, Evaluator, NoiseCalculator
from joint_tof_opt.compact_stat_process import (
    WindowedSum,
    NthOrderMoment,
    NthOrderCenteredMoment,
    CompactStatProcess,
    get_named_moment_module,
    named_moment_types,
)
from joint_tof_opt.signal_process import CombSeparator
from joint_tof_opt.metric_process import EnergyRatioMetric, ContrastToNoiseMetric
from joint_tof_opt.noise_calc import (
    WindowSumNoiseCalculator,
    FirstMomentNoiseCalculator,
    VarianceNoiseCalculator,
    compute_noise_window_sum,
    compute_noise_m1,
    compute_noise_variance,
    get_noise_calculator,
)
from joint_tof_opt.tof_batch_process import generate_tof

# Backward-compatible noise function table
noise_func_table = {
    "abs": compute_noise_window_sum,
    "m1": compute_noise_m1,
    "V": compute_noise_variance,
}

__all__ = [
    "CompactStatProcess",
    "WindowedSum",
    "NthOrderMoment",
    "NthOrderCenteredMoment",
    "get_named_moment_module",
    "CombSeparator",
    "EnergyRatioMetric",
    "ContrastToNoiseMetric",
    "NoiseCalculator",
    "WindowSumNoiseCalculator",
    "FirstMomentNoiseCalculator",
    "VarianceNoiseCalculator",
    "noise_func_table",
    "compute_noise_window_sum",
    "compute_noise_m1",
    "compute_noise_variance",
    "named_moment_types",
    "get_noise_calculator",
    "OptimizationExperiment",
    "Evaluator",
    "generate_tof",
    "noise_func_table",
]
