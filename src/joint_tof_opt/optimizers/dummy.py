"""
Dummy optimizer that always returns a fixed window, for comparison purposes.
"""

import torch
import torch.nn as nn

from joint_tof_opt.compact_stat_process import get_named_moment_module
from joint_tof_opt.core import CompactStatProcess, OptimizationExperiment, ToFData
from joint_tof_opt.optimizers.specs import DEFAULT_SPECS_PATH, load_optimizer_specs

_DUMMY_SPEC = load_optimizer_specs(DEFAULT_SPECS_PATH).dummy


class DummyOptimizationExperiment(OptimizationExperiment):
    """
    Always returns a unit window for testing purposes.
    Arguments:
        tof_data: ToFData instance to optimize on.
        measurand: CompactStatProcess instance or name of the moment to optimize.
        norm: If specified, normalizes the window to have this p-norm. Ex: norm=1 for L1 norm.

    """

    def __init__(self, tof_data: ToFData, measurand: CompactStatProcess | str, norm: float | None = _DUMMY_SPEC.norm):
        if isinstance(measurand, str):
            measurand = get_named_moment_module(measurand, tof_data)
        super().__init__(tof_data, measurand)
        self.norm = norm

    def optimize(self) -> None:
        self.window = torch.ones(self.tof_data.tof_series.shape[1], dtype=torch.float32)
        if self.norm is not None:
            self.window /= torch.norm(self.window, p=self.norm)
        self.final_signal = self.moment_module.forward(self.window)
        self.training_curves = []

    def __str__(self) -> str:
        return "DummyUnitWindowGenerator"

    def components(self) -> dict[str, nn.Module]:
        return {}
