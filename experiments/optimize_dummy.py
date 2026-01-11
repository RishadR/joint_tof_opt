"""
Dummy optimizers that always return a fixed window for comparison purposes.
"""
"""
Compare the Sensitivity between optmized vs. non-optimized windows and visualize the results.
"""

from pathlib import Path
import torch
import torch.nn as nn
from joint_tof_opt import (
    OptimizationExperiment,
    get_named_moment_module,
    OptimizationExperiment,
    CompactStatProcess,
    ToFData,
)
from optimize_liu import LiuOptimizer
from optimize_loop_paper import DIGSSOptimizer

class DummyOptimizationExperiment(OptimizationExperiment):
    """
    Always returns a unit window for testing purposes.
    """
    def __init__(self, tof_dataset_path: Path, measurand: CompactStatProcess | str):
        if isinstance(measurand, str):
            tof_data = ToFData.from_npz(tof_dataset_path)
            measurand = get_named_moment_module(measurand, tof_data)
        super().__init__(tof_dataset_path, measurand)
    
    def optimize(self) -> None:
        self.window = torch.ones(self.tof_data.tof_series.shape[1], dtype=torch.float32)
        self.window /= torch.norm(self.window, p=2)
        self.final_signal = self.moment_module(self.window) 
        self.training_curves = []
    
    def __str__(self) -> str:
        return "DummyUnitWindowGenerator"
    
    def components(self) -> dict[str, nn.Module]:
        return {}