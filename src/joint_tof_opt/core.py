"""
Core modules for joint TOF optimization to make life easier.
"""

from abc import ABC, abstractmethod
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from joint_tof_opt.compact_stat_process import NthOrderMoment, NthOrderCenteredMoment, WindowedSum
from joint_tof_opt.noise_calc import compute_noise_m1, compute_noise_variance, compute_noise_window_sum
from joint_tof_opt.compact_stat_process import CompactStatProcess
from typing import Any, Literal


# Single source of truth: define moment configurations once
MOMENT_CONFIGS = {
    "abs": lambda tof, edges, meta_data: WindowedSum(tof, edges, meta_data=meta_data),
    "m1": lambda tof, edges, meta_data: NthOrderMoment(tof, edges, order=1, meta_data=meta_data),
    "V": lambda tof, edges, meta_data: NthOrderCenteredMoment(tof, edges, order=2, meta_data=meta_data),
}

named_moment_types = list(MOMENT_CONFIGS.keys())


def get_named_moment_module(
    moment_type: str,
    tof_series_tensor: torch.Tensor,
    bin_edges_tensor: torch.Tensor,
    meta_data: dict | None = None,
) -> CompactStatProcess:
    if moment_type not in MOMENT_CONFIGS:
        raise ValueError(f"Invalid moment type: {moment_type}")

    config = MOMENT_CONFIGS[moment_type]
    return config(tof_series_tensor, bin_edges_tensor, meta_data)


noise_func_table = {
    "abs": compute_noise_window_sum,
    "m1": compute_noise_m1,
    "V": compute_noise_variance,
}


class OptimizationExperiment(ABC):
    """
    Base class for optimization experiments on TOF data.

    Things to Implement in Subclasses:
    -----------------------
    - self.optimize() : Method to perform the optimization and populate self.window and self.training_curves.
    - __str__() : String representation of the experiment for easy identification.
    - self.components() : A list of internal components/modules used in the experiment as a dictionary[str, nn.Module].

    Things It Stores:
    -----------------------
    - tof_dataset_path : Path to the TOF dataset (.npz file).
    - tof_data : Loaded TOF data from the dataset. This also contains metadata like bin edges, sampling rate, etc.
    - tof_series : Torch tensor of the TOF series data. Each row is a separate DTOF measurement.
    - bin_edges : Torch tensor of the bin edges of the DTOF. Correponds to the columns of the TOF series.
    - time_axis : Time axis corresponding each row of the TOF series.
    - moment_module : The moment calculation module (e.g., WindowedSum, NthOrderMoment) based on the measurand.
    - training_curves : Numpy array to store training curves (if any).
    - training_curve_labels : List of labels for the training curves. Leave empty if no curves are stored.
    - window : Torch tensor to store the optimized window. Leave empty if not yet optimized.
    """

    def __init__(self, tof_dataset_path: Path, measurand: str | nn.Module, lr: float = 0.01):
        self.tof_dataset_path = tof_dataset_path
        self.tof_data = np.load(tof_dataset_path)
        self.tof_series = torch.tensor(self.tof_data["tof_dataset"], dtype=torch.float32)
        self.bin_edges = torch.tensor(self.tof_data["bin_edges"], dtype=torch.float32)
        self.time_axis = self.tof_data["time_axis"]
        if isinstance(measurand, str):
            self.moment_module = get_named_moment_module(measurand, self.tof_series, self.bin_edges)
        else:
            self.moment_module = measurand
        self.training_curves = np.array([])
        self.training_curve_labels = []
        self.window = torch.tensor([])
        self.lr = lr

    @abstractmethod
    def optimize(self):
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass

    @abstractmethod
    def components(self) -> dict[str, nn.Module]:
        pass


class Evaluator(ABC):
    """
    Base class for evaluating any given window on some partial path data.
    
    Modifiable Attributes:
    -----------------------
    - ppath_file : Path to the partial path file (.json or similar).
    - window : Torch tensor representing the time-gating window.
    - measurand : The measurand to evaluate. Can be a string (named moment) or a custom nn.Module.
    
    Stored Attributes:
    -----------------------
    - final_metric : Float to store the final evaluation metric after calling evaluate().

    Things to Implement in Subclasses:
    -----------------------
    - self.evaluate() : Method to perform the evaluation and populate self.final_metric and return a float
    - __str__() : String representation of the evaluator for easy identification.
    """

    def __init__(self, ppath_file: Path, window: torch.Tensor, measurand: str | CompactStatProcess):
        self.ppath_file = ppath_file
        self.window = window
        self.measurand = measurand
        self.final_metric = None

    @abstractmethod
    def evaluate(self) -> float:
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass
