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
from typing import Any, Literal


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

    def __init__(self, tof_dataset_path: Path, measurand: str | nn.Module):
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

    Things to Implement in Subclasses:
    -----------------------
    - self.evaluate() : Method to perform the evaluation and populate self.final_metric. Ideally a float or a
    tuple of floats.
    - __str__() : String representation of the evaluator for easy identification.
    """

    def __init__(self, ppath_file: Path, window: torch.Tensor, measurand: str | nn.Module):
        self.ppath_file = ppath_file
        self.window = window
        self.measurand = measurand
        self.final_metric = None

    @abstractmethod
    def evaluate(self) -> Any:
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass
