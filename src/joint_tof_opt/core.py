"""
Core modules for joint TOF optimization to make life easier.
"""

from abc import ABC, abstractmethod
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from typing import Any, Literal


class CompactStatProcess(ABC, nn.Module):
    """
    Abstract base class for computing compact statistics from TOF data.

    Subclasses should implement the forward method to define how the compact statistic
    is computed from the TOF histograms and a window function.

    The initializer takes in the TOF series and bin edges, which are used in the computation. Additionally, you can
    pass in the generate_tof metadata - which should include
        - time_axis
        - sd_distance
        - maternal_hb_series
        - fetal_hb_series
        - wavelength
        - weight_threshold_fraction
        - fetal_f
        - maternal_f
        - sampling_rate
    """

    def __init__(self, tof_series: torch.Tensor, bin_edges: torch.Tensor, meta_data: dict | None = None):
        super().__init__()
        self.tof_series = tof_series
        self.bin_edges = bin_edges
        self.meta_data = meta_data
        self.bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    @abstractmethod
    def forward(self, window: torch.Tensor) -> torch.Tensor:
        """
        Compute the compact statistic given a window function.

        :param window: 1D tensor with same length as number of bins.
        :type window: torch.Tensor
        :return: 1D tensor of computed compact statistics for each ToF (flattened).
        :rtype: torch.Tensor
        """
        pass



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

    def __init__(self, tof_dataset_path: Path, measurand: CompactStatProcess, lr: float = 0.01):
        self.tof_dataset_path = tof_dataset_path
        self.tof_data = np.load(tof_dataset_path)
        self.tof_series = torch.tensor(self.tof_data["tof_dataset"], dtype=torch.float32)
        self.bin_edges = torch.tensor(self.tof_data["bin_edges"], dtype=torch.float32)
        self.time_axis = self.tof_data["time_axis"]
        self.moment_module = measurand
        self.training_curves = np.array([])
        self.training_curve_labels = []
        self.window = torch.tensor([])
        self.lr = lr
        self.final_signal = torch.tensor([])

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


class NoiseCalculator(ABC):
    """
    Base class for noise calculation modules.
    
    Things to Implement in Subclasses:
    -----------------------
    - self.compute_noise() : Method to compute the analytical noise for a given ToF series, bin edges, and window. 
    The method should return a 1D tensor of noise values - same length as number of ToF series.
    - __str__() : String representation of the noise calculator for easy identification.
    
    Extra:
    -----------------------
    If you need to have other parameters go into the noise calculator, you can add them to the __init__ method 
    """

    @abstractmethod
    def compute_noise(self, tof_series: torch.Tensor, bin_edges: torch.Tensor, window: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass
