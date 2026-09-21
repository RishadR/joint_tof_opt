"""
Core modules for joint TOF optimization to make life easier.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
from typing_extensions import override

from joint_tof_opt.config_loader import ToFConfig


@dataclass
class ToFData:
    """
    Dataclass to store discretized 2D ToF data

    Attributes:
    -----------------------
    tof_series: 2D tensor of shape (num_timepoints, num_bins) - holds the ToF histograms (Means for each point)
    bin_edges: 1D tensor of shape (num_bins + 1)
    bin_centers: 1D tensor of shape (num_bins)
    var_series: 2D tensor of shape (num_timepoints, num_bins) - holds corresponding variance values for each point
    inner_moments: Dictionary to hold precomputed inner moments if needed. Keys are moment orders (float),
        values are 2D tensors of shape (num_timepoints, num_bins) corresponding to that moment.
    meta_data: Optional dictionary to hold any additional metadata. This might include:
        - time_axis: 1D numpy array of time values corresponding to each row of tof_series
        - sd_distance: Source-detector distance in mm
        - maternal_hb_series: 1D numpy array of maternal hemoglobin concentrations over time
        - fetal_hb_series: 1D numpy array of fetal hemoglobin concentrations over time
        - wavelength: Wavelength of light used in nm
        - weight_threshold_fraction: Fraction of total photon weight considered
        - fetal_f: Fetal blood volume fraction
        - maternal_f: Maternal blood volume fraction
        - sampling_rate: Sampling rate in Hz

    Additional Notes:
    -----------------------
    If we ever want to compute higher order moments (>2), we need to store the actual arrival times and weights.
    However, for this paper, that is not necessary. If needed, we can always put that in meta_data.
    """

    tof_series: torch.Tensor
    bin_edges: torch.Tensor
    bin_centers: torch.Tensor
    var_series: torch.Tensor
    inner_moments: dict[float, torch.Tensor] = field(default_factory=dict)
    meta_data: dict[str, Any] | None = None

    @classmethod
    def from_npz(cls, npz_path: Path | BytesIO) -> "ToFData":
        """
        Create a ToFData instance from a .npz file, either on disk or already loaded into memory.

        :param npz_path: Path to the .npz file, or an in-memory buffer of its contents, containing tof_series,
        bin_edges, bin_centers, var_series, and optional meta_data
        :type npz_path: Path | BytesIO
        :return: ToFData instance
        :rtype: ToFData
        """
        data = np.load(npz_path)

        tof_series = torch.tensor(data["tof_dataset"], dtype=torch.float32)
        bin_edges = torch.tensor(data["bin_edges"], dtype=torch.float32)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        var_series = torch.tensor(data["var_dataset"], dtype=torch.float32)

        # Load meta_data if it exists in the npz file
        meta_data = {}
        all_keys = list(data.keys())
        meta_data_keys = [
            x
            for x in all_keys
            if x not in ["tof_dataset", "bin_edges", "var_dataset", "bin_centers"] and not x.startswith("inner_moment_")
        ]
        for key in meta_data_keys:
            meta_data[key] = data[key]
        inner_moments = {}
        for key in all_keys:
            if key.startswith("inner_moment_"):
                order_str = key[len("inner_moment_") :]
                inner_moments[float(order_str)] = torch.tensor(data[key], dtype=torch.float32)
        return cls(
            tof_series=tof_series,
            bin_edges=bin_edges,
            bin_centers=bin_centers,
            var_series=var_series,
            inner_moments=inner_moments,
            meta_data=meta_data,
        )

    def to_npz(self, npz_path: Path) -> None:
        """
        Save the ToFData instance to a .npz file.

        :param npz_path: Path to save the .npz file
        :type npz_path: Path
        """
        inner_moments_kwargs = {
            "inner_moment_" + str(order): moment.numpy() for order, moment in self.inner_moments.items()
        }
        save_dict = {
            "tof_dataset": self.tof_series.numpy(),
            "bin_edges": self.bin_edges.numpy(),
            "var_dataset": self.var_series.numpy(),
            "bin_centers": self.bin_centers.numpy(),
            **inner_moments_kwargs,
        }
        if self.meta_data is not None:
            for key, value in self.meta_data.items():
                save_dict[key] = value
        np.savez(npz_path, **save_dict, allow_pickle=True)


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

    def __init__(self, tof_data: ToFData):
        super().__init__()
        self.tof_data: ToFData = tof_data
        # Keep the relevant attributes for backward compatibility
        self.tof_series: torch.Tensor = tof_data.tof_series
        self.bin_edges: torch.Tensor = tof_data.bin_edges
        self.bin_centers: torch.Tensor = tof_data.bin_centers
        self.meta_data: dict[str, Any] | None = tof_data.meta_data

    @override
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
    - tof_data : TOF data to optimize on. This also contains metadata like bin edges, sampling rate, etc.
    - tof_series : Torch tensor of the TOF series data. Each row is a separate DTOF measurement.
    - bin_edges : Torch tensor of the bin edges of the DTOF. Correponds to the columns of the TOF series.
    - time_axis : Time axis corresponding each row of the TOF series.
    - moment_module : The moment calculation module (e.g., WindowedSum, NthOrderMoment) based on the measurand.
    - training_curves : Numpy array to store training curves (if any).
    - training_curve_labels : List of labels for the training curves. Leave empty if no curves are stored.
    - window : Torch tensor to store the optimized window. Leave empty if not yet optimized.
    """

    def __init__(self, tof_data: ToFData, measurand: CompactStatProcess, lr: float = 0.01):
        self.tof_data: ToFData = tof_data
        assert self.tof_data.meta_data is not None, "ToFData meta_data cannot be None"
        assert "time_axis" in self.tof_data.meta_data, "ToFData meta_data must contain time_axis"
        self.moment_module: CompactStatProcess = measurand
        self.training_curves: npt.NDArray[np.float64] = np.array([])
        self.training_curve_labels: list[str] = []
        self.window: torch.Tensor = torch.tensor([])
        self.lr: float = lr
        self.final_signal: torch.Tensor = torch.tensor([])

    @abstractmethod
    def optimize(self):
        pass

    @override
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
    - gen_config : ToFConfig. DTOF generation configs. This will be used on the ppath file to generate the ToF data.

    Stored Attributes:
    -----------------------
    - final_metric : Float to store the final evaluation metric after calling evaluate().

    Things to Implement in Subclasses:
    -----------------------
    - self.evaluate() : Method to perform the evaluation and populate self.final_metric and return a float
    - __str__() : String representation of the evaluator for easy identification.
    - self.get_log() : Method to return a dictionary of relevant evaluation metrics and their corresponding values.
    """

    def __init__(
        self, ppath_file: Path, window: torch.Tensor, measurand: str, gen_config: ToFConfig
    ):
        self.ppath_file: Path = ppath_file
        self.window: torch.Tensor = window
        self.measurand: str = measurand
        self.final_metric: float = 0.0
        self.gen_config: ToFConfig = gen_config

    @abstractmethod
    def evaluate(self) -> float:
        pass

    @override
    @abstractmethod
    def __str__(self) -> str:
        pass

    @abstractmethod
    def get_log(self) -> dict[str, Any]:
        pass


class NoiseCalculator(ABC):
    """
    Base class for noise calculation modules.

    Things to Implement in Subclasses:
    -----------------------
    - self.compute_noise() : Method to compute the analytical noise for a given ToFData instance and window.
    sum_axis controls how the (num_timepoints, num_bins) computation is collapsed: 1 (default) sums across
    bins, returning a 1D tensor of length num_timepoints; -1 skips summing entirely, returning the full
    (num_timepoints, num_bins) tensor. Not every subclass can support sum_axis=-1 - see its own docstring.
    - __str__() : String representation of the noise calculator for easy identification.

    Extra:
    -----------------------
    If you need to have other parameters go into the noise calculator, you can add them to the __init__ method
    """

    @abstractmethod
    def compute_noise(self, tof_data: ToFData, window: torch.Tensor, sum_axis: int = 1) -> torch.Tensor:
        pass

    @override
    @abstractmethod
    def __str__(self) -> str:
        pass


class ToFModifier(ABC):
    """
    Base class for ToF modifiers that can be applied to ToFData instances.

    Things to Implement in Subclasses:
    -----------------------
    - self.modify() : Method to take in a ToFData instance and return a modified ToFData instance.
    - __str__() : String representation of the modifier for easy identification.

    Extra:
    -----------------------
    If you need to have other parameters go into the modifier, you can add them to the __init__ method
    """

    @abstractmethod
    def modify(self, tof_data: ToFData) -> ToFData:
        pass

    @override
    @abstractmethod
    def __str__(self) -> str:
        pass
