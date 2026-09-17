"""
Loads tof_config.yaml into a validated, typed config object. Array fields are exposed as
numpy arrays rather than plain lists.
"""

from pathlib import Path
from typing import Annotated, Literal, cast

import numpy as np
import yaml
from pydantic import BaseModel, BeforeValidator, ConfigDict

# 1D shape, not npt.NDArray's `tuple[int, ...]`, so pyright can type element access/iteration
Vector1D = np.ndarray[tuple[int], np.dtype[np.float64]]
IntVector1D = np.ndarray[tuple[int], np.dtype[np.int64]]


def _to_float_array(value: object) -> Vector1D:
    # np.asarray's stub can't prove the result is exactly 1D; these fields always are (from yaml lists)
    return cast(Vector1D, np.asarray(value, dtype=np.float64))


def _to_int_array(value: object) -> IntVector1D:
    return cast(IntVector1D, np.asarray(value, dtype=np.int64))


FloatArray = Annotated[Vector1D, BeforeValidator(_to_float_array)]
IntArray = Annotated[IntVector1D, BeforeValidator(_to_int_array)]


class ToFConfig(BaseModel):
    """
    What configurations to use when generating Time-of-Flight (ToF) & its corresponding time series.

    Use load_tof_config( ) to create one from a JSON. Check `experiments/tof_config.yaml` for an example.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)  # pyright: ignore[reportUnannotatedClassAttribute]

    # Baseline Simulation Parameters
    total_photon_count: int
    wavelength: float
    epidermis_thickness: int
    donut_half_thickness: float

    # Time series parameters
    datapoint_count: int
    maternal_f: float
    fetal_f: float
    end_sec: float
    sampling_rate: float

    # Tissue Parameters
    maternal_hb_base: float
    fetal_hb_base: float
    maternal_saturation: float
    fetal_saturation: float
    light_speeds: FloatArray
    epi_thickness_mm: int
    derm_thickness_mm: int

    # Data selection
    selected_sdd_index: int

    # Histogram parameters
    bin_count: int
    time_limit_or_threshold: Literal["timelimit", "weightthreshold"]
    time_limit: FloatArray
    weight_threshold_fraction: float

    # Extra Info
    sdd_distances: IntArray
    dermis_thicknesses: IntArray


def load_tof_config(path: Path) -> ToFConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)  # pyright: ignore[reportAny]
    return ToFConfig.model_validate(raw)
