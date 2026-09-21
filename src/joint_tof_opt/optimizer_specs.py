"""
Single source of truth for the tunable-hyperparameter defaults of the optimize_*.py experiments
(DIGSSOptimizer/BoxCarOptimizer, LiuOptimizer, AltLiuOptimizer, DummyOptimizationExperiment).

Use load_optimizer_specs() to load experiments/optimizer_specs.yaml, then wire each field straight into
the corresponding __init__'s parameter defaults - see optimize_loop_paper.py for the pattern.
"""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel

RegType = Literal["l1", "l2"]
FilterType = Literal["comb", "fourier", "psafe_same_width", "psafe_true_width", "comb_psafe_hybrid"]
NormalizationScheme = Literal["unit_sum", "unit_max"]
DtofSelection = Literal["mean", "median", "first"]


class DIGSSOptimizerSpec(BaseModel):
    max_epochs: int
    lr: float
    filter_hw: float
    patience: int
    grad_clip: bool
    reg_type: RegType
    reg_weight: float
    window_smoothening: bool
    normalize_reward: bool
    filter_type: FilterType
    normalization_scheme: NormalizationScheme
    use_window_post_process: bool
    use_snr_left_bound: bool


class BoxCarOptimizerSpec(DIGSSOptimizerSpec):
    """BoxCarOptimizer takes the same __init__ params as DIGSSOptimizer, but keeps its own defaults."""


class LiuOptimizerSpec(BaseModel):
    dtof_to_find_max_on: DtofSelection
    half_width: float
    harmonic_count: int
    norm: float | None


class AltLiuOptimizerSpec(BaseModel):
    dtof_to_find_max_on: DtofSelection
    half_width: float
    harmonic_count: int
    norm: float | None


class DummyOptimizerSpec(BaseModel):
    norm: float | None


class OptimizerSpecs(BaseModel):
    digss: DIGSSOptimizerSpec
    boxcar: BoxCarOptimizerSpec
    liu: LiuOptimizerSpec
    alt_liu: AltLiuOptimizerSpec
    dummy: DummyOptimizerSpec


def load_optimizer_specs(path: Path) -> OptimizerSpecs:
    with open(path) as f:
        raw = yaml.safe_load(f)  # pyright: ignore[reportAny]
    return OptimizerSpecs.model_validate(raw)
