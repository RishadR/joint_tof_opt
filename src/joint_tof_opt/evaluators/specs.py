"""
Single source of truth for the tunable-hyperparameter defaults of the Evaluator subclasses in
joint_tof_opt.evaluators.paper (PaperEvaluator, AltPaperEvaluator2, AltPaperEvaluator3), plus which
evaluator to use and the noise-injection settings shared across experiments/*.py.

Use load_evaluator_specs() to load experiments/evaluator_specs.yaml, then wire each field straight into
the corresponding __init__'s parameter defaults - see joint_tof_opt.optimizers.specs for the pattern.
DEFAULT_SPECS_PATH points at that yaml file (it lives in experiments/, alongside the other
experiment-tunable config, not next to this package).
"""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel

EvaluatorType = Literal["paper", "alt_paper2", "alt_paper3"]

# src/joint_tof_opt/evaluators/specs.py -> parents[3] is the repo root.
DEFAULT_SPECS_PATH: Path = Path(__file__).resolve().parents[3] / "experiments" / "evaluator_specs.yaml"


class PaperEvaluatorSpec(BaseModel):
    filter_hw: float


class AltPaperEvaluator2Spec(BaseModel):
    filter_hw: float


class AltPaperEvaluator3Spec(BaseModel):
    filter_hw: float


class EvaluatorSpecs(BaseModel):
    """
    How is the system evaluated? Contains
    evaluator_to_use : ['paper', 'alt_paper2', 'alt_paper3']
    inject_noise: bool
    shot_noise_multiplier: float
    instrument_noise_variance: float
    repeats_if_noisy: int
    paper: PaperEvaluatorSpec
    alt_paper2: AltPaperEvaluator2Spec
    alt_paper3: AltPaperEvaluator3Spec
    """
    evaluator_to_use: EvaluatorType
    inject_noise: bool
    shot_noise_multiplier: float
    instrument_noise_variance: float
    repeats_if_noisy: int
    paper: PaperEvaluatorSpec
    alt_paper2: AltPaperEvaluator2Spec
    alt_paper3: AltPaperEvaluator3Spec


def load_evaluator_specs(path: Path) -> EvaluatorSpecs:
    with open(path) as f:
        raw = yaml.safe_load(f)  # pyright: ignore[reportAny]
    return EvaluatorSpecs.model_validate(raw)
