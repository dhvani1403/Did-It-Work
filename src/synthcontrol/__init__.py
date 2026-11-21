"""Synthetic control from scratch: solvers, placebo inference, DiD and a factor-model simulator."""

from . import dgp, did, inference, simplex, solvers, synth
from .simplex import project_simplex
from .synth import fit

__all__ = ["dgp", "did", "inference", "simplex", "solvers", "synth", "fit", "project_simplex"]
