"""Interpretable and explainable support-vector-machine tools."""

from .cvSet import cvSet
from .explanations import BoundaryCounterfactualResult, IntegratedGradientsResult
from .mixed_kernel import MixedKernel, MixedkernelWrapper
from .parameter_space import categorical, integer, loguniform, parameterSpace, uniform
from .svmSet import svmSet
from .utility import (
    combined_rank,
    kernelWrapper,
    paramSet,
    perDiff,
    score_ocsvm,
    score_svc,
    score_svr,
)

__version__ = "0.1.1"

__all__ = [
    "BoundaryCounterfactualResult",
    "IntegratedGradientsResult",
    "MixedKernel",
    "MixedkernelWrapper",
    "categorical",
    "combined_rank",
    "cvSet",
    "integer",
    "kernelWrapper",
    "loguniform",
    "paramSet",
    "parameterSpace",
    "perDiff",
    "score_ocsvm",
    "score_svc",
    "score_svr",
    "svmSet",
    "uniform",
]
