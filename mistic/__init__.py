"""Interpretable and explainable support-vector-machine tools."""

from .cvSet import cvSet
from .svmSet import svmSet
from .explanations import IntegratedGradientsResult
from .utility import (combined_rank, kernelWrapper, paramSet, perDiff,
                      score_ocsvm, score_svc, score_svr)

__version__ = "0.1.1"

__all__ = [
    "combined_rank",
    "IntegratedGradientsResult",
    "cvSet",
    "kernelWrapper",
    "paramSet",
    "perDiff",
    "score_ocsvm",
    "score_svc",
    "score_svr",
    "svmSet",
]
