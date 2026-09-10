"""Space-filling hyperparameter designs for MISTIC model tuning."""

from dataclasses import dataclass
from math import exp, log

import numpy as np
from scipy.stats import qmc

from .utility import paramSet


@dataclass(frozen=True)
class _Uniform:
    low: float
    high: float

    def transform(self, value):
        return self.low + value * (self.high - self.low)


@dataclass(frozen=True)
class _LogUniform:
    low: float
    high: float

    def transform(self, value):
        return exp(log(self.low) + value * (log(self.high) - log(self.low)))


@dataclass(frozen=True)
class _Integer:
    low: int
    high: int

    def transform(self, value):
        return min(self.high, self.low + int(value * (self.high - self.low + 1)))


@dataclass(frozen=True)
class _Categorical:
    values: tuple

    def transform(self, value):
        index = min(len(self.values) - 1, int(value * len(self.values)))
        return self.values[index]


def uniform(low, high):
    """Define a continuous uniform parameter on ``[low, high)``."""
    if not np.isfinite(low) or not np.isfinite(high) or low >= high:
        raise ValueError("uniform bounds must be finite and low < high")
    return _Uniform(float(low), float(high))


def loguniform(low, high):
    """Define a positive log-uniform parameter on ``[low, high)``."""
    if not np.isfinite(low) or not np.isfinite(high) or low <= 0 or low >= high:
        raise ValueError("loguniform bounds must be finite, positive, and low < high")
    return _LogUniform(float(low), float(high))


def integer(low, high):
    """Define a discrete integer parameter on inclusive bounds."""
    if isinstance(low, (bool, np.bool_)) or isinstance(high, (bool, np.bool_)):
        raise TypeError("integer bounds must be integers")
    if not isinstance(low, (int, np.integer)) or not isinstance(high, (int, np.integer)):
        raise TypeError("integer bounds must be integers")
    if low > high:
        raise ValueError("integer bounds must satisfy low <= high")
    return _Integer(int(low), int(high))


def categorical(values):
    """Define a categorical parameter from a non-empty sequence."""
    values = tuple(values)
    if not values:
        raise ValueError("categorical values cannot be empty")
    return _Categorical(values)


class parameterSpace:
    """Define model and kernel parameter spaces for space-filling sampling.

    Values produced by :func:`uniform`, :func:`loguniform`, :func:`integer`,
    and :func:`categorical` are sampled. Any other value is held fixed.
    """

    def __init__(self, model=None, kernel=None):
        self.model = dict(model or {})
        self.kernel = dict(kernel or {})

    @staticmethod
    def _freeze(value):
        if isinstance(value, dict):
            return tuple(sorted((key, parameterSpace._freeze(item)) for key, item in value.items()))
        if isinstance(value, np.ndarray):
            return (value.dtype.str, value.shape, value.tobytes())
        if isinstance(value, (list, tuple)):
            return tuple(parameterSpace._freeze(item) for item in value)
        try:
            hash(value)
            return value
        except TypeError:
            return repr(value)

    def sample(
        self,
        n_trials,
        strategy="latin_hypercube",
        random_state=None,
        optimization="random-cd",
    ):
        """Generate a deterministic list of :class:`paramSet` candidates.

        Discrete transformations can produce duplicates; duplicates are
        removed while preserving the first sampled candidate. The final list
        is grouped by kernel configuration so model-only variants are adjacent.
        """
        if isinstance(n_trials, (bool, np.bool_)) or not isinstance(n_trials, (int, np.integer)):
            raise TypeError("n_trials must be an integer")
        if n_trials < 1:
            raise ValueError("n_trials must be at least 1")
        if strategy != "latin_hypercube":
            raise ValueError("strategy must be 'latin_hypercube'")
        if optimization not in {None, "random-cd", "lloyd"}:
            raise ValueError("optimization must be None, 'random-cd', or 'lloyd'")

        dimensions = []
        for location, parameters in (("model", self.model), ("kernel", self.kernel)):
            for name, specification in parameters.items():
                if isinstance(specification, (_Uniform, _LogUniform, _Integer, _Categorical)):
                    dimensions.append((location, name, specification))

        if dimensions:
            design = qmc.LatinHypercube(
                d=len(dimensions), optimization=optimization, seed=random_state
            ).random(n=int(n_trials))
        else:
            design = np.empty((1, 0))

        candidates = []
        seen = set()
        for point in design:
            model = {
                name: value
                for name, value in self.model.items()
                if not isinstance(value, (_Uniform, _LogUniform, _Integer, _Categorical))
            }
            kernel = {
                name: value
                for name, value in self.kernel.items()
                if not isinstance(value, (_Uniform, _LogUniform, _Integer, _Categorical))
            }
            for coordinate, (location, name, specification) in zip(point, dimensions):
                target = model if location == "model" else kernel
                target[name] = specification.transform(float(coordinate))

            signature = (self._freeze(model), self._freeze(kernel))
            if signature not in seen:
                seen.add(signature)
                candidates.append(paramSet(model=model, kernel=kernel))

        candidates.sort(
            key=lambda candidate: (
                repr(self._freeze(candidate.kernel)),
                repr(self._freeze(candidate.model)),
            )
        )
        return candidates
