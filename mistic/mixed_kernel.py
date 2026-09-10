"""Composable positive-semidefinite kernels for MISTIC."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real

import numpy as np

from .utility import kernelWrapper


@dataclass(frozen=True)
class _Node:
    operation: str
    children: tuple = ()
    value: object = None


class MixedKernel:
    """A kernel expression made from sums, products, and positive integer powers.

    Base kernels may be combined directly with Python operators::

        linear = kernelWrapper("linear", name="linear")
        rbf = kernelWrapper("rbf", name="local", features=[0, 2])
        kernel = 0.25 * linear + 0.75 * rbf**2

    Parameters can be shared (``gamma``), addressed by leaf name
    (``local__gamma``), or addressed by zero-based leaf position
    (``kernel1__gamma``). Named scalar coefficients can be introduced with
    :meth:`weighted_sum` and are read from the same parameter dictionary.
    """

    def __init__(self, expression, feature_sets=None):
        self._root = self._coerce(expression)
        self.feature_sets = feature_sets
        self._validate()

    @classmethod
    def from_operand(cls, operand):
        return operand if isinstance(operand, cls) else cls(operand)

    @classmethod
    def sum(cls, *kernels):
        if not kernels:
            raise ValueError("at least one kernel is required")
        return cls(_Node("sum", tuple(cls._coerce(k) for k in kernels)))

    @classmethod
    def product(cls, *kernels):
        if not kernels:
            raise ValueError("at least one kernel is required")
        return cls(_Node("product", tuple(cls._coerce(k) for k in kernels)))

    @classmethod
    def weighted_sum(cls, kernels, weights=None, weight_parameters=None, feature_sets=None):
        kernels = tuple(kernels)
        if not kernels:
            raise ValueError("at least one kernel is required")
        if weights is not None and weight_parameters is not None:
            raise ValueError("use either weights or weight_parameters, not both")
        if weights is None:
            weights = weight_parameters or np.ones(len(kernels)) / len(kernels)
        if len(weights) != len(kernels):
            raise ValueError("weights must have the same length as kernels")
        terms = [_Node("scale", (cls._coerce(k),), w) for k, w in zip(kernels, weights)]
        return cls(_Node("sum", tuple(terms)), feature_sets=feature_sets)

    @staticmethod
    def _coerce(value):
        if isinstance(value, MixedKernel):
            return value._root
        if isinstance(value, _Node):
            return value
        if isinstance(value, kernelWrapper):
            return _Node("kernel", value=value)
        if isinstance(value, Real):
            return _Node("constant", value=float(value))
        raise TypeError("kernel expressions accept kernelWrapper, MixedKernel, or real scalars")

    def __add__(self, other):
        return MixedKernel(_Node("sum", (self._root, self._coerce(other))))

    def __radd__(self, other):
        return MixedKernel(_Node("sum", (self._coerce(other), self._root)))

    def __mul__(self, other):
        if isinstance(other, Real):
            return MixedKernel(_Node("scale", (self._root,), float(other)))
        return MixedKernel(_Node("product", (self._root, self._coerce(other))))

    def __rmul__(self, other):
        if isinstance(other, Real):
            return MixedKernel(_Node("scale", (self._root,), float(other)))
        return MixedKernel(_Node("product", (self._coerce(other), self._root)))

    def __pow__(self, power):
        if not isinstance(power, (int, np.integer)) or power <= 0:
            raise ValueError("kernel powers must be positive integers")
        return MixedKernel(_Node("power", (self._root,), int(power)))

    @property
    def base_kernels(self):
        kernels = []
        seen = set()
        for node in self._walk():
            if node.operation == "kernel" and id(node.value) not in seen:
                kernels.append(node.value)
                seen.add(id(node.value))
        return kernels

    @property
    def kernel_names(self):
        """Stable names used by kernel-specific feature mappings."""
        return tuple(kernel.name or f"kernel{index}" for index, kernel in enumerate(self.base_kernels))

    def params_needed(self):
        result = []
        for index, kernel in enumerate(self.base_kernels):
            prefix = kernel.name or f"kernel{index}"
            result.extend(f"{prefix}__{name}" for name in kernel.params_needed())
        result.extend(str(n.value) for n in self._walk() if n.operation == "scale" and isinstance(n.value, str))
        return result

    def _walk(self, node=None):
        node = self._root if node is None else node
        yield node
        for child in node.children:
            yield from self._walk(child)

    def _validate(self):
        kernels = self.base_kernels
        if not kernels:
            raise ValueError("a mixed kernel must contain at least one base kernel")
        names = [k.name for k in kernels if k.name is not None]
        if len(names) != len(set(names)):
            raise ValueError("base-kernel names must be unique")
        if self.feature_sets is not None and len(self.feature_sets) != len(kernels):
            raise ValueError("feature_sets must have one entry per base kernel")
        for node in self._walk():
            if node.operation == "scale" and isinstance(node.value, Real) and node.value < 0:
                raise ValueError("kernel weights must be non-negative")

    def _leaf_context(self, feature_index, parameters):
        feature_mapping = feature_index if isinstance(feature_index, Mapping) else None
        global_features = None if feature_mapping is not None else np.asarray(feature_index, dtype=int)
        contexts = {}
        for index, kernel in enumerate(self.base_kernels):
            prefix = kernel.name or f"kernel{index}"
            allowed = kernel.features
            if self.feature_sets is not None:
                allowed = self.feature_sets[index]
            if feature_mapping is None:
                features = global_features
            else:
                if prefix in feature_mapping:
                    features = np.asarray(feature_mapping[prefix], dtype=int)
                elif index in feature_mapping:
                    features = np.asarray(feature_mapping[index], dtype=int)
                else:
                    features = np.asarray([], dtype=int)
            if allowed is not None:
                features = features[np.isin(features, np.asarray(allowed, dtype=int))]
            local = {}
            for name in kernel.params_needed():
                if name in parameters:
                    local[name] = parameters[name]
                indexed = f"kernel{index}__{name}"
                named = f"{prefix}__{name}"
                if indexed in parameters:
                    local[name] = parameters[indexed]
                if named in parameters:
                    local[name] = parameters[named]
            contexts[id(kernel)] = (features, local)
        return contexts

    @staticmethod
    def _coefficient(value, parameters):
        coefficient = parameters[value] if isinstance(value, str) else value
        coefficient = float(coefficient)
        if coefficient < 0:
            raise ValueError("kernel weights must be non-negative")
        return coefficient

    def _evaluate(self, node, X, Y, contexts, parameters, wrt=None):
        if node.operation == "kernel":
            kernel = node.value
            features, local = contexts[id(kernel)]
            if features.size == 0:
                return None
            if wrt is None:
                return kernel.compute(X, features, local, Y)
            value = kernel.compute(X, features, local, Y)
            gradient = np.zeros_like(value)
            if wrt in features:
                gradient = kernel.compute_gradient(X, features, wrt, local, Y)
            return value, gradient
        if node.operation == "constant":
            shape = (len(X), len(X) if Y is None else len(Y))
            value = np.full(shape, node.value)
            return value if wrt is None else (value, np.zeros(shape))
        values = [self._evaluate(c, X, Y, contexts, parameters, wrt) for c in node.children]
        values = [v for v in values if v is not None]
        if not values:
            return None
        if node.operation == "scale":
            coefficient = self._coefficient(node.value, parameters)
            return coefficient * values[0] if wrt is None else tuple(coefficient * x for x in values[0])
        if node.operation == "power":
            power = node.value
            if wrt is None:
                return values[0] ** power
            value, gradient = values[0]
            return value**power, power * value ** (power - 1) * gradient
        if wrt is None:
            if node.operation == "sum":
                return sum(values[1:], values[0])
            return np.prod(np.stack(values), axis=0)
        matrices, gradients = zip(*values)
        if node.operation == "sum":
            return sum(matrices[1:], matrices[0]), sum(gradients[1:], gradients[0])
        product = np.prod(np.stack(matrices), axis=0)
        gradient = np.zeros_like(product)
        for index, child_gradient in enumerate(gradients):
            others = np.prod(np.stack(matrices[:index] + matrices[index + 1 :]), axis=0) if len(matrices) > 1 else 1.0
            gradient += child_gradient * others
        return product, gradient

    def compute(self, X, feature_index, parameters=None, Y=None):
        parameters = {} if parameters is None else parameters
        contexts = self._leaf_context(feature_index, parameters)
        result = self._evaluate(self._root, X, Y, contexts, parameters)
        if result is None:
            raise ValueError("no active kernels have selected features")
        return result

    def compute_gradient(self, X, feature_index, wrt, parameters=None, Y=None):
        if Y is None:
            raise ValueError("mixed-kernel input gradients require Y")
        parameters = {} if parameters is None else parameters
        contexts = self._leaf_context(feature_index, parameters)
        result = self._evaluate(self._root, X, Y, contexts, parameters, wrt=wrt)
        if result is None:
            raise ValueError("no active kernels have selected features")
        return result[1]


class MixedkernelWrapper(MixedKernel):
    """Compatibility constructor for the original mixed-kernel draft API."""

    def __init__(self, base_kernels, weights=None, mix_type="product", feature_sets=None):
        if mix_type == "product":
            expression = MixedKernel.product(*base_kernels)._root
        elif mix_type in ("sum", "weightedsum"):
            expression = MixedKernel.weighted_sum(base_kernels, weights=weights)._root
        else:
            raise ValueError("mix_type must be 'product', 'sum', or 'weightedsum'")
        super().__init__(expression, feature_sets=feature_sets)

    @property
    def kernel_feature_sets(self):
        return self.feature_sets

    @kernel_feature_sets.setter
    def kernel_feature_sets(self, value):
        if value is not None and len(value) != len(self.base_kernels):
            raise ValueError("kernel_feature_sets must have one entry per base kernel")
        self.feature_sets = value
