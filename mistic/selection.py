"""Kernel-aware feature-selection units and immutable active state."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np


@dataclass(frozen=True, order=True)
class KernelFeatureSet:
    """One selectable perturbation set assigned to one kernel.

    A singleton feature is represented by a one-element ``features`` tuple.
    The same feature tuple assigned to two kernels creates two independent
    selection units.
    """

    kernel_id: str
    features: tuple[int, ...]
    name: str | None = None

    def __post_init__(self):
        if not isinstance(self.kernel_id, str) or not self.kernel_id:
            raise ValueError("kernel_id must be a non-empty string")
        features = tuple(int(feature) for feature in self.features)
        if not features:
            raise ValueError("a kernel feature set cannot be empty")
        if any(feature < 0 for feature in features):
            raise ValueError("feature indices must be nonnegative")
        if len(features) != len(set(features)):
            raise ValueError("feature indices within a perturbation set must be unique")
        object.__setattr__(self, "features", features)


@dataclass(frozen=True)
class KernelFeatureSelectionState:
    """Active-unit indices into an immutable kernel-feature selection space."""

    selection_space: tuple[KernelFeatureSet, ...]
    active: frozenset[int]

    def __post_init__(self):
        space = tuple(self.selection_space)
        active = frozenset(int(index) for index in self.active)
        if len(space) != len(set(space)):
            raise ValueError("selection_space cannot contain duplicate units")
        if any(index < 0 or index >= len(space) for index in active):
            raise IndexError("active selection-unit index is out of range")
        object.__setattr__(self, "selection_space", space)
        object.__setattr__(self, "active", active)

    @classmethod
    def all_active(cls, selection_space: Iterable[KernelFeatureSet]):
        """Construct a state in which every eligible unit is active."""
        space = tuple(selection_space)
        return cls(space, frozenset(range(len(space))))

    @classmethod
    def none_active(cls, selection_space: Iterable[KernelFeatureSet]):
        """Construct a state in which no eligible unit is active."""
        return cls(tuple(selection_space), frozenset())

    @property
    def active_units(self):
        """Active units in stable selection-space order."""
        return tuple(self.selection_space[index] for index in sorted(self.active))

    @property
    def inactive_units(self):
        """Inactive units in stable selection-space order."""
        return tuple(
            unit for index, unit in enumerate(self.selection_space) if index not in self.active
        )

    @property
    def kernel_features(self):
        """Read-only mapping from kernel identifier to active feature union."""
        kernel_ids = dict.fromkeys(unit.kernel_id for unit in self.selection_space)
        values = {}
        for kernel_id in kernel_ids:
            features = {
                feature
                for unit in self.active_units
                if unit.kernel_id == kernel_id
                for feature in unit.features
            }
            array = np.asarray(sorted(features), dtype=int)
            array.flags.writeable = False
            values[kernel_id] = array
        return MappingProxyType(values)

    @property
    def features(self):
        """Sorted union of active original input columns across all kernels."""
        features = sorted(
            {feature for unit in self.active_units for feature in unit.features}
        )
        result = np.asarray(features, dtype=int)
        result.flags.writeable = False
        return result

    @property
    def num_kernel_feature_sets(self):
        """Number of active kernel-specific perturbation sets."""
        return len(self.active)

    @property
    def num_kernel_features(self):
        """Number of active ``(kernel, feature)`` assignments."""
        return sum(len(unit.features) for unit in self.active_units)

    def activate(self, indices):
        """Return a new state with the requested unit indices active."""
        return type(self)(self.selection_space, self.active.union(map(int, indices)))

    def deactivate(self, indices):
        """Return a new state with the requested unit indices inactive."""
        return type(self)(self.selection_space, self.active.difference(map(int, indices)))

    def only(self, indices):
        """Return a new state containing exactly the requested active units."""
        return type(self)(self.selection_space, frozenset(map(int, indices)))


def build_kernel_feature_selection_space(kernel_ids, perturbation_sets):
    """Create the Cartesian assignment of kernels and perturbation sets."""
    return tuple(
        KernelFeatureSet(str(kernel_id), tuple(int(feature) for feature in group))
        for kernel_id in kernel_ids
        for group in perturbation_sets
    )
