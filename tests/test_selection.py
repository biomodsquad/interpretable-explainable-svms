import numpy as np
import pytest

from mistic.selection import (
    KernelFeatureSelectionState,
    KernelFeatureSet,
    build_kernel_feature_selection_space,
)


def test_kernel_feature_set_normalizes_singletons_and_validates_indices():
    assert KernelFeatureSet("linear", [2]).features == (2,)
    with pytest.raises(ValueError):
        KernelFeatureSet("linear", [])
    with pytest.raises(ValueError):
        KernelFeatureSet("linear", [1, 1])
    with pytest.raises(ValueError):
        KernelFeatureSet("", [1])


def test_selection_state_derives_kernel_and_original_feature_views():
    space = (
        KernelFeatureSet("linear", (0,)),
        KernelFeatureSet("linear", (2, 3)),
        KernelFeatureSet("radial", (0,)),
        KernelFeatureSet("radial", (4,)),
    )
    state = KernelFeatureSelectionState(space, frozenset({1, 2, 3}))

    np.testing.assert_array_equal(state.kernel_features["linear"], [2, 3])
    np.testing.assert_array_equal(state.kernel_features["radial"], [0, 4])
    np.testing.assert_array_equal(state.features, [0, 2, 3, 4])
    assert state.num_kernel_feature_sets == 3
    assert state.num_kernel_features == 4


def test_selection_state_updates_are_immutable():
    space = build_kernel_feature_selection_space(("linear", "radial"), ((0,), (1,)))
    empty = KernelFeatureSelectionState.none_active(space)
    selected = empty.activate([0, 3])

    assert not empty.active
    assert selected.active == frozenset({0, 3})
    assert selected.deactivate([0]).active == frozenset({3})
    assert selected.only([1]).active == frozenset({1})
    with pytest.raises(IndexError):
        empty.activate([len(space)])


def test_same_feature_is_an_independent_unit_for_each_kernel():
    space = build_kernel_feature_selection_space(("linear", "radial"), ((5,),))
    state = KernelFeatureSelectionState.all_active(space)

    assert state.num_kernel_feature_sets == 2
    assert state.num_kernel_features == 2
    np.testing.assert_array_equal(state.features, [5])
