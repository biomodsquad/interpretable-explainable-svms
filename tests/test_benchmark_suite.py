"""Tests for synthetic benchmark construction."""

import numpy as np

from mistic.examples.classification_benchmark_suite import DatasetSpec, make_dataset


def test_benchmark_signal_columns_are_permuted_and_mask_stays_aligned():
    spec = DatasetSpec(
        name="correlated_test", n_samples=50, n_features=100,
        scenario="correlated", data_seed=2026, n_signal=20,
    )
    X, _, signal_mask = make_dataset(spec)

    signal_positions = np.flatnonzero(signal_mask)
    assert not np.array_equal(signal_positions, np.arange(spec.n_signal))
    original_indices = np.asarray([int(name[1:]) for name in X.columns])
    np.testing.assert_array_equal(signal_mask, original_indices < spec.n_signal)


def test_benchmark_column_permutation_is_reproducible():
    spec = DatasetSpec(
        name="nonlinear_test", n_samples=40, n_features=60,
        scenario="nonlinear", data_seed=11, n_signal=20,
    )
    first_X, first_y, first_mask = make_dataset(spec)
    second_X, second_y, second_mask = make_dataset(spec)

    np.testing.assert_array_equal(first_X, second_X)
    np.testing.assert_array_equal(first_y, second_y)
    np.testing.assert_array_equal(first_mask, second_mask)
    assert first_X.columns.tolist() == second_X.columns.tolist()
