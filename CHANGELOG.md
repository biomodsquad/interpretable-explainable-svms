# Changelog

## 0.2.0b1 (beta)

MISTIC 0.2 introduces a common mixed-kernel and feature-selection model. This
beta is published for evaluation before the stable 0.2 release.

### Added

- Composable `MixedKernel` expressions supporting sums, products, nonnegative
  weights, and positive integer powers.
- Named and tunable base-kernel parameters and mixture weights.
- Tanimoto kernels for nonnegative count and binary fingerprint features.
- Kernel-specific candidate feature pools and independently selectable
  `(kernel, perturbation set)` units.
- Shared or per-cross-validation-model selection states, with any-appearance
  and majority consensus policies for the final model.
- Explicit initial feature medoids for forward selection.
- Mixed-kernel analytical gradients and binary feature-flip attribution paths.
- A curated mixed-kernel breast-cancer notebook and mixed-kernel tutorial.

### Changed

- `greedy_forward_selection` and `greedy_backward_selection` now use the same
  kernel-aware feature-selection abstraction for single and mixed kernels.
- A conventional kernel is represented internally as a one-leaf
  `MixedKernel` with weight one.
- Hyperparameters are shared across CV members and selected against aggregate
  cross-validation performance.
- The final model can unify member-specific feature states using either any
  appearance or majority appearance.
- Experimental notebooks and scripts have moved from `mistic/examples` to the
  top-level `experimentation` directory. Curated examples remain packaged.

### Compatibility

- Bare `kernelWrapper` instances are still accepted and converted to a
  one-leaf mixed kernel.
- `MixedkernelWrapper`, `greedy_forward_kernel_selection`,
  `separate_feature_sets`, and `separate_parameters` remain available as
  transition interfaces and emit or document deprecation behavior.

See the [0.2 migration guide](docs/migration_0_2.rst) for code examples and
behavioral details.

## 0.1.1

Previous public release.
