<p align="center">
  <img
    src="docs/_static/mistic-logo.jpg"
    alt="MISTIC logo: hands interpreting a selected feature"
    width="220"
  >
</p>

# MISTIC

**Model Informed Feature Selection Through Importance and Contribution**

MISTIC provides feature selection, boundary counterfactuals, and attribution
for interpretable and explainable support vector machines.

## Installation

Install the release from PyPI:

```bash
python -m pip install mistic-svm
```

The PyPI distribution is named `mistic-svm` because `mistic` is already used
by an unrelated project. The Python import remains concise:

```python
from mistic import MixedKernel, cvSet, kernelWrapper, paramSet, score_svc, svmSet
```

For development, install the repository and its development tools in editable
mode:

```bash
python -m pip install -e ".[dev]"
```

The package requires Python 3.10 or newer. Example classification and
regression workflows are available in `mistic/examples`.

### MISTIC 0.2 beta

The 0.1 series remains the default stable release. Install the opt-in 0.2 beta
with an exact version:

```bash
python -m pip install "mistic-svm==0.2.0b1"
```

The beta source and documentation are maintained on the `0.2` branch. Beta
releases may introduce compatibility changes before 0.2 becomes stable; see
the migration guide and changelog before upgrading.

Latin-hypercube parameter designs are available without additional
dependencies:

```python
from mistic import loguniform, parameterSpace

grid = parameterSpace(
    model={"C": loguniform(1e-3, 1e3)},
    kernel={"gamma": loguniform(1e-6, 1e1)},
).sample(n_trials=24, random_state=7)
```

All kernels use the composable `MixedKernel` API. A single kernel is a
one-leaf mixture, while named leaves can be added, multiplied, powered, given
tunable nonnegative weights, and assigned independent feature sets:

```python
kernel = MixedKernel.weighted_sum(
    [
        kernelWrapper("linear", name="linear"),
        kernelWrapper("rbf", name="radial"),
    ],
    weight_parameters=["linear_weight", "radial_weight"],
)
```

See the mixed-kernel tutorial in the documentation for parameter naming,
kernel-specific candidate pools, and unified forward/backward selection.

Synthetic-data validation studies and their generated results are kept in
`validation` so they remain separate from the user-facing examples.

## Documentation and tests

Read the complete installation guide, framework overview, tutorials, examples,
and API reference at
[biomodsquad.org/interpretable-explainable-svms](https://biomodsquad.org/interpretable-explainable-svms/).

Build the documentation locally with
`sphinx-build -W docs docs/_build/html` and run the test suite with `pytest`.

## Repository

The canonical source repository is
[biomodsquad/interpretable-explainable-svms](https://github.com/biomodsquad/interpretable-explainable-svms).
