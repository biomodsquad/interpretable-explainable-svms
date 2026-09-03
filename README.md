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
from mistic import cvSet, kernelWrapper, paramSet, score_svc, svmSet
```

For development, install the repository and its development tools in editable
mode:

```bash
python -m pip install -e ".[dev]"
```

The package requires Python 3.10 or newer. Example classification and
regression workflows are available in `mistic/examples`.

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
