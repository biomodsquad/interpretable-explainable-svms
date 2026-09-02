"""Sphinx configuration for the MISTIC documentation."""

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10
    import tomli as tomllib

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "MISTIC"
copyright = "2025–2026, BioModSquad, Chris A. Kieslich"
author = "BioModSquad"

# Read the distribution version from its packaging metadata so the rendered
# documentation cannot drift from the wheel and source distribution.
repository_root = Path(__file__).resolve().parents[1]
with (repository_root / "pyproject.toml").open("rb") as project_file:
    release = tomllib.load(project_file)["project"]["version"]
version = release

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "alabaster"
html_static_path = []
