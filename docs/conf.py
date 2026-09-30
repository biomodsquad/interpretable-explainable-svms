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
is_prerelease = any(marker in release for marker in ("a", "b", "rc", "dev"))

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_title = f"MISTIC {release}"
documentation_root = "https://biomodsquad.org/interpretable-explainable-svms/"
html_baseurl = documentation_root + ("beta/" if is_prerelease else "")
html_logo = "_static/mistic-logo.jpg"
html_favicon = "_static/mistic-icon.png"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "source_repository": "https://github.com/biomodsquad/interpretable-explainable-svms/",
    "source_branch": "0.2" if is_prerelease else "main",
    "source_directory": "docs/",
    "footer_icons": [
        {
            "name": "MISTIC on GitHub",
            "url": "https://github.com/biomodsquad/interpretable-explainable-svms",
            "html": "<span>GitHub repository</span>",
            "class": "",
        }
    ],
    "light_css_variables": {
        "color-brand-primary": "#146b67",
        "color-brand-content": "#146b67",
        "color-admonition-background": "#edf8f6",
    },
    "dark_css_variables": {
        "color-brand-primary": "#62d4c9",
        "color-brand-content": "#62d4c9",
    },
}
if is_prerelease:
    html_theme_options["announcement"] = (
        "You are reading the MISTIC 0.2 beta documentation. "
        f'<a href="{documentation_root}">View the stable 0.1 documentation.</a>'
    )
