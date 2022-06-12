"""Sphinx configuration."""
project = "System Identification"
author = "Lucsa Vieira dos Santos"
copyright = "2022, Lucsa Vieira dos Santos"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_click",
    "myst_parser",
]
autodoc_typehints = "description"
html_theme = "furo"
