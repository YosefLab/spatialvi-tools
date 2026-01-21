"""Sphinx configuration for spatialvi-tools documentation."""

import os
import sys
from datetime import datetime

# Add the package to sys.path for autodoc
sys.path.insert(0, os.path.abspath("../src"))

# -- General configuration ------------------------------------------------

project = "spatialvi-tools"
author = "YosefLab"
copyright = f"{datetime.now().year}, {author}"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
]

autosummary_generate = True

templates_path = ["_templates"]
exclude_patterns = []

master_doc = "index"

# -- Options for HTML output --------------------------------------------

html_theme = "alabaster"
html_static_path = ["_static"]