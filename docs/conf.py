# docs/conf.py
project = "spatialvi-tools"
author = "Ori Kronfeld"
release = "0.1.0"
extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
]
html_theme = "sphinx_book_theme"
html_title = "spatialvi-tools"
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "anndata": ("https://anndata.readthedocs.io/en/stable/", None),
    "scvi": ("https://docs.scvi-tools.org/en/stable/", None),
}
