# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.


import os
import sys
import parallel_bilby

sys.path.insert(0, os.path.abspath("../parallel_bilby/"))
sys.path.insert(0, os.path.abspath("../examples/"))

# -- Project information -----------------------------------------------------

project = "Parallel Bilby"
copyright = "2020, Greg Ashton, Rory Smith"
author = "Greg Ashton, Rory Smith"

# The full version, including alpha/beta/rc tags
release = parallel_bilby.__version__

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "numpydoc",
    "sphinx.ext.graphviz",
    "nbsphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.viewcode",
    "sphinxarg.ext",
    "sphinx_tabs.tabs",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
gitlab_url = "https://git.ligo.org/lscsoft/parallel_bilby"
html_logo = "_static/logo.png"
html_favicon = "_static/favicon.ico"

html_theme_options = {
    'logo_only': True,
    'display_version': True,
    'vcs_pageview_mode': 'display_gitlab',
    'style_nav_header_background': '#343131'
}