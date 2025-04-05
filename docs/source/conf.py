"""Sphinx config file."""
import sys
from pathlib import Path

# ensure that the source code is loadable by sphinx
sys.path.insert(0, str(Path("../../src").resolve()))

project = "Peshitta CJ"
author = "Ryosuke Nagata"
copyright = "2025, Ryosuke Nagata"
version = "0.0.0"

html_theme = "sphinx_rtd_theme"

extensions = [
        "sphinx.ext.autodoc",
        "sphinx.ext.autosummary",
        "sphinx.ext.napoleon",
        ]

intersphinx_mapping = {
        "python": ("https://docs.python.org/3.12", None),
        "numpy": ("https://docs.scipy.org/doc/numpy/", None),
        "pandas": ("https://pandas.pydata.org/docs/", None),
    }


nitpicky = True
nitpick_ignore = {
    ("py:func", "int"),
    ("envvar", "LD_LIBRARY_PATH"),
}
