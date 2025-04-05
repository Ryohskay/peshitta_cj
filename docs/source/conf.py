"""Sphinx config file."""
import sys
from pathlib import Path

# ensure that the source code is loadable by sphinx
sys.path.insert(0, str(Path("../../src").resolve()))

project = "Peshitta CJ"
author = "Ryosuke Nagata"
copyright = "2025, Ryosuke Nagata"
version = "0.0"
extensions = [
        "sphinx.ext.autodoc",
        "sphinx.ext.autosummary",
        # "sphinx_autodoc_typehints",
        ]
nitpicky = True
