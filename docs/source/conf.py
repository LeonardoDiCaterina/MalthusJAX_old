# docs/source/conf.py

import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))

html_theme = 'sphinx_rtd_theme'

# docs/source/conf.py

extensions = [
    'sphinx.ext.autodoc',      # Pulls documentation from docstrings
    'sphinx.ext.napoleon',     # Understands Google-style docstrings
    'sphinx.ext.viewcode',     # Adds links to your source code
    'sphinx_autodoc_typehints', # Uses your type hints in the docs
    'myst_parser',           # Allows you to write .md files
]

# (Optional but recommended) Tell myst_parser to parse docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True