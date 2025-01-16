import os
import sys
sys.path.insert(0, os.path.abspath('../../src/'))


# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'TOPS'
copyright = '2025, Hallvar Haugdal'
author = 'Hallvar Haugdal'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['autoapi.extension', 'sphinx_mdinclude', 'sphinx.ext.napoleon',]

templates_path = ['_templates']
exclude_patterns = []

autoapi_dirs = ['../../src']
autoapi_add_toctree_entry = True



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']