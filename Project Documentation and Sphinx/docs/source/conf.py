# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('../..'))


# -- Project information -----------------------------------------------------

project = 'Dedoublement'
copyright = '2022, Erwan BARGAIN, Salif SAWADOGO'
author = 'Erwan BARGAIN, Salif SAWADOGO'
version = '1.0'
release = 'PoC'


# -- General configuration ---------------------------------------------------





# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

extensions = [
    "sphinx.ext.napoleon",
    "autodocsumm",
        
     "sphinx.ext.viewcode",

    "nbsphinx",
    "myst_parser",
    "sphinx.ext.coverage",
    'sphinx.ext.graphviz',
    'sphinx.ext.inheritance_diagram'
]

autodoc_default_options = {"autosummary": True}
#autodoc_member_order = "bysource"



# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.


# -- Options for HTML output -------------------------------------------------

html_theme = "classic"
html_theme_options = {"rightsidebar": "true", "relbarbgcolor": "black"}








# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'Dedoublement', 'Dedoublement Documentation',
     author, 'Dedoublement', 'One line description of project.',
     'Miscellaneous'),
]



html_favicon = '../assets/documentation-icon.svg'
html_logo = '../assets/cdandlp_logo.png'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


def setup(app):
    app.add_css_file('custom.css')