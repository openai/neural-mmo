# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

from unittest.mock import MagicMock

class Mock(MagicMock):
    @classmethod
    def __getattr__(cls, name):
        return MagicMock()

MOCK_MODULES = ['ray', 'ray.rllib', 'ray.rllib.utils', 'ray.rllib.utils.spaces', 'ray.rllib.utils.spaces.repeated', 'ray.rllib.utils.spaces.flexdict', 'torch', 'torch.autograd', 'torch.nn', 'torch.distributions', 'torch.nn.parameter']
autodoc_mock_imports = ["torch"]
sys.modules.update((mod_name, Mock()) for mod_name in MOCK_MODULES)


# -- Project information -----------------------------------------------------

project = 'Neural MMO v1.5'
copyright = '2020, Joseph Suarez'
author = 'Joseph Suarez'

# The full version, including alpha/beta/rc tags
release = '1.5'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
      'sphinx.ext.autodoc', 
      'sphinx.ext.coverage', 
      'sphinx.ext.napoleon',
   ]

#Don't sort method names
autodoc_member_order = 'bysource'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
#html_theme = 'neural-mmo'
html_theme = 'sphinx_rtd_theme'
html_theme_options = {
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

#Readable code theme
pygments_style = 'monokai'
