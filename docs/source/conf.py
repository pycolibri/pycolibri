# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

project = 'colibri'
copyright = '2023, hdsp'
author = 'hdsp'
version  = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "autodocsumm",
    'sphinx_gallery.gen_gallery'
]

exec_code_working_dir = "../.."

templates_path = ['_templates']
exclude_patterns = []

autodoc_inherit_docstrings = False
napoleon_google_docstring = True
napoleon_include_init_with_doc = True
napoleon_numpy_docstring = False

html_title = "Colibri"
html_logo = "logo.png"

autodoc_mock_imports = [
    "torch",
    "tqdm",
    "numpy",
    "timm",
    "cv2",
    "PIL",
    "pretrainedmodels",
    "torchvision",
    "efficientnet-pytorch",
    "segmentation_models_pytorch.encoders",
    "segmentation_models_pytorch.utils",
    "h5py"
    # 'segmentation_models_pytorch.base',
]

autoclass_content = "both"
autodoc_typehints = "description"
# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']


sphinx_gallery_conf = {
     'examples_dirs': '../../examples',   # path to your example scripts
     'gallery_dirs': 'auto_examples',  # path to where to save gallery generated output
     "filename_pattern": "/demo_",
     "ignore_pattern": r"__init__\.py",
}


class FakeSignature:
    def __getattribute__(self, *args):
        raise ValueError


def f(app, obj, bound_method):
    if "__new__" in obj.__name__:
        obj.__signature__ = FakeSignature()


def setup(app):
    app.connect("autodoc-before-process-signature", f)

autodoc_member_order = "bysource"




