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

templates_path = ["_templates"]
exclude_patterns = []

autodoc_inherit_docstrings = False
napoleon_google_docstring = True
napoleon_numpy_docstring = False

html_title = "Colibri"
html_logo = os.path.join("figures", "logo.png")

autodoc_default_options = {
    "exclude-members": "__init__"
}

# autodoc_mock_imports = [
#     "torch",
#     "tqdm",
#     "numpy",
#     "timm",
#     "cv2",
#     "PIL",
#     "torchvision",
#     "h5py"
# ]


# how to define macros: https://docs.mathjax.org/en/latest/input/tex/macros.html
mathjax3_config = {
    "tex": {
        "equationNumbers": {"autoNumber": "AMS", "useLabelIds": True},
        "macros": {
            "forwardLinear": r"\mathbf{H}",
            "learnedOptics": r"\mathbf{\Phi}",
            "noise":  r"\epsilon",
            "xset": r"\mathcal{X}",
            "thetaset": r"\mathcal{\Omega}",
            "yset": r"\mathcal{Y}",
            'argmin': '\\operatorname{arg\\,min}',
            'coloneqq': '\\mathrel{:=}',
        },
    }
}

autoclass_content = "both"
autodoc_typehints = "description"
# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = []


# sphinx_gallery_conf = {
#      'examples_dirs': '../../examples',   # path to your example scripts
#      'gallery_dirs': 'auto_examples',  # path to where to save gallery generated output
#      "filename_pattern": "/demo_",
#      "ignore_pattern": r"__init__.py",
# }

sphinx_gallery_conf = {
    "examples_dirs": ["../../examples/"],
    "gallery_dirs": "auto_examples",  # path to where to save gallery generated output
    "filename_pattern": "/demo_",
    "run_stale_examples": True,
    "ignore_pattern": r"__init__\.py",
    "reference_url": {
        # The module you locally document uses None
        "sphinx_gallery": None
    },
    # directory where function/class granular galleries are stored
    "backreferences_dir": "gen_modules/backreferences",
    # Modules for which function/class level galleries are created. In
    # this case sphinx_gallery and numpy in a tuple of strings.
    "doc_module": ("colibri_hdsp"),
    # objects to exclude from implicit backreferences. The default option
    # is an empty set, i.e. exclude nothing.
    "exclude_implicit_doc": {},
    "nested_sections": False,
}


autodoc_member_order = "bysource"




