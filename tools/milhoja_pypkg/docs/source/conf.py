# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import json

from milhoja import __version__

project = 'milhoja'
copyright = "Figure this out before making public!"
author = "Jared O'Neal and Wesley Kwiecinski"
release = __version__

latex_packages = [
    'xspace',
    'mathtools',
    'amsfonts', 'amsmath', 'amssymb'
]
latex_macro_files = ['base']

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.autosummary',
              'sphinx.ext.autosectionlabel',
              'sphinx.ext.todo',
              'sphinxcontrib.bibtex']
autosectionlabel_prefix_document = True

templates_path = ['_templates']
exclude_patterns = []

todo_include_todos = True

numfig = True
math_numfig = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

extensions += ['sphinx.ext.mathjax']

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_css_files = ['css/custom.css']

# Configure MathJax with LaTeX macros
mathjax3_config = {
    'loader': {},
    'tex': {
        'macros': {}
    }
}
for each in latex_macro_files:
    with open(f"latex_macros_{each}.json", "r") as fptr:
        macro_configs = json.load(fptr)
    for cmd_type, macros_all in macro_configs.items():
        for command, value in macros_all.items():
            assert command not in mathjax3_config['tex']['macros']
            mathjax3_config['tex']['macros'][command] = value

# -- LaTeX configuration -----------------------------------------------------
# Some of this configuration is from
# https://stackoverflow.com/questions/9728292/creating-latex-math-macros-within-sphinx

# Load packages
latex_elements = {
    "papersize": "letterpaper",
    "preamble": ""
}
for package in latex_packages:
    latex_elements['preamble'] += f'\\usepackage{{{package}}}\n'

# Use bibtex for bibliography
bibtex_bibfiles = ['milhoja.bib']

# Configure LaTeX with macros
for each in latex_macro_files:
    with open(f"latex_macros_{each}.json", "r") as fptr:
        macro_configs = json.load(fptr)
    for cmd_type, macros_all in macro_configs.items():
        for command, value in macros_all.items():
            if isinstance(value, str):
                macro = rf"\{cmd_type}{{\{command}}} {{{value}}}"
            elif len(value) == 2:
                value, n_args = value
                macro = rf"\{cmd_type}{{\{command}}}[{n_args}] {{{value}}}"
            else:
                raise NotImplementedError("No use case yet")
            latex_elements['preamble'] += (macro + "\n")
