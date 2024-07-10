=========================
How to Build Milhoja Docs
=========================

In order to build the Milhoja documentation, you will need to install the
`sphinx`_ pypackage, as well as the sphinx extension `sphinxcontrib.bibtex`.
This can be done by running the command

`pip install Sphinx sphinxcontrib.bibtex`

Once this is done, head to the directory

`tools/milhoja_pypkg/docs/source`

inside of your Milhoja repository clone and run the following command

`sphinx-build -M html ./ [build_directory]`

where [build_directory] is the directory where the html should be saved.

.. _sphinx: https://www.sphinx-doc.org/en/master/usage/installation.html