Getting Started
===============

General Installations
---------------------
There is presently only one option for installing this package for general use.
For developer installations, refer to the
:numref:`developers_guide:Developer Environment`.


Manual source distribution installation via `setuptools`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. _setuptools: https://setuptools.pypa.io/en/latest/index.html

To install the package from a source tarball, activate your target python and
execute

* ``which python``
* ``which pip``
* ``cd /path/to/milhoja_pypkg``
* Test manually with ``tox`` or python if so desired (see below)
* Install source distribution of the package by executing from the current
  working directory

.. code-block:: console

    $ python setup.py sdist
    $ python install --upgrade dist/milhoja-<version>.tar.gz
    $ pip list

where ``<version>`` is the version of the package to be installed.  This can be
determined by scanning the stdout generated when creating the ``sdist``.  Note
that the ``--user`` pip install flag might be useful if installing in a
community system in which the python/pip are write-protected.

Installation Testing
--------------------
Installations can be tested by running 

.. code-block:: console

    $ python
    >>> import milhoja 
    >>> milhoja.__version__
    '1.0.1'
    >>> milhoja.test()
        ...
    $ python -m pydoc milhoja
