This python package contains code for working with results generated by running
the Sedov test problem.

The list of python package dependencies for the Sedov package are specified in
the `reqs_list` variable in the file `setup.py`.

This package can be installed for use on a machine in at least one of two ways.
1. Install the package directly in python
   * Run the command `python setup.py test` and confirm that all tests pass.
   * Create latest package distribution by running the command `python setup.py sdist`
   * The package version built should appear near the end of the distribution build output (Ex. 1.0.0b2)
   * Install package in python by running the command
```
pip install --upgrade dist/sedov-<version>.tar.gz
```
2. Install package locally for use with python (for instance, on a machine
   for which the python installation cannot be altered)
   * Create a folder for storing local python packages (Ex. `~/local`)
   * Set `PYTHONPATH` so that python can find your local installation (Ex. `~/local/lib/<my python>/site-packages`)
   * Run the command `python setup.py test` and confirm that all tests pass.
   * Create latest package distribution by running the command `python setup.py sdist`
   * The package version built should appear near the end of the distribution build output (Ex. 1.0.0b2)
   * Install package locally by running the command
```
pip install --upgrade --install-option="--prefix=~/local" dist/sedov-<version>.tar.gz
```

Installation can be tested by running
```
$ python
>>> import sedov
>>> sedov.__version__
'1.0.0b2'
>>> sedov.test()
```
