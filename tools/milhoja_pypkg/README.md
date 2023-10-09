A general-use python package for working with Milhoja.  This packages includes
code needed by Milhoja code generators.

## Known Issues
* We acknowledge that `pytest` is a commonly-used and well-liked framework for
  testing python code.  We intend to study if it should be used to integrate
  testing in the package or if the present use of python's `unittest`
  facitilities is sufficient.

## Installation Instructions
There is presently only one option for installing this package for use.  For
installation for development, please see below.

#### Manual source distribution installation via [setuptools](https://setuptools.pypa.io/en/latest/index.html)
To install the package from a source tarball, activate your target python and
execute
* `which python`
* `which pip`
* `cd /path/to/milhoja_pypkg`
* Test manually with `tox` or python if so desired (see below)
* Install source distribution of the package by executing from the current working directory
```
$ python setup.py sdist
$ python install --upgrade dist/milhoja-<version>.tar.gz
$ pip list
```
where `<version>` is the version of the package to be installed.  This can be
determined by scanning the stdout generated when creating the `sdist`.  Note
that the `--user` pip install flag might be useful if installing in a community
system in which the python/pip are write-protected.

#### Installation Testing
Installations can be tested by running 
```
$ python
>>> import milhoja
>>> milhoja.__version__
'1.0.1'
>>> milhoja.test()
    ...
$ python -m pydoc milhoja
```

## Development with `tox`
The coverage reports for this package are managed with
[tox](https://tox.wiki/en/latest/index.html), which can be used for CI work
among other work.  However, the same `tox` setups can be used by developers if
so desired.  This can be useful since `tox` will automatically setup and manage
dedicated virtual environments for the developer.  The following guide can be
used to setup `tox` as a command line tool on an individual platform in a
dedicated, minimal virtual environment and is based on the a
[webinar](https://www.youtube.com/watch?v=PrAyvH-tm8E) by Oliver Bestwalter.  I
appreciate his solution as there is no need to activate the virtual environment
in order to use `tox`.

Developers that would like to use `tox` should learn about the tool so that, at
the very least, they understand the difference between running `tox` and `tox
-r`.

To create a python virtual environment based on a desired python dedicated to
hosting `tox`, execute some variation of
```
$ cd
$ deactivate (to deactivate the current virtual environment if you are in one)
$ /path/to/desired/python --version
$ /path/to/desired/python -m venv $HOME/.toxbase
$ ./.toxbase/bin/pip list
$ ./.toxbase/bin/python -m pip install --upgrade pip
$ ./.toxbase/bin/pip install --upgrade setuptools
$ ./.toxbase/bin/pip install tox
$ ./.toxbase/bin/tox --version
$ ./.toxbase/bin/pip list
```

To avoid the need to activate `.toxbase`, we setup `tox` in `PATH` for use
across all development environments that we might have on our system. In the
following, please replace `.bash_profile` with the appropriate shell
configuration file and tailor to your needs.
```
$ mkdir $HOME/local/bin
$ ln -s $HOME/.toxbase/bin/tox $HOME/local/bin/tox
$ vi $HOME/.bash_profile (add $HOME/local/bin to PATH)
$ . $HOME/.bash_profile
$ which tox
$ tox --version
```

Since `cgkit` is a dependency of this package but is not available on pypi,
`tox` must include it into each environment from a local clone.  To accomplish
this, users must set the environment variable `CGKIT_PATH` to the root of the
desired local clone.

If the environment variable `COVERAGE_FILE` is set, then this is the coverage
file that will be used with all associated work.  If it is not specified, then
the coverage file is `.coverage_milhoja`.

No work will be carried out by default with the calls `tox` and `tox -r`.

The following commands can be run from the directory that contains this file.
* `tox -r -e coverage`
  * Execute the full test suite for the package and save coverage results to the coverage file
  * The test runs the package code in the local clone rather than code installed into python so that coverage results accessed through web services such as Coveralls are clean and straightforward
* `tox -r -e nocoverage`
  * Execute the full test suite for the package using the code installed into python
* `tox -r -e report`
  * It is intended that this be run after or with coverage
  * Display a report and generate an HTML report for the package's full test suite
* `tox -r -e check`
  * This is likely only useful for developers working on a local clone
  * Run several checks on the code to report possible issues
  * No files are altered automatically by this task
* `tox -r -e format`
  * __IMPORTANT__: This task will potentially alter files without warnings or prompts.
  * This is likely only useful for developers working on a local clone
  * Apply `black` to all files in the package for cleaning/standardization

Additionally, you can run any combination of the above such as `tox -r -e report,coverage`.

## Manual Developer Testing
It is possible to test manually outside of `tox`, which could be useful for
testing at the level of a single test.

The following example shows how to run only a single test case using the
`coverage` virtual environment setup by `tox`
```
$ cd /path/to/milhoja_pypkg
$ tox -r -e coverage
$ . ./.tox/coverage/bin/activate
$ which python
$ python --version
$ pip list
$ python -m unittest milhoja.tests.TestTileWrapperGenerator
```
