import codecs

from pathlib import Path
from setuptools import (find_packages,
                        setup)

__PKG_ROOT = Path(__file__).resolve().parent

def readme_md():
    fname = __PKG_ROOT.joinpath("README.md")
    with codecs.open(fname, encoding="utf8") as fptr:
        return fptr.read()

def version():
    fname = __PKG_ROOT.joinpath("VERSION")
    with open(fname, "r") as fptr:
        return fptr.read().strip()

# wheel must be installed before installing this package so that the LICENSE
# file is actually installed as part of the source distribution when it is
# installed with pip.  This implies that the LICENSE is in the sdist tarball for
# those who access the code outside of pip.
python_requires = ">=3"
install_requires = ["wheel", "nose", "Code-Generation-Toolkit"]
# TODO: Integrate testing with pytests & tox?
# test_requires = [???]

packages = find_packages(include=["milhoja"])
package_data = {"milhoja": ["TestData/*"]}

project_urls = {"Source": "Git Hub",
                "Documentation": "milhoja.org",
                "Tracker": "Git Hub Issues"}

setup(
    name="milhoja",
    version=version(),
    author="Tom Klosterman, Wesley Kwiecinski, Jared O'Neal",
    author_email="joneal@anl.gov",
    maintainer="Jared O'Neal",
    maintainer_email="joneal@anl.gov",
    packages=packages,
    package_data=package_data, 
    url="http://milhoja.org",
    project_urls=project_urls,
    license="???",
    description="Milhoja AMR-specific parallelization for heterogenous platforms",
    long_description=readme_md(),
    long_description_content_type="text/markdown",
    python_requires=python_requires,
    install_requires=install_requires,
    test_suite="milhoja.test_suite",
    keywords="Milhoja",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
#        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",  
        "Intended Audience :: Science/Research", 
        "Natural Language :: English", 
        "Operating System :: MacOS :: MacOS X", 
        "Topic :: Scientific/Engineering"]
)

