import codecs

from pathlib import Path
from setuptools import setup

__PKG_ROOT = Path(__file__).resolve().parent


def readme_md():
    fname = __PKG_ROOT.joinpath("README.md")
    with codecs.open(fname, encoding="utf8") as fptr:
        return fptr.read()


def version():
    fname = __PKG_ROOT.joinpath("VERSION")
    with open(fname, "r") as fptr:
        return fptr.read().strip()


description = "Milhoja AMR-specific parallelization for heterogenous platforms"

# Changes made to python_requires should be propagated to all tox.ini and all
# CI build server config files.
# Latest version of cg-kit needs python 3.8
python_requires = ">=3.10"
# Assuming cg-kit 0.2.2 for now.
code_requires = ["Code-Generation-Toolkit"]
test_requires = ["numpy"]
install_requires = code_requires + test_requires

package_data = {"milhoja": ["tests/data/*",
                            "tests/data/fake/*",
                            "tests/data/runtime/*",
                            "tests/data/Sedov/*",
                            "tests/data/FlashX/*",
                            "tests/data/Spark/*",
                            "templates/*"]}

project_urls = {
    "Source": "Git Hub",
    "Documentation": "milhoja.org",
    "Tracker": "Git Hub Issues"
}

setup(
    name="milhoja",
    version=version(),
    author="Tom Klosterman, Wesley Kwiecinski, Jared O'Neal",
    author_email="joneal@anl.gov",
    maintainer="Jared O'Neal",
    maintainer_email="joneal@anl.gov",
    package_dir={"": "src"},
    package_data=package_data,
    url="http://milhoja.org",
    project_urls=project_urls,
    license="???",
    description=description,
    long_description=readme_md(),
    long_description_content_type="text/markdown",
    python_requires=python_requires,
    install_requires=install_requires,
    keywords="Milhoja",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Programming Language :: Python :: 3",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Operating System :: MacOS :: MacOS X",
        "Topic :: Scientific/Engineering"
    ]
)
