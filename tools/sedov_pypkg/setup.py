import codecs
import os.path
from setuptools import setup

here = os.path.abspath(os.path.dirname(__file__))

def readme():
    fname = os.path.join(here, 'README.md')
    with codecs.open(fname, encoding='utf8') as fptr:
        return fptr.read()

def version():
    fname = os.path.join(here, 'VERSION')
    with open(fname, 'r') as fptr:
        return fptr.read().strip()

reqs_list = ['numpy', 'pandas', 'matplotlib', 'yt']

pkg_dict = {'sedov': ['PkgData/*', 'TestData/*']}

setup(
    name='Sedov',
    version=version(),
    author='Jared O\'Neal',
    author_email='joneal@anl.gov',
    maintainer='Jared O\'Neal',
    maintainer_email='joneal@anl.gov', 
    packages=['sedov'],
    package_data=pkg_dict, 
    url='http://git.cels.anl.gov/OrchestrationRuntime',
    license='TBD',
    description='Code for working with Sedov results',
    long_description=readme(),
    setup_requires=['nose>=1.0'],
    install_requires=reqs_list,
    # tests_require=['nose>=1.0'],
    # test_suite='sedov.test_suite',
    keywords='sedov',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Programming Language :: Python :: 2.7', 
        'Intended Audience :: Science/Research', 
        'Natural Language :: English', 
        'Operating System :: MacOS :: MacOS X', 
        'Operating System :: POSIX', 
        'Topic :: Scientific/Engineering']
)
