"""
A general-use python package for working with Milhoja.  This packages includes
code needed by Milhoja code generators.
"""

import unittest
import nose.core

from importlib.metadata import version

# thread team configuration topologies
# TODO: Add one for each configuration.  This will be needed so that
#       CG-Kit in Flash-X can map a particular action/block loop onto the
#       matching TT configuration.

# functions

# classes
from .CodeGenerationLogger import CodeGenerationLogger
from .TileWrapperGenerator import TileWrapperGenerator
from .CppTaskFunctionGenerator import CppTaskFunctionGenerator

# unittests
from .TestTileWrapperGenerator import suite as suite1
from .TestCppTaskFunctionGenerator import suite as suite2

# Follow typical version-access interface used by other packages
# (e.g., numpy, scipy, pandas, matplotlib)
__version__ = version("milhoja")

def test_suite():
    return unittest.TestSuite([suite1(), suite2()])

def test(verbosity=1):
    if verbosity not in [0, 1, 2]:
        msg = f"Invalid verbosity level - {verbosity} not in {{0,1,2}}"
        raise ValueError(msg)

    nose.core.TextTestRunner(verbosity=verbosity).run(test_suite())

