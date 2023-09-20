"""
A general-use python package for working with Milhoja.  This packages includes
code needed by Milhoja code generators.
"""

from importlib.metadata import version

# Follow typical version-access interface used by other packages
# (e.g., numpy, scipy, pandas, matplotlib)
__version__ = version("milhoja")

# thread team configuration topologies
# TODO: Add one for each configuration.  This will be needed so that
#       CG-Kit in Flash-X can map a particular action/block loop onto the
#       matching TT configuration.

# functions

# classes
from .CodeGenerationLogger import CodeGenerationLogger
from .TileWrapperGenerator import TileWrapperGenerator
from .CppTaskFunctionGenerator import CppTaskFunctionGenerator

# ----- Python unittest-based test framework
# Used for automatic test discovery
from .load_tests import load_tests

# Allow users to run full test suite as milhoja.test()
from .test import test
