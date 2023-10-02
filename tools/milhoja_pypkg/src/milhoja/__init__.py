"""
Please refer to package's README file.
"""

from importlib.metadata import version

# Follow typical version-access interface used by other packages
# (e.g., numpy, scipy, pandas, matplotlib)
__version__ = version("milhoja")

# thread team configuration topologies
# TODO: Add one for each configuration.  This will be needed so that
#       CG-Kit in Flash-X can map a particular action/block loop onto the
#       matching TT configuration.

# constants
from .constants import MILHOJA_JSON_FORMAT
from .constants import CURRENT_MILHOJA_JSON_VERSION
from .constants import TASK_FUNCTION_FORMATS
from .constants import LOG_LEVEL_NONE
from .constants import LOG_LEVEL_BASIC
from .constants import LOG_LEVEL_BASIC_DEBUG
from .constants import LOG_LEVEL_MAX
from .constants import LOG_LEVELS

# Functions used by classes
from .generate_tile_metadata_extraction import generate_tile_metadata_extraction

# classes
from .TaskFunction import TaskFunction
from .CodeGenerationLogger import CodeGenerationLogger
from .AbcCodeGenerator import AbcCodeGenerator
from .TileWrapperGenerator_cpp import TileWrapperGenerator_cpp
from .TaskFunctionGenerator_cpu_cpp import TaskFunctionGenerator_cpu_cpp

# Functions that use classes
from .generate_data_item import generate_data_item
from .generate_task_function import generate_task_function

# ----- Python unittest-based test framework
# Used for automatic test discovery
from .load_tests import load_tests

# Allow users to run full test suite as milhoja.test()
from .test import test
