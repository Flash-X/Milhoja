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
from .constants import EXTERNAL_ARGUMENT
from .constants import SCRATCH_ARGUMENT
from .constants import GRID_DATA_ARGUMENT
from .constants import TILE_GRID_INDEX_ARGUMENT
from .constants import TILE_LEVEL_ARGUMENT
from .constants import TILE_LO_ARGUMENT
from .constants import TILE_HI_ARGUMENT
from .constants import TILE_LBOUND_ARGUMENT
from .constants import TILE_UBOUND_ARGUMENT
from .constants import TILE_INTERIOR_ARGUMENT
from .constants import TILE_ARRAY_BOUNDS_ARGUMENT
from .constants import TILE_DELTAS_ARGUMENT
from .constants import TILE_COORDINATES_ARGUMENT
from .constants import TILE_FACE_AREAS_ARGUMENT
from .constants import TILE_CELL_VOLUMES_ARGUMENT
from .constants import TILE_ARGUMENTS_ALL
from .constants import THREAD_INDEX_ARGUMENT
from .constants import THREAD_INDEX_VAR_NAME

# Custom exceptions
from .LogicError import LogicError

# Functions used by classes
from .check_operation_specification import check_operation_specification
from .check_grid_specification import check_grid_specification
from .check_subroutine_specification import check_subroutine_specification
from .check_tile_specification import check_tile_specification
from .check_external_specification import check_external_specification
from .check_scratch_specification import check_scratch_specification
from .check_grid_data_specification import check_grid_data_specification
from .check_thread_index_specification import check_thread_index_specification

# classes
from .TaskFunction import TaskFunction
from .TaskFunctionAssembler import TaskFunctionAssembler
from .AbcLogger import AbcLogger
from .BasicLogger import BasicLogger
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
