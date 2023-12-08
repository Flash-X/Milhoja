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
from .constants import (
    MILHOJA_JSON_FORMAT, CURRENT_MILHOJA_JSON_VERSION,
    TASK_FUNCTION_FORMATS,
    LOG_LEVEL_NONE, LOG_LEVEL_BASIC, LOG_LEVEL_BASIC_DEBUG,
    LOG_LEVEL_MAX, LOG_LEVELS,
    INTERNAL_ARGUMENT,
    EXTERNAL_ARGUMENT,
    SCRATCH_ARGUMENT,
    GRID_DATA_ARGUMENT,
    TILE_GRID_INDEX_ARGUMENT,
    TILE_LEVEL_ARGUMENT,
    TILE_LO_ARGUMENT, TILE_HI_ARGUMENT,
    TILE_LBOUND_ARGUMENT, TILE_UBOUND_ARGUMENT,
    TILE_INTERIOR_ARGUMENT, TILE_ARRAY_BOUNDS_ARGUMENT,
    TILE_DELTAS_ARGUMENT,
    TILE_COORDINATES_ARGUMENT,
    TILE_FACE_AREAS_ARGUMENT,
    TILE_CELL_VOLUMES_ARGUMENT,
    TILE_ARGUMENTS_ALL,
    THREAD_INDEX_ARGUMENT, THREAD_INDEX_VAR_NAME, GRID_DATA_LBOUNDS
)

# Custom exceptions
from .LogicError import LogicError

# classes
from .AbcLogger import AbcLogger
from .BasicLogger import BasicLogger
from .TaskFunction import TaskFunction
from .AbcCodeGenerator import AbcCodeGenerator
from .TileWrapperGenerator_cpp import TileWrapperGenerator_cpp
from .TaskFunctionGenerator_cpu_cpp import TaskFunctionGenerator_cpu_cpp
from .DataPacketGenerator import DataPacketGenerator

# Functions that use classes
from .generate_data_item import generate_data_item
from .generate_task_function import generate_task_function
from .generate_packet_file import generate_packet_file

# ----- Python unittest-based test framework
# Used for automatic test discovery
from .load_tests import load_tests

# Allow users to run full test suite as milhoja.test()
from .test import test
