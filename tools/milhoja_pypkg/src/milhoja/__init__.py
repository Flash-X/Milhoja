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
    LBOUND_ARGUMENT,
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
    GRID_DATA_LBOUNDS, GRID_DATA_EXTENTS,
    C2F_TYPE_MAPPING, VECTOR_ARRAY_EQUIVALENT,
    GRID_DATA_PTRS, SOURCE_DATATYPES, F2C_TYPE_MAPPING,
    THREAD_INDEX_ARGUMENT,
    THREAD_INDEX_VAR_NAME,
    SUPPORTED_LANGUAGES,
    SUPPORTED_PROCESSORS
)

# Custom exceptions
from .LogicError import LogicError
# Functions used by classes
from .check_grid_specification import check_grid_specification
from .check_tile_specification import check_tile_specification
from .check_external_specification import check_external_specification
from .check_scratch_specification import check_scratch_specification
from .check_grid_data_specification import check_grid_data_specification
from .check_lbound_specification import check_lbound_specification
from .check_thread_index_specification import check_thread_index_specification
from .check_partial_tf_specification import check_partial_tf_specification

# classes
from .AbcLogger import AbcLogger
from .BasicLogger import BasicLogger
from .SubroutineGroup import SubroutineGroup
from .TaskFunction import TaskFunction
from .TaskFunctionAssembler import TaskFunctionAssembler
from .AbcCodeGenerator import AbcCodeGenerator
from .TileWrapperGenerator import TileWrapperGenerator
from .TileWrapperModGenerator import TileWrapperModGenerator
from .TaskFunctionGenerator_cpu_cpp import TaskFunctionGenerator_cpu_cpp
from .DataPacketGenerator import DataPacketGenerator
from .TaskFunctionCpp2CGenerator_cpu_F import TaskFunctionCpp2CGenerator_cpu_F
from .TaskFunctionC2FGenerator_cpu_F import TaskFunctionC2FGenerator_cpu_F
from .TaskFunctionGenerator_OpenACC_F import TaskFunctionGenerator_OpenACC_F
from .TaskFunctionGenerator_cpu_F import TaskFunctionGenerator_cpu_F
from .TaskFunctionC2FGenerator_OpenACC_F \
    import TaskFunctionC2FGenerator_OpenACC_F
from .TaskFunctionCpp2CGenerator_OpenACC_F \
    import TaskFunctionCpp2CGenerator_OpenACC_F
from .DataPacketModGenerator import DataPacketModGenerator

# Functions that use classes
from .generate_data_item import generate_data_item
from .generate_task_function import generate_task_function
from .generate_packet_file import generate_packet_file

# ----- Python unittest-based test framework
# Used for automatic test discovery
from .load_tests import load_tests

# Allow users to run full test suite as milhoja.test()
from .test import test
