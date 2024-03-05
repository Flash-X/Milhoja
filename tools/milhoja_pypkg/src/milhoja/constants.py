# ----- SPECIFICATION FORMAT INFORMATION
# Format strings are used as tool command line arguments.  Therefore, keep
# short and don't insert white space.
MILHOJA_JSON_FORMAT = "Milhoja-JSON"
CURRENT_MILHOJA_JSON_VERSION = "1.0.0"

TASK_FUNCTION_FORMATS = [MILHOJA_JSON_FORMAT]

# ----- MILHOJA CODE GENERATION LOG CONFIGURATION
LOG_LEVEL_NONE = 0
LOG_LEVEL_BASIC = 1
LOG_LEVEL_BASIC_DEBUG = 2
LOG_LEVEL_MAX = 3

LOG_LEVELS = list(range(LOG_LEVEL_NONE, LOG_LEVEL_MAX+1))

# All code in the package that exists specifically to check for errors (e.g.,
# check_*_specification) should use this as its log tag
#
# Don't use the word error here since that might make it hard to grep out
# real errors in logs
ERROR_CHECK_LOG_TAG = "Milhoja Cordura"

# ----- TASK FUNCTION ARGUMENT CLASSIFICATION SCHEME
# Case-sensitive source keys to be used in specification files (e.g.,
# Milhoja-JSON files) throughout
#
# Scheme for keys
# - concise one word all lowercase for all keys when possible
# - tile metadata keys all begin with tile_ (all lowercase) with remainder
#   in camelcase with no separation
INTERNAL_ARGUMENT = "internal"
EXTERNAL_ARGUMENT = "external"
SCRATCH_ARGUMENT = "scratch"
GRID_DATA_ARGUMENT = "grid_data"
LBOUND_ARGUMENT = "lbound"
TILE_GRID_INDEX_ARGUMENT = "tile_gridIndex"
TILE_LEVEL_ARGUMENT = "tile_level"
TILE_LO_ARGUMENT = "tile_lo"
TILE_HI_ARGUMENT = "tile_hi"
TILE_LBOUND_ARGUMENT = "tile_lbound"
TILE_UBOUND_ARGUMENT = "tile_ubound"
TILE_DELTAS_ARGUMENT = "tile_deltas"
TILE_COORDINATES_ARGUMENT = "tile_coordinates"
TILE_FACE_AREAS_ARGUMENT = "tile_faceAreas"
TILE_CELL_VOLUMES_ARGUMENT = "tile_cellVolumes"
# These two are included for Flash-X and are 2D arrays
# with the first index being LOW/HIGH; the second, axis.
TILE_INTERIOR_ARGUMENT = "tile_interior"
TILE_ARRAY_BOUNDS_ARGUMENT = "tile_arrayBounds"

TILE_ARGUMENTS_ALL = {
    TILE_GRID_INDEX_ARGUMENT,
    TILE_LEVEL_ARGUMENT,
    TILE_LO_ARGUMENT, TILE_HI_ARGUMENT,
    TILE_LBOUND_ARGUMENT, TILE_UBOUND_ARGUMENT,
    TILE_INTERIOR_ARGUMENT, TILE_ARRAY_BOUNDS_ARGUMENT,
    TILE_DELTAS_ARGUMENT,
    TILE_COORDINATES_ARGUMENT,
    TILE_FACE_AREAS_ARGUMENT,
    TILE_CELL_VOLUMES_ARGUMENT
}

GRID_DATA_EXTENTS = {
    "CENTER": [
        '{0} + 2 * {1} * MILHOJA_K1D',
        '{0} + 2 * {1} * MILHOJA_K2D',
        '{0} + 2 * {1} * MILHOJA_K3D'
    ],
    "FLUXX": [
        '({0} + 1) + 2 * {1} * MILHOJA_K1D',
        '{0} + 2 * {1} * MILHOJA_K2D',
        '{0} + 2 * {1} * MILHOJA_K3D'
    ],
    "FLUXY": [
        '({0} + 2 * {1} * MILHOJA_K1D',
        '({0} + 1) + 2 * {1} * MILHOJA_K2D',
        '{0} + 2 * {1} * MILHOJA_K3D'
    ],
    "FLUXZ": [
        '{0} + 2 * {1} * MILHOJA_K1D',
        '{0} + 2 * {1} * MILHOJA_K2D',
        '({0} + 1) + 2 * {1} * MILHOJA_K3D'
    ]
}

# The lbounds for grid data are fairly consistent, so we can use this mapping
# to insert lbounds for grid data variables inside of the tf spec that need
# an lbound array.
GRID_DATA_LBOUNDS = {
    "CENTER": "(tile_lbound, {0})",  # CC data does use guard cells.
    "FLUXX": "(tile_lo, {0})",  # assume that flux arrays use 0 guard cells.
    "FLUXY": "(tile_lo, {0})",
    "FLUXZ": "(tile_lo, {0})"
}

# The type mapping for sources that have predetermined types.
# External and scratch do not have predetermined types, so they do not
# appear in here.
# todo::
#   * There are few unsupported tile metadata types like cell volumes and
#     face areas. Eventually these should be included in the source datatypes
#     mapping.
SOURCE_DATATYPES = {
    TILE_LO_ARGUMENT: "IntVect",
    TILE_HI_ARGUMENT: "IntVect",
    TILE_LBOUND_ARGUMENT: "IntVect",
    TILE_UBOUND_ARGUMENT: "IntVect",
    TILE_DELTAS_ARGUMENT: "RealVect",
    TILE_INTERIOR_ARGUMENT: "IntVect",
    TILE_ARRAY_BOUNDS_ARGUMENT: "IntVect",
    LBOUND_ARGUMENT: "IntVect",
    TILE_LEVEL_ARGUMENT: "unsigned int",
    GRID_DATA_ARGUMENT: "real"
}

# get string and call with format to insert the name of your tile descriptor
GRID_DATA_PTRS = {
    "CENTER": "{0}->dataPtr()",
    "FLUXX": "&{0}->fluxData(milhoja::Axis::I)",
    "FLUXY": "&{0}->fluxData(milhoja::Axis::J)",
    "FLUXZ": "&{0}->fluxData(milhoja::Axis::K)"
}

# This is specifically in reference to tile_metadata specific types like
# IntVect and RealVect, where the arrays need to be converted to an
# array of a primitive that can be used in fortran.
VECTOR_ARRAY_EQUIVALENT = {
    "IntVect": "int",
    "RealVect": "real"
}

# For converting Fortran types to C++/C types.
F2C_TYPE_MAPPING = {
    "logical": "bool",
    "real": "real",
    "integer": "int"
}

# For converting potential C++ types to Fortran types.
C2F_TYPE_MAPPING = {
    "bool": "logical",
    "int": "integer",
    "unsigned int": "integer",
    "real": "real"
}

TILE_KEY_MAPPING = {
    TILE_LO_ARGUMENT: "",
    TILE_HI_ARGUMENT: "",
    TILE_LBOUND_ARGUMENT: "",
    TILE_UBOUND_ARGUMENT: ""
}

# Task functions can include subroutines that take as an actual argument
# the unique thread index of the runtime thread that is effectively calling
# it.  Since this value is purely internal and is passed in, it is managed
# differently from other arguments.
THREAD_INDEX_ARGUMENT = "milhoja_thread_index"
# JSON generators need to insert the same variable name that the TF code
# generators use.
THREAD_INDEX_VAR_NAME = "threadIndex"

# TODO:: Should we have a specific list of type aliases that
#        we can include to avoid having to constantly manipulate strings?
C_TYPE_ALIASES = {
    "int": "",
    "milhoja::Real": "",
    "bool": "",
    "integer": "using integer = int;",
    "real": "using real = milhoja::Real;",
    "logical": "using logical = bool;"
}