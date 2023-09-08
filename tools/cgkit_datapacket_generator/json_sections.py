"""
A file that constains the name constants of all sections in the JSON. 
If something is being loaded from a section in the JSON, this file should be used to keep consistency.
"""

# Packet Metadata keywords
#: Determines the byte-align to use for the packet.
BYTE_ALIGN = 'byte_align'
#: Task function argument list argument order.
ORDER = "task_function_argument_list"
#: The number of extra streams to use. 
EXTRA_STREAMS = "n_extra_streams"
TASK_FUNCTION_NAME = 'task_function_name'
FILE_NAME = "file_name"
NAME = "name"
LANG = "language"
OUTER = "outer"
HELPERS = "helpers"
SIZES = "sizes"

# All sections.
GENERAL = "constructor"
T_SCRATCH = "tile_scratch"
T_MDATA = "tile_metadata"
T_IN = "tile_in"
T_IN_OUT = "tile_in_out"
T_OUT = "tile_out"

# Possible keys
EXTENTS = 'extents'
START = 'start'
END = 'end'
START_IN = 'start_in'
START_OUT = 'start_out'
END_IN = 'end_in'
END_OUT = 'end_out'
DTYPE = 'type'
LBOUND = 'lbound'
LB_IN = 'lbound_in'
LB_OUT = 'lbound_out'
EXT_IN = 'extents_in'
EXT_OUT = 'extents_out'

ALL_SECTIONS = { GENERAL, LBOUND, T_MDATA, T_IN, T_IN_OUT, T_OUT, T_SCRATCH }
