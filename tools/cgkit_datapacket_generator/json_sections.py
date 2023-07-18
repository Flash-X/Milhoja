"""A file that constains the name constants of all sections in the JSON. If something is being loaded from a section in the JSON, this file should be used to keep consistency."""

# All sections.
BYTE_ALIGN = 'byte-align'
ORDER = "task-function-argument-list"
EXTRA_STREAMS = "n-extra-streams"
GENERAL = "constructor"
T_SCRATCH = "tile-scratch"
T_MDATA = "tile-metadata"
T_IN = "tile-in"
T_IN_OUT = "tile-in-out"
T_OUT = "tile-out"
LBOUND = 'lbound'

# Possible keys
EXTENTS = 'extents'
START = 'start'
END = 'end'
START_IN = 'start-in'
START_OUT = 'start-out'
END_IN = 'end-in'
END_OUT = 'end-out'
DTYPE = 'type'

ALL_SECTIONS = { GENERAL, LBOUND, T_MDATA, T_IN, T_IN_OUT, T_OUT, T_SCRATCH }
