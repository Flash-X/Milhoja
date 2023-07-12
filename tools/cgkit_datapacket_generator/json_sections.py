"""A file that constains the name constants of all sections in the JSON. If something is being loaded from a section in the JSON, this file should be used to keep consistency."""

ORDER = "task-function-argument-list"
EXTRA_STREAMS = "n-extra-streams"
GENERAL = "constructor"
T_SCRATCH = "tile-scratch"
T_MDATA = "tile-metadata"
T_IN = "tile-in"
T_IN_OUT = "tile-in-out"
T_OUT = "tile-out"
LBOUND = 'lbound'

ALL_SECTIONS = { GENERAL, LBOUND, T_MDATA, T_IN, T_IN_OUT, T_OUT, T_SCRATCH }
