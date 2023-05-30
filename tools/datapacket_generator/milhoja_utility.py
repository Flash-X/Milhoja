import sys
import warnings
from enum import Enum
from typing import Tuple
from typing import Union

class Language(Enum):
    cpp = 'cpp'
    fortran = 'fortran'

    def __str__(self):
        return self.value

tile_known_types = {
    'levels': 'unsigned int',
    'gridIndex': 'int',
    'tileIndex': 'int',
    'nCcVariables': 'unsigned int',
    'nFluxVariables': 'unsigned int',
    'deltas': 'RealVect',
    'lo': "IntVect",
    'hi': "IntVect",
    'loGC': "IntVect",
    'hiGC': "IntVect",
    'data': 'FArray4D',
    'dataPtr': 'Real*',
}

cpp_equiv = {
    "RealVect": "Real",
    "IntVect": "int"
}

imap = {
    'IntVect':  '<Milhoja_IntVect.h>',
    'Real':     '<Milhoja_real.h>',
    'RealVect': '<Milhoja_RealVect.h>',
    'FArray1D': '<Milhoja_FArray1D.h>',
    'FArray2D': '<Milhoja_FArray2D.h>',
    'FArray3D': '<Milhoja_FArray3D.h>',
    'FArray4D': '<Milhoja_FArray4D.h>'
}

# NOTE: These are the fortran spaces, while cpp size map is specifically for use with cpp.
fortran_size_map = {
    'cc': "nxbGC_h * nybGC_h * nzbGC_h * ( {unk} ) * sizeof({size})",
    'fcx': "(nxbGC_h + 1) * nybGC_h * nzbGC_h * ( {unk} ) * sizeof({size})",
    'fcy': "nxbGC_h * (nybGC_h + 1) * nzbGC_h * ( {unk} ) * sizeof({size})",
    'fcz': "nxbGC_h * nybGC_h * (nzbGC_h + 1) * ( {unk} ) * sizeof({size})"
}

cpp_size_map = {
    'cc': "(nxb_ + 2 * {guard} * MILHOJA_K1D) * (nyb_ + 2 * {guard} * MILHOJA_K2D) * (nzb_ + 2 * {guard} * MILHOJA_K3D) * ( {unk} ) * sizeof({size})",
    'fcx': "((nxb_+1) + 2 * {guard}) * ((nyb_) + 2 * {guard}) * ((nzb_) + 2 * {guard}) * ( {unk} ) * sizeof({size})",
    'fcy': "((nxb_) + 2 * {guard}) * ((nyb_+1) + 2 * {guard}) * ((nzb_) + 2 * {guard}) * ( {unk} ) * sizeof({size})",
    'fcz': "((nxb_) + 2 * {guard}) * ((nyb_) + 2 * {guard}) * ((nzb_+1) + 2 * {guard}) * ( {unk} ) * sizeof({size})"
}

finterface_constructor_args = {
    'cc': "loGC, hiGC",
    'fcx': "lo, IntVect{ LIST_NDIM( hi.I()+1, hi.J(), hi.K() ) }",
    'fcy': "lo, IntVect{ LIST_NDIM( hi.I(), hi.J()+1, hi.K() ) }",
    'fcz': "lo, IntVect{ LIST_NDIM( hi.I(), hi.J(), hi.K()+1 ) }"
}

# Government sanctioned constants.
constants = {
    "NGUARD",
    "NFLUXES",
    "NUNKVAR"
}

def parse_extents(extents, start, end, size='') -> Tuple[str, str, str]:
    """
    Parses the extents string found in the packet JSON file.

    Parameters:
        extents - The string to parse\n
        start - the starting index of the array\n
        end - the ending index of the array\n
        size - the variable type
    Returns:
        str - extents string\n
        str - nunkvars string\n
        str - indexer type\n
    """
    # check if extents is a string or or an enumerable
    if isinstance(extents, str):
        if extents[-1] == ')': extents = extents[:-1]
        else: print(f"{extents} is not closed properly.")
        sp = extents.split('(')
        indexer = sp[0]
        nguard = sp[1].strip()

        try:
            nguard = int(nguard)
        except:
            # if nguard is in the constants use as is
            if nguard not in constants:
                # our constants is contained in nguard, then we can use the string as is but print an error.
                for constant in constants:
                    if constant in nguard:
                        break # if we find one of the constants in the string
                else: #no break
                    print(f"{nguard} not found in string. Aborting.", file=sys.stderr)
                    exit(-1)
                warnings.warn("Constant found in string, continuing...")
            if nguard == "NGUARD":
                nguard = "nGuard_"
            elif nguard == "NFLUXES":
                nguard = "nCcVars_"
        
        return cpp_size_map[indexer].format(guard=nguard, unk=f"( ({end}) + ({start}) + 1 )", size=size), f"( ({end}) + ({start}) + 1 )", indexer
    
    elif isinstance(extents, list):
        return "(" + ' * '.join([str(item) for item in extents]) + f'){ "" if size == "" else " * sizeof({size})" }', extents[-1], None
    else:
        print("Extents is not a string or list of numbers. Please refer to the documentation.")
        exit(-1)