import sys
import warnings

known_types = {
    'deltas': 'RealVect',
    'lo': "IntVect",
    'hi': "IntVect",
    'loGC': "IntVect",
    'hiGC': "IntVect"
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

ispace_map = {
    'cc': "(nxb + 2 * {guard} * MILHOJA_K1D) * (nyb + 2 * {guard} * MILHOJA_K2D) * (nzb + 2 * {guard} * MILHOJA_K3D) * ({unk}) * sizeof({size})",
    'fcx': "((nxb+1) + 2 * {guard}) * ((nyb) + 2 * {guard}) * ((nzb) + 2 * {guard}) * {unk} * sizeof({size})",
    'fcy': "((nxb) + 2 * {guard}) * ((nyb+1) + 2 * {guard}) * ((nzb) + 2 * {guard}) * {unk} * sizeof({size})",
    'fcz': "((nxb) + 2 * {guard}) * ((nyb) + 2 * {guard}) * ((nzb+1) + 2 * {guard}) * {unk} * sizeof({size})"
}

constructor_args = {
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

# A helper method that parses the extents array in the JSON file.
# returns the final string to be used in the code.
def parse_extents(extents, size='') -> str:
    # check if extents is a string or or an enumerable
    if isinstance(extents, str):
        if extents[-1] == ')': extents = extents[:-1]
        else: print(f"{extents} is not closed properly.")
        sp = extents.split('(')
        indexer = sp[0]
        sp = sp[1].split(',')
        nguard = sp[0].upper().strip()
        nunkvar = sp[1].upper().strip()

        try:
            nguard = int(nguard)
        except:
            # print("Nguard is a string...")
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

        try:
            nunkvar = int(nunkvar)
        except:
            # print("Nguard is a string...")
            # if nguard is in the constants use as is
            if nunkvar not in constants:
                # our constants is contained in nguard, then we can use the string as is but print an error.
                for constant in constants:
                    if constant in nunkvar:
                        break # if we find one of the constants in the string
                else: #no break
                    print("Constant not found in string. Aborting.", file=sys.stderr)
                    exit(-1)
                warnings.warn("Constant found in string, continuing...")
        
        return ispace_map[indexer].format(guard=nguard, unk=nunkvar, size=size), nunkvar, indexer
    
    elif isinstance(extents, list):
        return "(" + ' * '.join([str(item) for item in extents]) + f'){ "" if size == "" else " * sizeof({size})" }', extents[-1], None
    else:
        print("Extents is not a string or list of numbers. Please refer to the documentation.")
        exit(-1)