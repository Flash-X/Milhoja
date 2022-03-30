/**
 * C/C++ interoperability layer - compile C++ code with C linkage convention
 *
 * Refer to documentation in milhoja_runtime_C_interface for general
 * Fortran/C/C++ interoperability documentation.
 */

#include <iostream>

#include "Milhoja.h"
#include "Milhoja_real.h"
#include "Milhoja_Tile.h"
#include "Milhoja_interface_error_codes.h"

extern "C" {
    /**
     * gId is specific to the AMR library and is only used by the library.
     * Therefore, its index set specification is unimportant.
     *
     * level index is 1-based.
     *
     * lo, hi, loGC, hiGC are all 1-based.  Values above MILHOJA_NDIM are not
     * set or altered and, therefore, such values can be safely set before
     * calling this function.
     *
     * \todo error check casts for overflow.
     * \todo The use of local variables to cache coordinates was done as part
     *       of developing this layer.  Is it possible to get rid of this
     *       and therefore directly connect the arguments to the data source?
     * \todo Try to get all this data in one function call to Milhoja?
     */
    int   milhoja_tile_get_metadata_c(void* item,
                                      int* gId, int* level,
                                      int* lo, int* hi, int* loGC, int* hiGC,
                                      int* nVars, milhoja::Real** data) {
        if (!item) {
            std::cerr << "[milhoja_tile_get_metadata_c] Null item pointer" << std::endl;
            return MILHOJA_ERROR_POINTER_IS_NULL;
        } else if (!gId || !level || !lo || !hi || !loGC || !hiGC || !nVars) {
            std::cerr << "[milhoja_tile_get_metadata_c] Null pointers" << std::endl;
            return MILHOJA_ERROR_POINTER_IS_NULL;
        } else if (*data) {
            std::cerr << "[milhoja_tile_get_metadata_c] data already allocated" << std::endl;
            return MILHOJA_ERROR_POINTER_NOT_NULL; 
        }

        try {
            milhoja::Tile*   tileDesc = static_cast<milhoja::Tile*>(item);

            // Level is 0-based in Grid C++ code
            *gId   =                  tileDesc->gridIndex();
            *level = static_cast<int>(tileDesc->level()) + 1;

            int  lo_i, hi_i, loGC_i, hiGC_i;
            int  lo_j, hi_j, loGC_j, hiGC_j;
            int  lo_k, hi_k, loGC_k, hiGC_k;
            tileDesc->lo(&lo_i, &lo_j, &lo_k);
            tileDesc->hi(&hi_i, &hi_j, &hi_k);
            tileDesc->loGC(&loGC_i, &loGC_j, &loGC_k);
            tileDesc->hiGC(&hiGC_i, &hiGC_j, &hiGC_k);

            unsigned int nVars_ui = tileDesc->nVariables();
            *nVars = static_cast<int>(nVars_ui);

            *data = tileDesc->dataPtr();

            // spatial indices are 0-based in C++ code
            lo[0]   = lo_i   + 1;
            hi[0]   = hi_i   + 1;
            loGC[0] = loGC_i + 1;
            hiGC[0] = hiGC_i + 1;
#if MILHOJA_NDIM >= 2
            lo[1]   = lo_j   + 1;
            hi[1]   = hi_j   + 1;
            loGC[1] = loGC_j + 1;
            hiGC[1] = hiGC_j + 1;
#endif
#if MILHOJA_NDIM == 3
            lo[2]   = lo_k   + 1;
            hi[2]   = hi_k   + 1;
            loGC[2] = loGC_k + 1;
            hiGC[2] = hiGC_k + 1;
#endif
        } catch (const std::exception& exc) {
            std::cerr << "[milhoja_tile_get_metadata_c]" << exc.what() << std::endl;
            return MILHOJA_ERROR_UNABLE_TO_GET_METADATA;
        } catch (...) {
            std::cerr << "[milhoja_tile_get_metadata_c] Unknown error caught" << std::endl;
            return MILHOJA_ERROR_UNABLE_TO_GET_METADATA;
        }

        return MILHOJA_SUCCESS;
    }
}

