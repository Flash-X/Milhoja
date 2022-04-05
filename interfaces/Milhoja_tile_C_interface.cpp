/**
 * C/C++ interoperability layer - compile C++ code with C linkage convention
 *
 * Refer to documentation in milhoja_runtime_C_interface for general
 * Fortran/C/C++ interoperability documentation.
 */

#include <iostream>

#include "Milhoja.h"
#include "Milhoja_real.h"
#include "Milhoja_IntVect.h"
#include "Milhoja_axis.h"
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
                                      int* nCcVars, int* nFluxVars,
                                      milhoja::Real** data,
                                      milhoja::Real** fluxX, 
                                      milhoja::Real** fluxY, 
                                      milhoja::Real** fluxZ) {
        if (!item) {
            std::cerr << "[milhoja_tile_get_metadata_c] Null item pointer" << std::endl;
            return MILHOJA_ERROR_POINTER_IS_NULL;
        } else if (!gId || !level || !lo || !hi || !loGC || !hiGC || !nCcVars || !nFluxVars) {
            std::cerr << "[milhoja_tile_get_metadata_c] Null pointers" << std::endl;
            return MILHOJA_ERROR_POINTER_IS_NULL;
        } else if (!data) {
            std::cerr << "[milhoja_tile_get_metadata_c] data is null" << std::endl;
            return MILHOJA_ERROR_POINTER_IS_NULL; 
        } else if (*data) {
            std::cerr << "[milhoja_tile_get_metadata_c] *data must be null" << std::endl;
            return MILHOJA_ERROR_POINTER_NOT_NULL; 
        } else if (!fluxX) {
            std::cerr << "[milhoja_tile_get_metadata_c] fluxX null" << std::endl;
            return MILHOJA_ERROR_POINTER_IS_NULL;
        } else if (*fluxX) {
            std::cerr << "[milhoja_tile_get_metadata_c] *fluxX not null" << std::endl;
            return MILHOJA_ERROR_POINTER_NOT_NULL; 
#if MILHOJA_NDIM >= 2
        } else if (!fluxY) {
            std::cerr << "[milhoja_tile_get_metadata_c] fluxY null" << std::endl;
            return MILHOJA_ERROR_POINTER_IS_NULL;
        } else if (*fluxY) {
            std::cerr << "[milhoja_tile_get_metadata_c] *fluxY not null" << std::endl;
            return MILHOJA_ERROR_POINTER_NOT_NULL; 
#endif
#if MILHOJA_NDIM == 3
        } else if (!fluxZ) {
            std::cerr << "[milhoja_tile_get_metadata_c] fluxZ null" << std::endl;
            return MILHOJA_ERROR_POINTER_IS_NULL;
        } else if (*fluxZ) {
            std::cerr << "[milhoja_tile_get_metadata_c] *fluxZ not null" << std::endl;
            return MILHOJA_ERROR_POINTER_NOT_NULL; 
#endif
        }

        try {
            milhoja::Tile*   tileDesc = static_cast<milhoja::Tile*>(item);

            // Level is 0-based in Grid C++ code
            *gId   =                  tileDesc->gridIndex();
            *level = static_cast<int>(tileDesc->level()) + 1;
            milhoja::IntVect  loVec   = tileDesc->lo();
            milhoja::IntVect  hiVec   = tileDesc->hi();
            milhoja::IntVect  loVecGC = tileDesc->loGC();
            milhoja::IntVect  hiVecGC = tileDesc->hiGC();

            unsigned int nCcVars_ui   = tileDesc->nCcVariables();
            unsigned int nFluxVars_ui = tileDesc->nFluxVariables();
            *nCcVars   = static_cast<int>(nCcVars_ui);
            *nFluxVars = static_cast<int>(nFluxVars_ui);
            *data = tileDesc->dataPtr();

            std::vector<milhoja::Real*>   fluxPtrs = tileDesc->fluxDataPtrs();
            auto   nFluxes = fluxPtrs.size();
            if ((nFluxes != 0) && (nFluxes != MILHOJA_NDIM)) {
                std::cerr << "[milhoja_tile_get_metadata_c] Invalid N fluxes" << std::endl;
                return MILHOJA_ERROR_INVALID_N_FLUX_VARS; 
            } else if (nFluxes == 1) {
                *fluxX = fluxPtrs[milhoja::Axis::I];
            } else if (nFluxes == 2) {
                *fluxX = fluxPtrs[milhoja::Axis::I];
                *fluxY = fluxPtrs[milhoja::Axis::J];
            } else if (nFluxes == 3) {
                *fluxX = fluxPtrs[milhoja::Axis::I];
                *fluxY = fluxPtrs[milhoja::Axis::J];
                *fluxZ = fluxPtrs[milhoja::Axis::K];
            }

            // spatial indices are 0-based in C++ code
            lo[0]   = loVec[0]   + 1;
            hi[0]   = hiVec[0]   + 1;
            loGC[0] = loVecGC[0] + 1;
            hiGC[0] = hiVecGC[0] + 1;
#if MILHOJA_NDIM >= 2
            lo[1]   = loVec[1]   + 1;
            hi[1]   = hiVec[1]   + 1;
            loGC[1] = loVecGC[1] + 1;
            hiGC[1] = hiVecGC[1] + 1;
#endif
#if MILHOJA_NDIM == 3
            lo[2]   = loVec[2]   + 1;
            hi[2]   = hiVec[2]   + 1;
            loGC[2] = loVecGC[2] + 1;
            hiGC[2] = hiVecGC[2] + 1;
#endif
        } catch (const std::exception& exc) {
            std::cerr << exc.what() << std::endl;
            return MILHOJA_ERROR_UNABLE_TO_GET_METADATA;
        } catch (...) {
            std::cerr << "[milhoja_tile_get_metadata_c] Unknown error caught" << std::endl;
            return MILHOJA_ERROR_UNABLE_TO_GET_METADATA;
        }

        return MILHOJA_SUCCESS;
    }
}

