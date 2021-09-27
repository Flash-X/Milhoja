/**
 * C/C++ interoperability layer - compile C++ code with C linkage convention
 *
 * Refer to documentation in milhoja_runtime_C_interface for general
 * Fortran/C/C++ interoperability documentation.
 */

#include <cstdio>
#include <iostream>

#include "milhoja_interface_error_codes.h"
#include "DataItem.h"
#include "Tile.h"

#include "Grid_IntVect.h"

// NOTE: Most, if not all, functions in this C-interface do a cast from a
// generic dataItem pointer to a more-specific Tile pointer using C++'s
// dynamic_cast.  This throws a runtime error if the dataItem pointer
// doesn't point to an actual Tile object.  TODO: What if the pointer is null?

extern "C" {
    /**
     * gId is specific to the AMR library and is only used by the library.
     * Therefore, its index set specification is unimportant.
     *
     * level index is 1-based.
     *
     * lo, hi, loGC, hiGC are all 1-based.
     *
     * \todo Does converting item to a void* help with the C linkage?  Would
     *       this prevent us from getting the runtime error checking of the
     *       dynamic_cast?
     * \todo error check cast for overflow.
     * \todo error check given pointers.
     * \todo associate debug output with fortran/c layer verbosity level?
     * \todo The use of local variables to cache coordinates was done as part
     *       of developing this layer.  Is it possible to get rid of this
     *       and therefore directly connect the arguments to the data source?
     */
    int   milhoja_tile_get_metadata_c(orchestration::DataItem* item,
                                      int* gId, int* level,
                                      int* lo, int* hi, int* loGC, int* hiGC) {
        using namespace orchestration;

        try {
            Tile*   tileDesc = dynamic_cast<Tile*>(item);

            // Level is 0-based in Grid C++ code
            *gId   = static_cast<int>(tileDesc->gridIndex());
            *level = static_cast<int>(tileDesc->level()) + 1;

            int  lo_i, hi_i, loGC_i, hiGC_i;
            int  lo_j, hi_j, loGC_j, hiGC_j;
            int  lo_k, hi_k, loGC_k, hiGC_k;
            tileDesc->lo(&lo_i, &lo_j, &lo_k);
            tileDesc->hi(&hi_i, &hi_j, &hi_k);
            tileDesc->loGC(&loGC_i, &loGC_j, &loGC_k);
            tileDesc->hiGC(&hiGC_i, &hiGC_j, &hiGC_k);

            // spatial indices are 0-based in C++ code
            lo[0]   = lo_i   + 1;
            hi[0]   = hi_i   + 1;
            loGC[0] = loGC_i + 1;
            hiGC[0] = hiGC_i + 1;
#if NDIM >= 2
            lo[1]   = lo_j   + 1;
            hi[1]   = hi_j   + 1;
            loGC[1] = loGC_j + 1;
            hiGC[1] = hiGC_j + 1;
#endif
#if NDIM == 3
            lo[2]   = lo_k   + 1;
            hi[2]   = hi_k   + 1;
            loGC[2] = loGC_k + 1;
            hiGC[2] = hiGC_k + 1;
#endif

// This was useful for developing the Fortran/C interoperability layer
//            printf("[milhoja_tile_get_metadata_c] Tile %d (%p)\n\tlo_ijk=(%d,%d,%d)\n\thi_ijk=(%d,%d,%d)\n\tloGC_ijk=(%d,%d,%d)\n\thiGC_ijk=(%d,%d,%d)\n\tlo (%p)=(%d,%d,%d)\n\thi (%p)=(%d,%d,%d)\n\tloGC (%p)=(%d,%d,%d)\n\thiGC (%p)=(%d,%d,%d)\n",
//                   *gId, tileDesc, 
//                   lo_i,   lo_j,   lo_k, 
//                   hi_i,   hi_j,   hi_k, 
//                   loGC_i, loGC_j, loGC_k, 
//                   hiGC_i, hiGC_j, hiGC_k, 
//                   lo,     lo[0],   lo[1],   lo[2],
//                   hi,     hi[0],   hi[1],   hi[2],
//                   loGC, loGC[0], loGC[1], loGC[2],
//                   hiGC, hiGC[0], hiGC[1], hiGC[2]);
        } catch (const std::exception& exc) {
            std::cerr << "[milhoja_tile_get_metadata_c] Unable to get limits\n" 
                      << exc.what() << std::endl;
            return ERROR_UNABLE_TO_GET_LIMITS;
        } catch (...) {
            std::cerr << "[milhoja_tile_get_metadata_c] Unable to get limits\n" 
                      << "Unknown error caught" << std::endl;
            return ERROR_UNABLE_TO_GET_LIMITS;
        }

        return 0;
    }

    /**
     * \todo error check given pointers.
     * \todo associate debug output with fortran/c layer verbosity level?
     */
    int   milhoja_tile_get_data_ptr_c(orchestration::DataItem* item,
                                      orchestration::Real** data) {
        using namespace orchestration;

        try {
            Tile*   tileDesc = dynamic_cast<Tile*>(item);
            Real*   ptr = tileDesc->dataPtr();
            *data = ptr;
// This was useful for developing the Fortran/C interoperability layer
//            printf("[milhoja_tile_get_data_ptr_c]\n\tTile %p\n\tdataPtr=%p\n\tUptr=%p\n",
//                   tileDesc, ptr, data);
        } catch (const std::exception& exc) {
            std::cerr << "[milhoja_tile_get_data_ptr_c] Unable to get pointer\n" 
                      << exc.what() << std::endl;
            return ERROR_UNABLE_TO_GET_POINTER;
        } catch (...) {
            std::cerr << "[milhoja_tile_get_data_ptr_c] Unable to get pointer\n" 
                      << "Unknown error caught" << std::endl;
            return ERROR_UNABLE_TO_GET_POINTER;
        }

        return 0;
    }
}

