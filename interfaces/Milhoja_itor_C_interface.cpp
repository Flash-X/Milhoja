/**
 * C/C++ interoperability layer - compile C++ code with C linkage convention
 *
 * Refer to documentation in Milhoja_runtime_C_interface for general
 * Fortran/C/C++ interoperability documentation.
 */

#include <iostream>

#include "Milhoja_interface_error_codes.h"
#include "Milhoja_Grid.h"
#include "Milhoja_Tile.h"
#include "Milhoja_TileIter.h"

extern "C" {
    /**
     * Build and access a tile iterator.  This includes allocating dynamically
     * memory for the iterator object.  The pointer to this object is given to
     * calling code and calling code owns this resource.  As a consequence,
     * calling code is required to call milhoja_itor_destroy_c once it is
     * finished with the iterator and to pass to the function the pointer that
     * it received when it called this function.
     *
     * It is the responsibility of calling code to ensure that the pointer is
     * used in common and reasonable ways so that the pointer is never
     * "dangling".  For instance, the pointer should not be used if the Grid
     * data structures might have been altered by actions such as regridding
     * after the pointer was acquired.
     *
     * \param  itor   The pointer whose variable is to be set to the pointer to
     *                the dynamically allocated iterator object.
     * \return The milhoja error code
     */
    int    milhoja_itor_build_c(void** itor) {
        if (*itor) {
            std::cerr << "[milhoja_itor_build_c] Pointer already allocated" << std::endl;
            return MILHOJA_ERROR_POINTER_NOT_NULL; 
        }

        try {
            milhoja::Grid&   grid = milhoja::Grid::instance();
            *itor = static_cast<void*>(grid.buildTileIter_ForFortran(0));
        } catch (const std::exception& exc) {
            std::cerr << exc.what() << std::endl;
            return MILHOJA_ERROR_UNABLE_TO_BUILD_ITERATOR;
        } catch (...) {
            std::cerr << "[milhoja_itor_build_c] Unknown error caught" << std::endl;
            return MILHOJA_ERROR_UNABLE_TO_BUILD_ITERATOR;
        }

        return MILHOJA_SUCCESS;
    }

    /**
     * Destroy the given iterator.  This includes releasing dynamically
     * allocated resources.
     *
     * \param  itor   The pointer to the iterator to destroy.
     * \return The milhoja error code
     */
    int    milhoja_itor_destroy_c(void* itor) {
        if (!itor) {
            std::cerr << "[milhoja_itor_destroy_c] Pointer not allocated" << std::endl;
            return MILHOJA_ERROR_POINTER_IS_NULL; 
        }
        milhoja::TileIter*   toDelete = static_cast<milhoja::TileIter*>(itor);

        try {
            delete toDelete;
            toDelete = nullptr;
        } catch (const std::exception& exc) {
            std::cerr << exc.what() << std::endl;
            return MILHOJA_ERROR_UNABLE_TO_DESTROY_ITERATOR;
        } catch (...) {
            std::cerr << "[milhoja_itor_destroy_c] Unknown error caught" << std::endl;
            return MILHOJA_ERROR_UNABLE_TO_DESTROY_ITERATOR;
        }

        return MILHOJA_SUCCESS;
    }

    /**
     * Determine if the given iterator is valid and, therefore, if calling code
     * can safely call next().
     *
     * \param  itor     The pointer to the iterator to validate.
     * \param  isValid  True if the iterator is valid and can be advanced with
     *                  next.  False, if the iterator is set to the last tile
     *                  and should *not* be advanced with next.
     * \return The milhoja error code
     */
    int    milhoja_itor_is_valid_c(void* itor, bool* isValid) {
        if (!itor) {
            std::cerr << "[milhoja_itor_is_valid_c] Pointer not allocated" << std::endl;
            return MILHOJA_ERROR_POINTER_IS_NULL; 
        }
        milhoja::TileIter*   MH_itor = static_cast<milhoja::TileIter*>(itor);

        try {
            *isValid = MH_itor->isValid(); 
        } catch (const std::exception& exc) {
            std::cerr << exc.what() << std::endl;
            return MILHOJA_ERROR_UNABLE_TO_VALIDATE_ITERATOR;
        } catch (...) {
            std::cerr << "[milhoja_itor_is_valid_c] Unknown error caught" << std::endl;
            return MILHOJA_ERROR_UNABLE_TO_VALIDATE_ITERATOR;
        }

        return MILHOJA_SUCCESS;
    }

    /**
     * Advance the iterator to the next tile.  Refer to the documentation for
     * isValid() for more information on the proper usage of this function.
     *
     * \param  itor   The pointer to the iterator to advance.
     * \return The milhoja error code
     */
    int    milhoja_itor_next_c(void* itor) {
        if (!itor) {
            std::cerr << "[milhoja_itor_next_c] Pointer not allocated" << std::endl;
            return MILHOJA_ERROR_POINTER_IS_NULL; 
        }
        milhoja::TileIter*   MH_itor = static_cast<milhoja::TileIter*>(itor);

        try {
            MH_itor->next();
        } catch (const std::exception& exc) {
            std::cerr << exc.what() << std::endl;
            return MILHOJA_ERROR_UNABLE_TO_ADVANCE_ITERATOR;
        } catch (...) {
            std::cerr << "[milhoja_itor_next_c] Unknown error caught" << std::endl;
            return MILHOJA_ERROR_UNABLE_TO_ADVANCE_ITERATOR;
        }

        return MILHOJA_SUCCESS;
    }

    /**
     * Obtain the minimal set of metadata for the current tile.
     *
     * \todo Can we make a Fortran version that will accept the C pointers
     * directly and set the values under the hood?  If so, then the routine
     * could set the data directly into the Fortran variable.  Doing this now
     * would be premature optimization.
     *
     * \param  itor   The pointer to the iterator.
     * \return The milhoja error code
     */
    int    milhoja_itor_current_tile_metadata_c(void* itor) {
        if (!itor) {
            std::cerr << "[milhoja_itor_current_tile_metadata_c] Pointer not allocated" << std::endl;
            return MILHOJA_ERROR_POINTER_IS_NULL; 
        }
        milhoja::TileIter*   MH_itor = static_cast<milhoja::TileIter*>(itor);

        try {
            std::unique_ptr<milhoja::Tile>   tileDesc = MH_itor->buildCurrentTile();
            unsigned int        gridIndex = tileDesc->gridIndex();
            milhoja::IntVect    lo        = tileDesc->lo();
            milhoja::IntVect    hi        = tileDesc->hi();
            std::cout << "Tile " << gridIndex << "\n";
            std::cout << "   lo = ("
                      << lo[0] << ", "
                      << lo[1] << ", "
                      << lo[2] << ")\n";
            std::cout << "   hi = ("
                      << hi[0] << ", "
                      << hi[1] << ", "
                      << hi[2] << ")" << std::endl;
        } catch (const std::exception& exc) {
            std::cerr << exc.what() << std::endl;
            return MILHOJA_ERROR_UNABLE_TO_GET_TILE_METADATA;
        } catch (...) {
            std::cerr << "[milhoja_itor_current_tile_metadata_c] Unknown error caught" << std::endl;
            return MILHOJA_ERROR_UNABLE_TO_GET_TILE_METADATA;
        }

        return MILHOJA_SUCCESS;
    }
}

