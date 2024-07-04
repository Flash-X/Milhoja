/**
 * C/C++ interoperability layer - compile C++ code with C linkage convention
 *
 * Refer to documentation in Milhoja_runtime_C_interface for general
 * Fortran/C/C++ interoperability documentation.
 */

#include <iostream>

#include "Milhoja_Grid.h"
#include "Milhoja_Tile.h"
#include "Milhoja_TileIter.h"
#include "Milhoja_interface_error_codes.h"

#ifdef FULL_MILHOJAGRID
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
        if        (!itor) {
            std::cerr << "[milhoja_itor_build_c] itor is null" << std::endl;
            return MILHOJA_ERROR_POINTER_IS_NULL; 
        } else if (*itor) {
            std::cerr << "[milhoja_itor_build_c] *itor not null" << std::endl;
            return MILHOJA_ERROR_POINTER_NOT_NULL; 
        }

        try {
            milhoja::Grid&   grid = milhoja::Grid::instance();
            *itor = static_cast<void*>(grid.buildTileIter_forFortran(0));
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
            std::cerr << "[milhoja_itor_destroy_c] itor is null" << std::endl;
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
     * \param  itor     The iterator to validate.
     * \param  isValid  True if the iterator is valid and can be advanced with
     *                  next.  False, if the iterator is invalid and should
     *                  *not* be advanced with next.
     * \return The milhoja error code
     */
    int    milhoja_itor_is_valid_c(void* itor, bool* isValid) {
        if (!itor) {
            std::cerr << "[milhoja_itor_is_valid_c] itor is null" << std::endl;
            return MILHOJA_ERROR_POINTER_IS_NULL; 
        } else if (!isValid) {
            std::cerr << "[milhoja_itor_is_valid_c] isValid is null" << std::endl;
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
     * \param  itor   The iterator to advance.
     * \return The milhoja error code
     */
    int    milhoja_itor_next_c(void* itor) {
        if (!itor) {
            std::cerr << "[milhoja_itor_next_c] itor is null" << std::endl;
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
     * Obtain a pointer to the tile object that is currently indexed by the
     * iterator.
     *
     * As part of this, resources for the tile object are allocated dynamically.
     * The calling code therefore takes ownership of the tile and its resources
     * and is responsible for releasing the tile and its resources.
     *
     * It is intended that Fortran calling code acquire the tile pointer, use
     * the pointer to access all of the tile's metadata, cache the metadata, and
     * immediately release the tile/tile resources.
     *
     * \param  itor   The pointer to the iterator.
     * \param  tile   The variable associated with this pointer will be set to
     *                the pointer to the current tile.
     * \return The milhoja error code
     */
    int    milhoja_itor_acquire_current_tile_c(void* itor, void** tile) {
        if        (!itor) {
            std::cerr << "[milhoja_itor_acquire_current_tile_c] itor is null" << std::endl;
            return MILHOJA_ERROR_POINTER_IS_NULL; 
        } else if (!tile) {
            std::cerr << "[milhoja_itor_acquire_current_tile_c] tile is null" << std::endl;
            return MILHOJA_ERROR_POINTER_IS_NULL; 
        } else if (*tile) {
            std::cerr << "[milhoja_itor_acquire_current_tile_c] *tile is not null" << std::endl;
            return MILHOJA_ERROR_POINTER_NOT_NULL; 
        }
        milhoja::TileIter*   MH_itor = static_cast<milhoja::TileIter*>(itor);

        try {
            *tile = static_cast<void*>(MH_itor->buildCurrentTile_forFortran());
        } catch (const std::exception& exc) {
            std::cerr << exc.what() << std::endl;
            return MILHOJA_ERROR_UNABLE_TO_ACQUIRE_TILE;
        } catch (...) {
            std::cerr << "[milhoja_itor_acquire_current_tile_c] Unknown error caught" << std::endl;
            return MILHOJA_ERROR_UNABLE_TO_ACQUIRE_TILE;
        }

        return MILHOJA_SUCCESS;
    }

    /**
     * Release the resources associated with the tile as well as the tile
     * itself.  Refer to the documentation for
     * milhoja_itor_acquire_current_tile_c for more information.
     *
     * Note that the tile resource management scheme is different when tiles are
     * accessed directly in Fortran using the iterator compared to tile access
     * in Fortran via the runtime.  It was decided to put this release mechanism
     * in the iterator (as opposed to tile) since ownership of tile resources is
     * assumed only using the iterator - routines in this interface provide
     * access, therefore releasing should also be only in this interface.
     *
     * \param tile   The tile to release
     * \return The milhoja error code
     */
    int    milhoja_itor_release_current_tile_c(void* tile) {
        if (!tile) {
            std::cerr << "[milhoja_itor_release_current_tile_c] tile is null" << std::endl;
            return MILHOJA_ERROR_POINTER_IS_NULL; 
        }
        milhoja::Tile*   toDelete = static_cast<milhoja::Tile*>(tile);

        try {
            delete toDelete;
            toDelete = nullptr;
        } catch (const std::exception& exc) {
            std::cerr << exc.what() << std::endl;
            return MILHOJA_ERROR_UNABLE_TO_RELEASE_TILE;
        } catch (...) {
            std::cerr << "[milhoja_itor_release_current_tile_c] Unknown error caught" << std::endl;
            return MILHOJA_ERROR_UNABLE_TO_RELEASE_TILE;
        }

        return MILHOJA_SUCCESS;
    }
}
#endif
