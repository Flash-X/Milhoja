/* _connector:tf_cpp2c */

#include <iostream>
#include <Milhoja.h>
#include <Milhoja_real.h>
#include <Milhoja_IntVect.h>
#include <Milhoja_RealVect.h>
#include <Milhoja_Tile.h>
#include <Milhoja_interface_error_codes.h>

#include "_param:data_item_header_file_name"

using real = milhoja::Real;

extern "C" {
    //----- C DECLARATION OF FORTRAN ROUTINE WITH C-COMPATIBLE INTERFACE
    void _param:c2f_function_name (
        /* _link:c2f_dummy_args */
    );

    int _param:instance(
        /* _link:external_args */,
        void** wrapper
    ) {
        if (wrapper == nullptr) {
            std::cerr << "[_param:instance] wrapper is NULL" << std::endl;
            return MILHOJA_ERROR_POINTER_IS_NULL;
        } else if (*wrapper != nullptr) {
            std::cerr << "[_param:instance] *wrapper not NULL" << std::endl;
            return MILHOJA_ERROR_POINTER_NOT_NULL;
        }

        try {
            *wrapper = static_cast<void*>(new _param:data_item_class{
                /* _link:instance_args */
            });
        } catch (const std::exception& exc) {
            std::cerr << exc.what() << std::endl;
            return MILHOJA_ERROR_UNABLE_TO_CREATE_WRAPPER;
        } catch (...) {
            std::cerr << "[_param:instance] Unknown error caught" << std::endl;
            return MILHOJA_ERROR_UNABLE_TO_CREATE_WRAPPER;
        }
        return MILHOJA_SUCCESS;
    }

    int _param:deletion(void* wrapper) {
        if (wrapper == nullptr) {
            std::cerr << "[_param:deletion] wrapper is NULL" << std::endl;
            return MILHOJA_ERROR_POINTER_IS_NULL;
        }
        delete static_cast<_param:data_item_class*>(wrapper);
        return MILHOJA_SUCCESS;
    }

    int _param:acquire_scratch_function(void) {
        _param:data_item_class::acquireScratch();
        return MILHOJA_SUCCESS;
    }

    int _param:release_scratch_function(void) {
        _param:data_item_class::releaseScratch();
        return MILHOJA_SUCCESS;
    }

    //----- C DECLARATION OF ACTUAL TASK FUNCTION TO PASS TO RUNTIME
    void  _param:cpp2c_function_name (
        const int threadIndex,
        milhoja::DataItem* dataItem
    ) {
        _param:data_item_class* wrapper = dynamic_cast<_param:data_item_class*>(dataItem);
        milhoja::Tile* tileDesc = wrapper->tile_.get();

        // tile_data includes any tile_metadata, tile_in / in_out / out
        /* _link:tile_data */

        // acquire scratch data
        /* _link:acquire_scratch */

        // consolidate tile arrays.
        /* _link:consolidate_tile_data */

        _param:c2f_function_name(
            /* _link:real_args */
        );
    }
}