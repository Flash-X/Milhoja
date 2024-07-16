#include <iostream>
#include <Milhoja.h>
#include <Milhoja_real.h>
#include <Milhoja_interface_error_codes.h>
#include "_param:data_item_header_file_name"

extern "C" {
    //----- C DECLARATION OF FORTRAN ROUTINE WITH C-COMPATIBLE INTERFACE
    int _param:instance(
        /* _link:c2f_dummy_args */
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
            *wrapper = static_cast<void*>(new _param:class_name{
                /* _link:constructor_args */
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
        delete static_cast<_param:class_name*>(wrapper);
        return MILHOJA_SUCCESS;
    }

    int _param:acquire_scratch(void) {
        _param:class_name::acquireScratch();
        return MILHOJA_SUCCESS;
    }

    int _param:release_scratch(void) {
        _param:class_name::releaseScratch();
        return MILHOJA_SUCCESS;
    }
}