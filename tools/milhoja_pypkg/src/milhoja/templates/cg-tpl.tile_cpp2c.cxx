/**
 * @copyright Copyright 2022 UChicago Argonne, LLC and contributors
 *
 * @licenseblock
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * @endlicenseblock
 *
 * @file
 */

#include <iostream>

#include <Milhoja.h>
#include <Milhoja_real.h>
#include <Milhoja_interface_error_codes.h>
#include "_param:data_item_header_file_name.h"

extern "C" {
    //----- C DECLARATION OF FORTRAN ROUTINE WITH C-COMPATIBLE INTERFACE
    int _param:instance_function (
        /* _link:constructor_dummy_arguments */
        void** wrapper
    ) {
        if ( wrapper == nullptr) {
            std::cerr << "[_param:instance_function] wrapper is NULL" << std::endl;
            return MILHOJA_ERROR_POINTER_IS_NULL;
        } else if (*wrapper != nullptr) {
            std::cerr << "[_param:instance_function] *wrapper not NULL" << std::endl;
            return MILHOJA_ERROR_POINTER_NOT_NULL;
        }

        try {
            *wrapper = static_cast<void*>(new _param:data_item_class_name{
                /* _link:constructor_dummy_arguments */
            });
        } catch (const std::exception& exc) {
            std::cerr << exc.what() << std::endl;
            return MILHOJA_ERROR_UNABLE_TO_CREATE_WRAPPER;
        } catch (...) {
            std::cerr << "[_param:instance_function] Unknown error caught" << std::endl;
            return MILHOJA_ERROR_UNABLE_TO_CREATE_WRAPPER;
        }

        return MILHOJA_SUCCESS;
    }

    int _param:delete_function (void* wrapper) {
        if (wrapper == nullptr) {
            std::cerr << "[_param:delete_function] wrapper is NULL" << std::endl;
            return MILHOJA_ERROR_POINTER_IS_NULL;
        }
        delete static_cast<_param:data_item_class_name *>(wrapper);

        return MILHOJA_SUCCESS;
    }

    int _param:acquire_scratch_function (void) {
        _param:data_item_class_name::acquireScratch();

        return MILHOJA_SUCCESS;
    }

    int _param:release_scratch_function (void) {
        _param:data_item_class_name::releaseScratch();

        return MILHOJA_SUCCESS;
    }
}