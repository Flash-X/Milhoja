/* _connector:cpp2c */

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

#include <Milhoja.h>
#include <Milhoja_real.h>
#include <Milhoja_interface_error_codes.h>

// include the datapacket. this may need to change depending on how the name is determined. for now,
// cgkit outputs will be named cgkit.datapacket.h.
#include "cgkit.datapacket.h"

#ifndef MILHOJA_OPENACC_OFFLOADING
#error "This file should only be compiled if using OpenACC offloading"
#endif

using milhoja::Real;

extern "C" {
    //----- C DECLARATION OF FORTRAN ROUTINE WITH C-COMPATIBLE INTERFACE
    void dr_hydro_advance_packet_oacc_c2f(
        /* _link:c2f_argument_list */
    );

    int instantiate_hydro_advance_packet_c(
        /* _link:instance_args */
        ) {
        if ( packet == nullptr) {
            std::cerr << "[instantiate_hydro_advance_packet_c] packet is NULL" << std::endl;
            return MILHOJA_ERROR_POINTER_IS_NULL;
        } else if (*packet != nullptr) {
            std::cerr << "[instantiate_hydro_advance_packet_c] *packet not NULL" << std::endl;
            return MILHOJA_ERROR_POINTER_NOT_NULL;
        }

        try {
            *packet = static_cast<void*>(new _param:class_name(
                /* _link:host_members */
            ));
        } catch (const std::exception& exc) {
            std::cerr << exc.what() << std::endl;
            return MILHOJA_ERROR_UNABLE_TO_CREATE_PACKET;
        } catch (...) {
            std::cerr << "[instantiate_hydro_advance_packet_c] Unknown error caught" << std::endl;
            return MILHOJA_ERROR_UNABLE_TO_CREATE_PACKET;
        }

        return MILHOJA_SUCCESS;
    }

    int delete_hydro_advance_packet_c(void* packet) {
        if (packet == nullptr) {
            std::cerr << "[delete_hydro_advance_packet_c] packet is NULL" << std::endl;
            return MILHOJA_ERROR_POINTER_IS_NULL;
        }
        delete static_cast< _param:class_name *>(packet);
        return MILHOJA_SUCCESS;
    }

    /* _link:release_extra_queue */

    //----- C TASK FUNCTION TO BE CALLED BY RUNTIME
    void dr_hydro_advance_packet_oacc_tf(const int tId, void* dataItem_h) {
        _param:class_name* packet_h = static_cast<_param:class_name*>(dataItem_h);
        /* _link:get_host_members */

        /* _link:get_device_members */

        // Pass data packet info to C-to-Fortran Reinterpretation Layer
        dr_hydro_advance_packet_oacc_c2f(
            /* _link:c2f_arguments */
        );
    }
}
