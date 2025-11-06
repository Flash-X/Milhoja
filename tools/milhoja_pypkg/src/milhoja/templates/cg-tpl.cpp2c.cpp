/* _connector:cpp2c */
#include <iostream>
#include <Milhoja.h>
#include <Milhoja_real.h>
#include <Milhoja_interface_error_codes.h>

#include "_param:file_name"

//#ifndef MILHOJA_OPENACC_OFFLOADING
//#error "This file should only be compiled if using OpenACC offloading"
//#endif

using milhoja::Real;

/* _link:omp_requires */

extern "C" {
    //----- C DECLARATION OF FORTRAN ROUTINE WITH C-COMPATIBLE INTERFACE
    void _param:taskfunctionnamec2f (
        /* _link:c2f_argument_list */
    );

    int _param:instantiate (
        /* _link:instance_args */
    ) {
        if ( packet == nullptr) {
            std::cerr << "[_param:instantiate] packet is NULL" << std::endl;
            return MILHOJA_ERROR_POINTER_IS_NULL;
        } else if (*packet != nullptr) {
            std::cerr << "[_param:instantiate] *packet not NULL" << std::endl;
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
            std::cerr << "[_param:instantiate] Unknown error caught" << std::endl;
            return MILHOJA_ERROR_UNABLE_TO_CREATE_PACKET;
        }

        return MILHOJA_SUCCESS;
    }

    int _param:deletion (void* packet) {
        if (packet == nullptr) {
            std::cerr << "[_param:deletion] packet is NULL" << std::endl;
            return MILHOJA_ERROR_POINTER_IS_NULL;
        }
        delete static_cast< _param:class_name *>(packet);
        return MILHOJA_SUCCESS;
    }

    /* _link:release_extra_queue */

    //----- C TASK FUNCTION TO BE CALLED BY RUNTIME
    void _param:taskfunctionnametf (const int _param:thread_id, void* dataItem_h) {
        _param:class_name* packet_h = static_cast<_param:class_name*>(dataItem_h);
        /* _link:get_host_members */

        /* _link:get_device_members */

        // Pass data packet info to C-to-Fortran Reinterpretation Layer
        _param:taskfunctionnamec2f (
            /* _link:c2f_arguments */
        );
    }
}
