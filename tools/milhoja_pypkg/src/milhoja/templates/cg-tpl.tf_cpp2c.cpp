/* _connector:tf_cpp2c */
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
#include <Milhoja_IntVect.h>
#include <Milhoja_RealVect.h>
#include <Milhoja_Tile.h>
#include <Milhoja_interface_error_codes.h>

#include "_param:data_item_header_file_name.h"

extern "C" {
    //----- C DECLARATION OF FORTRAN ROUTINE WITH C-COMPATIBLE INTERFACE
    void _param:c2f_function_name (
        /* _link:c2f_dummy_args */
    );

    //----- C DECLARATION OF ACTUAL TASK FUNCTION TO PASS TO RUNTIME
    void  _param:cpp2c_function_name (
        const int threadIndex,
        milhoja::DataItem* dataItem
    ) {
        _param:data_item_class*  wrapper = dynamic_cast<_param:data_item_class*>(dataItem);
        milhoja::Tile*      tileDesc = wrapper->tile_.get();

        // tile_data includes any tile_metadata, tile_in / in_out / out
        /* _link:tile_data */

        // acquire scratch data
        /* _link:acquire_scratch */

        // consolidate tile arrays.
        /* _link:consolidate_tile_data */

        _param:c2f_function_name(
            /* _link:real_args */
        ));
    }
}