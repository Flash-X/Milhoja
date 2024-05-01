#include <iostream>
#include <Milhoja.h>
#include <Milhoja_real.h>
#include <Milhoja_IntVect.h>
#include <Milhoja_RealVect.h>
#include <Milhoja_Tile.h>
#include <Milhoja_interface_error_codes.h>

#include "Tile_cpu_tf_hydro.h"

using real = milhoja::Real;

extern "C" {
    //----- C DECLARATION OF FORTRAN ROUTINE WITH C-COMPATIBLE INTERFACE
    void cpu_tf_hydro_c2f (
        const milhoja::Real external_hydro_op1_dt,
        const int external_hydro_op1_eosMode,
        const void* tile_deltas_array,
        const void* tile_hi_array,
        const void* tile_interior,
        const void* tile_lo_array,
        const void* CC_1,
        const void* scratch_hydro_op1_auxC,
        const void* scratch_hydro_op1_flX,
        const void* scratch_hydro_op1_flY,
        const void* scratch_hydro_op1_flZ,
        const void* lbdd_CC_1,
        const void* lbdd_scratch_hydro_op1_auxC,
        const void* lbdd_scratch_hydro_op1_flX,
        const void* lbdd_scratch_hydro_op1_flY,
        const void* lbdd_scratch_hydro_op1_flZ
    );

    int instantiate_cpu_tf_hydro_wrapper_c(
        const milhoja::Real external_hydro_op1_dt,
        const int external_hydro_op1_eosMode,
        void** wrapper
    ) {
        if (wrapper == nullptr) {
            std::cerr << "[instantiate_cpu_tf_hydro_wrapper_c] wrapper is NULL" << std::endl;
            return MILHOJA_ERROR_POINTER_IS_NULL;
        } else if (*wrapper != nullptr) {
            std::cerr << "[instantiate_cpu_tf_hydro_wrapper_c] *wrapper not NULL" << std::endl;
            return MILHOJA_ERROR_POINTER_NOT_NULL;
        }

        try {
            *wrapper = static_cast<void*>(new Tile_cpu_tf_hydro{
                external_hydro_op1_dt,
                external_hydro_op1_eosMode
            });
        } catch (const std::exception& exc) {
            std::cerr << exc.what() << std::endl;
            return MILHOJA_ERROR_UNABLE_TO_CREATE_WRAPPER;
        } catch (...) {
            std::cerr << "[instantiate_cpu_tf_hydro_wrapper_c] Unknown error caught" << std::endl;
            return MILHOJA_ERROR_UNABLE_TO_CREATE_WRAPPER;
        }
        return MILHOJA_SUCCESS;
    }

    int delete_cpu_tf_hydro_wrapper_c(void* wrapper) {
        if (wrapper == nullptr) {
            std::cerr << "[delete_cpu_tf_hydro_wrapper_c] wrapper is NULL" << std::endl;
            return MILHOJA_ERROR_POINTER_IS_NULL;
        }
        delete static_cast<Tile_cpu_tf_hydro*>(wrapper);
        return MILHOJA_SUCCESS;
    }

    int acquire_scratch_cpu_tf_hydro_wrapper_c(void) {
        Tile_cpu_tf_hydro::acquireScratch();
        return MILHOJA_SUCCESS;
    }

    int release_scratch_cpu_tf_hydro_wrapper_c(void) {
        Tile_cpu_tf_hydro::releaseScratch();
        return MILHOJA_SUCCESS;
    }

    //----- C DECLARATION OF ACTUAL TASK FUNCTION TO PASS TO RUNTIME
    void  cpu_tf_hydro_cpp2c (
       const int threadIndex,
       milhoja::DataItem* dataItem
) {
       Tile_cpu_tf_hydro* wrapper = dynamic_cast<Tile_cpu_tf_hydro*>(dataItem);
       milhoja::Tile* tileDesc = wrapper->tile_.get();

       // tile_data includes any tile_metadata, tile_in / in_out / out
       const auto tile_deltas = tileDesc->deltas();
       const auto tile_hi = tileDesc->hi();
       const auto tile_lo = tileDesc->lo();
       real* CC_1 = tileDesc->dataPtr();
       const auto tile_lbound = tileDesc->loGC();

       // acquire scratch data
       milhoja::Real* scratch_hydro_op1_auxC = static_cast<milhoja::Real*>(Tile_cpu_tf_hydro::scratch_hydro_op1_auxc_) + Tile_cpu_tf_hydro::SCRATCH_HYDRO_OP1_AUXC_SIZE_ * threadIndex;
       milhoja::Real* scratch_hydro_op1_flX = static_cast<milhoja::Real*>(Tile_cpu_tf_hydro::scratch_hydro_op1_flx_) + Tile_cpu_tf_hydro::SCRATCH_HYDRO_OP1_FLX_SIZE_ * threadIndex;
       milhoja::Real* scratch_hydro_op1_flY = static_cast<milhoja::Real*>(Tile_cpu_tf_hydro::scratch_hydro_op1_fly_) + Tile_cpu_tf_hydro::SCRATCH_HYDRO_OP1_FLY_SIZE_ * threadIndex;
       milhoja::Real* scratch_hydro_op1_flZ = static_cast<milhoja::Real*>(Tile_cpu_tf_hydro::scratch_hydro_op1_flz_) + Tile_cpu_tf_hydro::SCRATCH_HYDRO_OP1_FLZ_SIZE_ * threadIndex;

       // consolidate tile arrays.
       real tile_deltas_array[] = {
              tile_deltas.I(),
              tile_deltas.J(),
              tile_deltas.K()
       };
       int tile_hi_array[] = {
              tile_hi.I(),
              tile_hi.J(),
              tile_hi.K()
       };
       int tile_interior[] = {
              tileDesc->lo().I(),tileDesc->hi().I(),
              tileDesc->lo().J(),tileDesc->hi().J(),
              tileDesc->lo().K(),tileDesc->hi().K()
       };
       int tile_lo_array[] = {
              tile_lo.I(),
              tile_lo.J(),
              tile_lo.K()
       };
       int lbdd_CC_1[] = {
              tile_lbound.I(),
              tile_lbound.J(),
              tile_lbound.K(),
              1
       };
       int lbdd_scratch_hydro_op1_auxC[] = {
              tile_lo.I()-1,
              tile_lo.J()-1,
              tile_lo.K()-1
       };
       int lbdd_scratch_hydro_op1_flX[] = {
              tile_lo.I(),
              tile_lo.J(),
              tile_lo.K(),
              1
       };
        int lbdd_scratch_hydro_op1_flY[] = {
              tile_lo.I(),
              tile_lo.J(),
              tile_lo.K(),
              1
       };
        int lbdd_scratch_hydro_op1_flZ[] = {
              tile_lo.I(),
              tile_lo.J(),
              tile_lo.K(),
              1
       };

       cpu_tf_hydro_c2f(
       wrapper->external_hydro_op1_dt_,
       wrapper->external_hydro_op1_eosMode_,
       static_cast<void*>(tile_deltas_array),
       static_cast<void*>(tile_hi_array),
       static_cast<void*>(tile_interior),
       static_cast<void*>(tile_lo_array),
       static_cast<void*>(CC_1),
       static_cast<void*>(scratch_hydro_op1_auxC),
       static_cast<void*>(scratch_hydro_op1_flX),
       static_cast<void*>(scratch_hydro_op1_flY),
       static_cast<void*>(scratch_hydro_op1_flZ),
       static_cast<void*>(lbdd_CC_1),
       static_cast<void*>(lbdd_scratch_hydro_op1_auxC),
       static_cast<void*>(lbdd_scratch_hydro_op1_flX),
       static_cast<void*>(lbdd_scratch_hydro_op1_flY),
       static_cast<void*>(lbdd_scratch_hydro_op1_flZ)
       );
    }
}