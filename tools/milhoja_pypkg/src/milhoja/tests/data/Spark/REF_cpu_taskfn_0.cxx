
#include <iostream>
#include <Milhoja.h>
#include <Milhoja_real.h>
#include <Milhoja_IntVect.h>
#include <Milhoja_RealVect.h>
#include <Milhoja_Tile.h>
#include <Milhoja_interface_error_codes.h>

#include "TileWrapper_cpu_taskfn_0.h"

using real = milhoja::Real;

extern "C" {
    //----- C DECLARATION OF FORTRAN ROUTINE WITH C-COMPATIBLE INTERFACE
    void cpu_taskfn_0_C2F (
    const milhoja::Real external_Hydro_dt,
    const milhoja::Real external_Hydro_dtOld,
    const int external_Hydro_stage,
    const void* tile_arrayBounds,
    const void* tile_deltas_array,
    const void* tile_interior,
    const void* tile_lbound_array,
    const void* tile_lo_array,
    const void* CC_1,
    const void* FLX_1,
    const void* FLY_1,
    const void* scratch_Hydro_cvol_fake,
    const void* scratch_Hydro_fareaX_fake,
    const void* scratch_Hydro_fareaY_fake,
    const void* scratch_Hydro_fareaZ_fake,
    const void* scratch_Hydro_fluxBufZ,
    const void* scratch_Hydro_hy_Vc,
    const void* scratch_Hydro_hy_flat3d,
    const void* scratch_Hydro_hy_flux,
    const void* scratch_Hydro_hy_flx,
    const void* scratch_Hydro_hy_fly,
    const void* scratch_Hydro_hy_flz,
    const void* scratch_Hydro_hy_grav,
    const void* scratch_Hydro_hy_rope,
    const void* scratch_Hydro_hy_starState,
    const void* scratch_Hydro_hy_tmpState,
    const void* scratch_Hydro_hy_uMinus,
    const void* scratch_Hydro_hy_uPlus,
    const void* scratch_Hydro_xCenter_fake,
    const void* scratch_Hydro_xLeft_fake,
    const void* scratch_Hydro_xRight_fake,
    const void* scratch_Hydro_yCenter_fake,
    const void* scratch_Hydro_yLeft_fake,
    const void* scratch_Hydro_yRight_fake,
    const void* scratch_Hydro_zCenter_fake
    );

    int instantiate_cpu_taskfn_0_wrapper_c(
    const milhoja::Real external_Hydro_dt,
    const milhoja::Real external_Hydro_dtOld,
    const int external_Hydro_stage,
        void** wrapper
    ) {
        if (wrapper == nullptr) {
            std::cerr << "[instantiate_cpu_taskfn_0_wrapper_c] wrapper is NULL" << std::endl;
            return MILHOJA_ERROR_POINTER_IS_NULL;
        } else if (*wrapper != nullptr) {
            std::cerr << "[instantiate_cpu_taskfn_0_wrapper_c] *wrapper not NULL" << std::endl;
            return MILHOJA_ERROR_POINTER_NOT_NULL;
        }

        try {
            *wrapper = static_cast<void*>(new Tile_cpu_taskfn_0{
                external_Hydro_dt,
                external_Hydro_dtOld,
                external_Hydro_stage
            });
        } catch (const std::exception& exc) {
            std::cerr << exc.what() << std::endl;
            return MILHOJA_ERROR_UNABLE_TO_CREATE_WRAPPER;
        } catch (...) {
            std::cerr << "[instantiate_cpu_taskfn_0_wrapper_c] Unknown error caught" << std::endl;
            return MILHOJA_ERROR_UNABLE_TO_CREATE_WRAPPER;
        }
        return MILHOJA_SUCCESS;
    }

    int delete_cpu_taskfn_0_wrapper_c(void* wrapper) {
        if (wrapper == nullptr) {
            std::cerr << "[delete_cpu_taskfn_0_wrapper_c] wrapper is NULL" << std::endl;
            return MILHOJA_ERROR_POINTER_IS_NULL;
        }
        delete static_cast<Tile_cpu_taskfn_0*>(wrapper);
        return MILHOJA_SUCCESS;
    }

    int acquire_scratch_cpu_taskfn_0_wrapper_c(void) {
        Tile_cpu_taskfn_0::acquireScratch();
        return MILHOJA_SUCCESS;
    }

    int release_scratch_cpu_taskfn_0_wrapper_c(void) {
        Tile_cpu_taskfn_0::releaseScratch();
        return MILHOJA_SUCCESS;
    }

    //----- C DECLARATION OF ACTUAL TASK FUNCTION TO PASS TO RUNTIME
    void  cpu_taskfn_0_Cpp2C (
        const int threadIndex,
        milhoja::DataItem* dataItem
    ) {
        Tile_cpu_taskfn_0* wrapper = dynamic_cast<Tile_cpu_taskfn_0*>(dataItem);
        milhoja::Tile* tileDesc = wrapper->tile_.get();

        // tile_data includes any tile_metadata, tile_in / in_out / out
        const auto tile_deltas = tileDesc->deltas();
        const auto tile_lbound = tileDesc->loGC();
        const auto tile_lo = tileDesc->lo();
        real* CC_1 = tileDesc->dataPtr();
        real* FLX_1 = tileDesc->fluxDataPtrs()[0];
        real* FLY_1 = tileDesc->fluxDataPtrs()[1];

        // acquire scratch data
        milhoja::Real* scratch_Hydro_cvol_fake = static_cast<milhoja::Real*>(Tile_cpu_taskfn_0::scratch_Hydro_cvol_fake_) + Tile_cpu_taskfn_0::SCRATCH_HYDRO_CVOL_FAKE_SIZE_ * threadIndex;
        milhoja::Real* scratch_Hydro_fareaX_fake = static_cast<milhoja::Real*>(Tile_cpu_taskfn_0::scratch_Hydro_fareaX_fake_) + Tile_cpu_taskfn_0::SCRATCH_HYDRO_FAREAX_FAKE_SIZE_ * threadIndex;
        milhoja::Real* scratch_Hydro_fareaY_fake = static_cast<milhoja::Real*>(Tile_cpu_taskfn_0::scratch_Hydro_fareaY_fake_) + Tile_cpu_taskfn_0::SCRATCH_HYDRO_FAREAY_FAKE_SIZE_ * threadIndex;
        milhoja::Real* scratch_Hydro_fareaZ_fake = static_cast<milhoja::Real*>(Tile_cpu_taskfn_0::scratch_Hydro_fareaZ_fake_) + Tile_cpu_taskfn_0::SCRATCH_HYDRO_FAREAZ_FAKE_SIZE_ * threadIndex;
        milhoja::Real* scratch_Hydro_fluxBufZ = static_cast<milhoja::Real*>(Tile_cpu_taskfn_0::scratch_Hydro_fluxBufZ_) + Tile_cpu_taskfn_0::SCRATCH_HYDRO_FLUXBUFZ_SIZE_ * threadIndex;
        milhoja::Real* scratch_Hydro_hy_Vc = static_cast<milhoja::Real*>(Tile_cpu_taskfn_0::scratch_Hydro_hy_Vc_) + Tile_cpu_taskfn_0::SCRATCH_HYDRO_HY_VC_SIZE_ * threadIndex;
        milhoja::Real* scratch_Hydro_hy_flat3d = static_cast<milhoja::Real*>(Tile_cpu_taskfn_0::scratch_Hydro_hy_flat3d_) + Tile_cpu_taskfn_0::SCRATCH_HYDRO_HY_FLAT3D_SIZE_ * threadIndex;
        milhoja::Real* scratch_Hydro_hy_flux = static_cast<milhoja::Real*>(Tile_cpu_taskfn_0::scratch_Hydro_hy_flux_) + Tile_cpu_taskfn_0::SCRATCH_HYDRO_HY_FLUX_SIZE_ * threadIndex;
        milhoja::Real* scratch_Hydro_hy_flx = static_cast<milhoja::Real*>(Tile_cpu_taskfn_0::scratch_Hydro_hy_flx_) + Tile_cpu_taskfn_0::SCRATCH_HYDRO_HY_FLX_SIZE_ * threadIndex;
        milhoja::Real* scratch_Hydro_hy_fly = static_cast<milhoja::Real*>(Tile_cpu_taskfn_0::scratch_Hydro_hy_fly_) + Tile_cpu_taskfn_0::SCRATCH_HYDRO_HY_FLY_SIZE_ * threadIndex;
        milhoja::Real* scratch_Hydro_hy_flz = static_cast<milhoja::Real*>(Tile_cpu_taskfn_0::scratch_Hydro_hy_flz_) + Tile_cpu_taskfn_0::SCRATCH_HYDRO_HY_FLZ_SIZE_ * threadIndex;
        milhoja::Real* scratch_Hydro_hy_grav = static_cast<milhoja::Real*>(Tile_cpu_taskfn_0::scratch_Hydro_hy_grav_) + Tile_cpu_taskfn_0::SCRATCH_HYDRO_HY_GRAV_SIZE_ * threadIndex;
        milhoja::Real* scratch_Hydro_hy_rope = static_cast<milhoja::Real*>(Tile_cpu_taskfn_0::scratch_Hydro_hy_rope_) + Tile_cpu_taskfn_0::SCRATCH_HYDRO_HY_ROPE_SIZE_ * threadIndex;
        milhoja::Real* scratch_Hydro_hy_starState = static_cast<milhoja::Real*>(Tile_cpu_taskfn_0::scratch_Hydro_hy_starState_) + Tile_cpu_taskfn_0::SCRATCH_HYDRO_HY_STARSTATE_SIZE_ * threadIndex;
        milhoja::Real* scratch_Hydro_hy_tmpState = static_cast<milhoja::Real*>(Tile_cpu_taskfn_0::scratch_Hydro_hy_tmpState_) + Tile_cpu_taskfn_0::SCRATCH_HYDRO_HY_TMPSTATE_SIZE_ * threadIndex;
        milhoja::Real* scratch_Hydro_hy_uMinus = static_cast<milhoja::Real*>(Tile_cpu_taskfn_0::scratch_Hydro_hy_uMinus_) + Tile_cpu_taskfn_0::SCRATCH_HYDRO_HY_UMINUS_SIZE_ * threadIndex;
        milhoja::Real* scratch_Hydro_hy_uPlus = static_cast<milhoja::Real*>(Tile_cpu_taskfn_0::scratch_Hydro_hy_uPlus_) + Tile_cpu_taskfn_0::SCRATCH_HYDRO_HY_UPLUS_SIZE_ * threadIndex;
        milhoja::Real* scratch_Hydro_xCenter_fake = static_cast<milhoja::Real*>(Tile_cpu_taskfn_0::scratch_Hydro_xCenter_fake_) + Tile_cpu_taskfn_0::SCRATCH_HYDRO_XCENTER_FAKE_SIZE_ * threadIndex;
        milhoja::Real* scratch_Hydro_xLeft_fake = static_cast<milhoja::Real*>(Tile_cpu_taskfn_0::scratch_Hydro_xLeft_fake_) + Tile_cpu_taskfn_0::SCRATCH_HYDRO_XLEFT_FAKE_SIZE_ * threadIndex;
        milhoja::Real* scratch_Hydro_xRight_fake = static_cast<milhoja::Real*>(Tile_cpu_taskfn_0::scratch_Hydro_xRight_fake_) + Tile_cpu_taskfn_0::SCRATCH_HYDRO_XRIGHT_FAKE_SIZE_ * threadIndex;
        milhoja::Real* scratch_Hydro_yCenter_fake = static_cast<milhoja::Real*>(Tile_cpu_taskfn_0::scratch_Hydro_yCenter_fake_) + Tile_cpu_taskfn_0::SCRATCH_HYDRO_YCENTER_FAKE_SIZE_ * threadIndex;
        milhoja::Real* scratch_Hydro_yLeft_fake = static_cast<milhoja::Real*>(Tile_cpu_taskfn_0::scratch_Hydro_yLeft_fake_) + Tile_cpu_taskfn_0::SCRATCH_HYDRO_YLEFT_FAKE_SIZE_ * threadIndex;
        milhoja::Real* scratch_Hydro_yRight_fake = static_cast<milhoja::Real*>(Tile_cpu_taskfn_0::scratch_Hydro_yRight_fake_) + Tile_cpu_taskfn_0::SCRATCH_HYDRO_YRIGHT_FAKE_SIZE_ * threadIndex;
        milhoja::Real* scratch_Hydro_zCenter_fake = static_cast<milhoja::Real*>(Tile_cpu_taskfn_0::scratch_Hydro_zCenter_fake_) + Tile_cpu_taskfn_0::SCRATCH_HYDRO_ZCENTER_FAKE_SIZE_ * threadIndex;

        // consolidate tile arrays.
        int tile_arrayBounds[] = {
            tileDesc->loGC().I(),tileDesc->hiGC().I(),
            tileDesc->loGC().J(),tileDesc->hiGC().J(),
            tileDesc->loGC().K(),tileDesc->hiGC().K()
        };
        real tile_deltas_array[] = {
            tile_deltas.I(),
            tile_deltas.J(),
            tile_deltas.K()
        };
        int tile_interior[] = {
            tileDesc->lo().I(),tileDesc->hi().I(),
            tileDesc->lo().J(),tileDesc->hi().J(),
            tileDesc->lo().K(),tileDesc->hi().K()
        };
        int tile_lbound_array[] = {
            tile_lbound.I(),
            tile_lbound.J(),
            tile_lbound.K()
        };
        int tile_lo_array[] = {
            tile_lo.I(),
            tile_lo.J(),
            tile_lo.K()
        };

        cpu_taskfn_0_C2F(
        wrapper->external_Hydro_dt_,
        wrapper->external_Hydro_dtOld_,
        wrapper->external_Hydro_stage_,
        static_cast<void*>(tile_arrayBounds),
        static_cast<void*>(tile_deltas_array),
        static_cast<void*>(tile_interior),
        static_cast<void*>(tile_lbound_array),
        static_cast<void*>(tile_lo_array),
        static_cast<void*>(CC_1),
        static_cast<void*>(FLX_1),
        static_cast<void*>(FLY_1),
        static_cast<void*>(scratch_Hydro_cvol_fake),
        static_cast<void*>(scratch_Hydro_fareaX_fake),
        static_cast<void*>(scratch_Hydro_fareaY_fake),
        static_cast<void*>(scratch_Hydro_fareaZ_fake),
        static_cast<void*>(scratch_Hydro_fluxBufZ),
        static_cast<void*>(scratch_Hydro_hy_Vc),
        static_cast<void*>(scratch_Hydro_hy_flat3d),
        static_cast<void*>(scratch_Hydro_hy_flux),
        static_cast<void*>(scratch_Hydro_hy_flx),
        static_cast<void*>(scratch_Hydro_hy_fly),
        static_cast<void*>(scratch_Hydro_hy_flz),
        static_cast<void*>(scratch_Hydro_hy_grav),
        static_cast<void*>(scratch_Hydro_hy_rope),
        static_cast<void*>(scratch_Hydro_hy_starState),
        static_cast<void*>(scratch_Hydro_hy_tmpState),
        static_cast<void*>(scratch_Hydro_hy_uMinus),
        static_cast<void*>(scratch_Hydro_hy_uPlus),
        static_cast<void*>(scratch_Hydro_xCenter_fake),
        static_cast<void*>(scratch_Hydro_xLeft_fake),
        static_cast<void*>(scratch_Hydro_xRight_fake),
        static_cast<void*>(scratch_Hydro_yCenter_fake),
        static_cast<void*>(scratch_Hydro_yLeft_fake),
        static_cast<void*>(scratch_Hydro_yRight_fake),
        static_cast<void*>(scratch_Hydro_zCenter_fake)
        );
    }
}

