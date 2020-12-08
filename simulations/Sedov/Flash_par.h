#ifndef FLASH_PAR_H__
#define FLASH_PAR_H__

#include "Grid_REAL.h"
#include "Grid_IntVect.h"
#include "Grid_RealVect.h"

namespace rp_Driver {
    // This is how I fake a Driver_computeDt
    constexpr orchestration::Real   DT_AFTER            = 5.0e-5_wp;
    constexpr unsigned int          WRITE_EVERY_N_STEPS = 10;
}

namespace rp_Io {
    // computeIntegralQuantitiesByBlock run in CPU-only thread team
    // configuration using blocks
    constexpr unsigned int          N_THREADS_FOR_INT_QUANTITIES = 2;
}

namespace rp_Grid {
    constexpr orchestration::Real   X_MIN       = 0.0_wp;
    constexpr orchestration::Real   X_MAX       = 1.0_wp;
    constexpr orchestration::Real   Y_MIN       = 0.0_wp;
    constexpr orchestration::Real   Y_MAX       = 1.0_wp;
    constexpr orchestration::Real   Z_MIN       = 0.0_wp;
    constexpr orchestration::Real   Z_MAX       = 1.0_wp;

    constexpr unsigned int          LREFINE_MAX = 1;

    constexpr unsigned int          N_BLOCKS_X  = 32;
    constexpr unsigned int          N_BLOCKS_Y  = 32;
    constexpr unsigned int          N_BLOCKS_Z  = 1;
}

namespace rp_Runtime {
    constexpr unsigned int          N_THREADS_PER_TEAM     = 4;
    constexpr unsigned int          N_THREAD_TEAMS         = 1;
    constexpr int                   N_STREAMS              = 32; 
    constexpr std::size_t           MEMORY_POOL_SIZE_BYTES = 12884901888;
}

namespace rp_Eos {
    constexpr orchestration::Real   GAMMA = 1.4_wp;
}

namespace rp_Hydro {
    // advanceSolution run in CPU-only thread team
    // configuration using blocks
    constexpr unsigned int          N_THREADS_FOR_ADV_SOLN = 2;
}

namespace rp_Simulation {
    const std::string               NAME                         = "sedov";
    const std::string               LOG_FILENAME                 = NAME + ".log";
    const std::string               INTEGRAL_QUANTITIES_FILENAME = NAME + ".dat";

    constexpr  orchestration::Real  T_0       = 0.0_wp;
    constexpr  orchestration::Real  T_MAX     = 0.5_wp;
    constexpr  unsigned int         MAX_STEPS = 70;
    // When FLASH-X runs Sedov/2D with a dtInit that is too small, it sets
    // dtInit to one-tenth the CFL-limited dt value for Hydro, which is this
    // value.
    constexpr  orchestration::Real  DT_INIT   = 5.6922183414086268e-6_wp;

    // setInitialConditions run in CPU-only thread team
    // configuration using blocks
    constexpr  unsigned int         N_THREADS_FOR_IC = 4;

    constexpr unsigned int          N_PROFILE    = 10000;
    constexpr orchestration::Real   P_AMBIENT    = 1.0e-5_wp;
    constexpr orchestration::Real   RHO_AMBIENT  = 1.0_wp;
    constexpr orchestration::Real   EXP_ENERGY   = 1.0_wp;
    constexpr orchestration::Real   MIN_RHO_INIT = 1.0e-20_wp;
    constexpr orchestration::Real   R_INIT       = 0.013671875_wp;
    constexpr orchestration::Real   SMALL_RHO    = 1.0e-10_wp;
    constexpr orchestration::Real   SMALL_P      = 1.0e-10_wp;
    constexpr orchestration::Real   SMALL_T      = 1.0e-10_wp;
    constexpr orchestration::Real   SMALL_E      = 1.0e-10_wp;
    constexpr unsigned int          N_SUB_ZONES  = 7;
    constexpr orchestration::Real   X_CENTER     = 0.5_wp*(rp_Grid::X_MAX - rp_Grid::X_MIN);
    constexpr orchestration::Real   Y_CENTER     = 0.5_wp*(rp_Grid::Y_MAX - rp_Grid::Y_MIN);
    constexpr orchestration::Real   Z_CENTER     = 0.5_wp*(rp_Grid::Z_MAX - rp_Grid::Z_MIN);

    // Value from FLASH-X constants.h
    constexpr orchestration::Real   PI           = 3.1415926535897932384_wp;
#if NDIM == 1
    constexpr orchestration::Real   vctr         = 2.0_wp * R_INIT;
#elif NDIM == 2
    constexpr orchestration::Real   vctr         = PI * R_INIT*R_INIT;
#else
    constexpr orchestration::Real   vctr         = 4.0_wp / 3.0_wp * PI * R_INIT*R_INIT*R_INIT;
#endif
    constexpr orchestration::Real   P_EXP        = (rp_Eos::GAMMA - 1.0_wp) * EXP_ENERGY / vctr;
    constexpr orchestration::Real   IN_SUBZONES  = 1.0 / orchestration::Real(N_SUB_ZONES);
}

#endif

