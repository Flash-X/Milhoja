#ifndef FLASH_PAR_H__
#define FLASH_PAR_H__

#include "Grid_REAL.h"

// NOTE: The ThreadTeam test does not use the Grid unit at all.  However, to
// keep the build system simple, all tests build in the Grid unit.  Therefore,
// we need these runtime parameters to be defined.  Using non-sensical values so
// that it should be obvious if these values are accidentally used at any time.
namespace rp_Grid {
    constexpr orchestration::Real   X_MIN       =  0.0_wp;
    constexpr orchestration::Real   X_MAX       = -1.0_wp;
    constexpr orchestration::Real   Y_MIN       =  0.0_wp;
    constexpr orchestration::Real   Y_MAX       = -1.0_wp;
    constexpr orchestration::Real   Z_MIN       =  0.0_wp;
    constexpr orchestration::Real   Z_MAX       = -1.0_wp;

    constexpr unsigned int          LREFINE_MAX = 0;

    constexpr unsigned int          N_BLOCKS_X  = 0;
    constexpr unsigned int          N_BLOCKS_Y  = 0;
    constexpr unsigned int          N_BLOCKS_Z  = 0;
}

namespace rp_Simulation {
    constexpr  unsigned int         N_DISTRIBUTOR_THREADS_FOR_IC = 0;
    constexpr  unsigned int         N_THREADS_FOR_IC             = 0;
}

#endif

