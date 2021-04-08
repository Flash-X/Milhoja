#ifndef FLASH_PAR_H__
#define FLASH_PAR_H__

#include "Grid_REAL.h"

namespace rp_Grid {
    constexpr orchestration::Real   X_MIN       = 0.0_wp;
    constexpr orchestration::Real   X_MAX       = 1.0_wp;
    constexpr orchestration::Real   Y_MIN       = 0.0_wp;
    constexpr orchestration::Real   Y_MAX       = 1.0_wp;
    constexpr orchestration::Real   Z_MIN       = 0.0_wp;
    constexpr orchestration::Real   Z_MAX       = 1.0_wp;

    constexpr unsigned int          LREFINE_MAX = 1;

    constexpr unsigned int          N_BLOCKS_X  = 256;
    constexpr unsigned int          N_BLOCKS_Y  = 128;
    constexpr unsigned int          N_BLOCKS_Z  = 1;
}

namespace rp_Simulation {
    constexpr  unsigned int         N_DISTRIBUTOR_THREADS_FOR_IC = 1;
    constexpr  unsigned int         N_THREADS_FOR_IC             = 4;
}

#endif

