#ifndef FLASH_PAR_H__
#define FLASH_PAR_H__

#include <Milhoja_real.h>

namespace rp_Grid {
    constexpr milhoja::Real         X_MIN       = 0.0_wp;
    constexpr milhoja::Real         X_MAX       = 1.0_wp;
    constexpr milhoja::Real         Y_MIN       = 0.0_wp;
    constexpr milhoja::Real         Y_MAX       = 1.0_wp;
    constexpr milhoja::Real         Z_MIN       = 0.0_wp;
    constexpr milhoja::Real         Z_MAX       = 1.0_wp;

    constexpr unsigned int          LREFINE_MAX = 1;

    constexpr unsigned int          NXB = 8;
    constexpr unsigned int          NYB = 16;
    constexpr unsigned int          NZB = 1;

    constexpr unsigned int          N_BLOCKS_X  = 256;
    constexpr unsigned int          N_BLOCKS_Y  = 128;
    constexpr unsigned int          N_BLOCKS_Z  = 1;
}

namespace rp_Simulation {
    // setInitialConditions run in CPU-only thread team
    // configuration using blocks
    constexpr unsigned int          N_DISTRIBUTOR_THREADS_FOR_IC = 2;
    constexpr unsigned int          N_THREADS_FOR_IC = 4;
}

#endif

