#ifndef GRIDTEST_H__
#define GRIDTEST_H__

#if 0
Currently needed by DataPacket.h. Will be fixed after consultation with Jared.
#endif
#define NXB (1+7*K1D)
#define NYB (1+7*K2D)
#define NZB (1+7*K3D)
#define NGUARD 1

#if 0
NOTE: The data packet variable masking presently requires that
      the first variable be zero and that the indices be 
      the first NUNKVAR integers {0, ..., NUNKVAR-1}.
#endif

#define DENS_VAR_C 0
#define VELX_VAR_C 1
#define VELY_VAR_C 2
#define VELZ_VAR_C 3
#define PRES_VAR_C 4
#define ENER_VAR_C 5
#define TEMP_VAR_C 6
#define EINT_VAR_C 7
#define GAMC_VAR_C 8
#define GAME_VAR_C 9

#define NUNKVAR    10
#define UNK_VARS_BEGIN_C  DENS_VAR_C
#define UNK_VARS_END_C    GAME_VAR_C

#define NFLUXES    5
#define HY_DENS_FLUX_C  0
#define HY_XMOM_FLUX_C  1
#define HY_YMOM_FLUX_C  2
#define HY_ZMOM_FLUX_C  3
#define HY_ENER_FLUX_C  4

#define NMASS_SCALARS   0


namespace rp_Simulation {
    constexpr  unsigned int         N_THREADS_FOR_IC = 6;
    constexpr  unsigned int         N_DISTRIBUTOR_THREADS_FOR_IC = 1;
}
#endif
