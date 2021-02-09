#ifndef FLASH_H__
#define FLASH_H__

#define NGUARD 1

#if 0
GAMC and GAME intentionally set as last two UNK variables.  See
Hydro_advanceSolutionHll_packet_oacc_summit_2 for more info.
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

#define REAL_IS_DOUBLE
#define ORCH_REAL    MPI_DOUBLE_PRECISION

#if 0
Assuming 2D only for now
#endif
#define K1D        1
#define K2D        1
#define K3D        0

#endif

