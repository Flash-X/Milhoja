#ifndef FLASH_H__
#define FLASH_H__

#define NGUARD 1

#if 0
NOTE: The data packet variable masking presently requires that
      the first variable be zero and that the indices be 
      the first NUNKVAR integers {0, ..., NUNKVAR-1}.
#endif

#define DENS_VAR 0
#define VELX_VAR 1
#define VELY_VAR 2
#define VELZ_VAR 3
#define PRES_VAR 4
#define ENER_VAR 5
#define TEMP_VAR 6
#define EINT_VAR 7
#define GAMC_VAR 8
#define GAME_VAR 9

#define NUNKVAR    10
#define UNK_VARS_BEGIN  0
#define UNK_VARS_END    9

#define NFLUXES    5
#define HY_DENS_FLUX  0
#define HY_XMOM_FLUX  1
#define HY_YMOM_FLUX  2
#define HY_ZMOM_FLUX  3
#define HY_ENER_FLUX  4

#if 0
Assuming 2D only for now
#endif
#define K1D        1
#define K2D        0
#define K3D        0

#endif

