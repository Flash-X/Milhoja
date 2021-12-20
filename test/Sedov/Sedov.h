#ifndef SEDOV_H__
#define SEDOV_H__

#define GLOBAL_COMM     MPI_COMM_WORLD
#define LEAD_RANK       0

#define NGUARD          1

#if 0
GAMC and GAME intentionally set as last two UNK variables.  See
Hydro_advanceSolutionHll_packet_oacc_summit_2 for more info.
#endif
#define NUNKVAR         10

#define DENS_VAR        0
#define VELX_VAR        1
#define VELY_VAR        2
#define VELZ_VAR        3
#define PRES_VAR        4
#define ENER_VAR        5
#define TEMP_VAR        6
#define EINT_VAR        7
#define GAMC_VAR        8
#define GAME_VAR        9

#define UNK_VARS_BEGIN  DENS_VAR
#define UNK_VARS_END    GAME_VAR

#define NFLUXES         5

#define HY_DENS_FLUX    0
#define HY_XMOM_FLUX    1
#define HY_YMOM_FLUX    2
#define HY_ZMOM_FLUX    3
#define HY_ENER_FLUX    4

#endif

