#ifndef FLASH_H__
#define FLASH_H__

#define NGUARD 1

#if 0
NOTE: The data packet variable masking presently requires that
      the first variable be zero and that the indices be 
      the first NUNKVAR integers {0, ..., NUNKVAR-1}.
#endif
#define DENS_VAR 0
#define ENER_VAR 1
#define NUNKVAR    2
#define UNK_VARS_BEGIN    0
#define UNK_VARS_END      1

#define NFLUXES    0

#if 0
Assuming 2D only for now
#endif
#define K1D        1
#define K2D        1
#define K3D        0

#endif

