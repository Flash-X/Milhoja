#ifndef FLASH_H__
#define FLASH_H__

#define NGUARD 1

#if 0
NOTE: The data packet variable masking presently requires that
      the first variable be zero and that the indices be 
      the first NUNKVAR integers {0, ..., NUNKVAR-1}.
#endif
#define DENS_VAR_C 0
#define ENER_VAR_C 1
#define DENS_VAR   1
#define ENER_VAR   2
#define NUNKVAR    2
#define UNK_VARS_BEGIN_C  DENS_VAR_C
#define UNK_VARS_END_C    ENER_VAR_C
#define UNK_VARS_BEGIN    DENS_VAR
#define UNK_VARS_END      ENER_VAR

#define X_MIN      0.0
#define X_MAX      1.0
#define Y_MIN      0.0
#define Y_MAX      1.0
#define Z_MIN      0.0
#define Z_MAX      0.0
#define N_BLOCKS_X N_BLOCKS_ALONG_X
#define N_BLOCKS_Y N_BLOCKS_ALONG_Y
#define N_BLOCKS_Z N_BLOCKS_ALONG_Z

#define LREFINE_MAX REFINEMENT_LEVELS

#define REAL_IS_DOUBLE

#if 0
Assuming 2D only for now
#endif
#define K1D        1
#define K2D        1
#define K3D        0

#endif

