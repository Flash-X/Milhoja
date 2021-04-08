#ifndef FLASH_H__
#define FLASH_H__

#define NGUARD 1

#define TEMP_VAR_C 0
#define RHST_VAR_C 1

#define NUNKVAR    2
#define UNK_VARS_BEGIN_C  TEMP_VAR_C
#define UNK_VARS_END_C    RHST_VAR_C

#define NFLUXES    0

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
