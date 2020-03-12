#ifndef CONSTANTS_H__
#define CONSTANTS_H__

/**
 * The contents of this file are meant for inclusion in Fortran code by the
 * pre-processor and is a mix of constants.h and Flash.h
 */

#define NGUARD 1
#define NXB 8
#define NYB 16
#define NZB 1

#define MDIM 3
#define NDIM 2

#define LOW 1
#define HIGH 2

#define IAXIS 1
#define JAXIS 2
#define KAXIS 3

// TODO: This two are presently set for C++, but should eventually be set for
// Fortran
#define DENS_VAR 0
#define ENER_VAR 1
#define NUNKVAR 2

#define X_MIN      0.0
#define X_MAX      1.0
#define Y_MIN      0.0
#define Y_MAX      1.0
#define Z_MIN      0.0
#define Z_MAX      0.0
#define N_BLOCKS_X 512
#define N_BLOCKS_Y 256
#define N_BLOCKS_Z   1

#endif
