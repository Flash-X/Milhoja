#include "computeLaplacianEnergy_cpu.h"

#include <cstdio>

#include <unistd.h>

#include "constants.h"

void ThreadRoutines::computeLaplacianEnergy_cpu(const unsigned int tId,
                                                const std::string& name,
                                                Block& block) {
    unsigned int                  idx       = block.index();
    unsigned int                  nGuard    = block.nGuardcells();
    std::array<unsigned int,NDIM> lo        = block.lo();
    std::array<unsigned int,NDIM> hi        = block.hi();
    std::array<int,NDIM>          loGC      = block.loGC();
    std::array<unsigned int,NDIM> blockSize = block.size();
    std::array<double,NDIM>       deltas    = block.deltas();
    double***                     f         = block.dataPtr();

    unsigned int nCellsX = blockSize[IAXIS] + 2*nGuard;
    unsigned int nCellsY = blockSize[JAXIS] + 2*nGuard;

    double**  buffer = new double*[nCellsX];
    buffer[0] = new double[nCellsX * nCellsY];
    for (unsigned int i=1; i<nCellsX; ++i) {
        buffer[i] = buffer[i-1] + nCellsY;
    }

    double f_i     = 0.0;
    double f_x_im1 = 0.0;
    double f_x_ip1 = 0.0;
    double f_y_im1 = 0.0;
    double f_y_ip1 = 0.0;
 
    double   dx_sqr_inv = 1.0 / (deltas[IAXIS] * deltas[IAXIS]);
    double   dy_sqr_inv = 1.0 / (deltas[JAXIS] * deltas[JAXIS]);

    // Compute Laplacian in buffer
    int i0 = loGC[IAXIS];
    int j0 = loGC[JAXIS];
    for      (int i=lo[IAXIS]; i<=hi[IAXIS]; ++i) {
         for (int j=lo[JAXIS]; j<=hi[JAXIS]; ++j) {
              f_i     = f[ENER_VAR][i-i0  ][j-j0  ];
              f_x_im1 = f[ENER_VAR][i-i0-1][j-j0  ];
              f_x_ip1 = f[ENER_VAR][i-i0+1][j-j0  ];
              f_y_im1 = f[ENER_VAR][i-i0  ][j-j0-1];
              f_y_ip1 = f[ENER_VAR][i-i0  ][j-j0+1];
              buffer[i-i0][j-j0] =   ((f_x_im1 + f_x_ip1) - 2.0*f_i) * dx_sqr_inv
                                   + ((f_y_im1 + f_y_ip1) - 2.0*f_i) * dy_sqr_inv;
         }
    }

#ifdef VERBOSE
    printf("[%s / Thread %d] Applied Laplacian to block %d\n", name.c_str(), tId, idx);
#endif

    // Overwrite interior of given block with Laplacian result
    for      (int i=lo[IAXIS]; i<=hi[IAXIS]; ++i) {
         for (int j=lo[JAXIS]; j<=hi[JAXIS]; ++j) {
            f[ENER_VAR][i-i0][j-j0] = buffer[i-i0][j-j0];
         }
    } 

    delete [] buffer[0];
    delete [] buffer;
}

