#include "gpuKernel.h"

#include <cstdlib>

#include "Grid_REAL.h"
#include "FArray4D.h"

#include "constants.h"

using namespace orchestration;

void   gpuKernel::kernel(void* packet_d) {
    constexpr std::size_t   N_CELLS =   (NXB + 2 * NGUARD * K1D)
                                      * (NYB + 2 * NGUARD * K2D)
                                      * (NZB + 2 * NGUARD * K3D)
                                      * NUNKVAR;

    // Unmarshall data from given data packet
    char*  p = static_cast<char*>(packet_d);

    const int*  loGC_d = reinterpret_cast<int*>(p);
    p += MDIM * sizeof(int);
    const int*  hiGC_d = reinterpret_cast<int*>(p);
    p += MDIM * sizeof(int);
    // No need to get pointer to data in device since we use the following
    // object to access the data and it is already configured with this pointer.
    p += N_CELLS*sizeof(Real);
    FArray4D*   f_d = reinterpret_cast<FArray4D*>(p);

    // Run our kernel in the device with the anticipation
    // that the data is alredy present in the device memory
    #pragma acc data deviceptr(loGC_d, hiGC_d, f_d)
    {
        #pragma acc parallel loop default(none)
        for         (int k=loGC_d[KAXIS_C]; k<=hiGC_d[KAXIS_C]; ++k) {
            #pragma acc loop
            for     (int j=loGC_d[JAXIS_C]; j<=hiGC_d[JAXIS_C]; ++j) {
                #pragma acc loop
                for (int i=loGC_d[IAXIS_C]; i<=hiGC_d[IAXIS_C]; ++i) {
                    (*f_d)(i, j, k, DENS_VAR_C) += 2.1;
                    (*f_d)(i, j, k, ENER_VAR_C) += 2.1;
                }
            }
        }
    }
}

