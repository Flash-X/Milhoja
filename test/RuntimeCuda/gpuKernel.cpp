#include "gpuKernel.h"

#include <cstdlib>

#include "Grid_REAL.h"

#include "constants.h"

using namespace orchestration;

void   gpuKernel::kernel(void* packet_d) {
    // Unmarshall data from given data packet
    char*  p = static_cast<char*>(packet_d);

    int*       loGC_d = reinterpret_cast<int*>(p);
    p += MDIM * sizeof(int);
    int*       hiGC_d = reinterpret_cast<int*>(p);
    p += MDIM * sizeof(int);
    Real*      data_d = reinterpret_cast<Real*>(p);

    std::size_t   n = 0;

    // Run our kernel in the device with the anticipation
    // that the data is alredy present in the device memory
    #pragma acc data create(n) deviceptr(loGC_d, hiGC_d, data_d)
    {
    #pragma acc kernels
    n = (  (hiGC_d[IAXIS_C] - loGC_d[IAXIS_C] + 1)
         * (hiGC_d[JAXIS_C] - loGC_d[JAXIS_C] + 1)
         * (hiGC_d[KAXIS_C] - loGC_d[KAXIS_C] + 1)
         * NUNKVAR);

    #pragma acc parallel loop default(none)
    for (int i=0; i<n; ++i) {
        data_d[i] += 2.1;
    }
    }
}

