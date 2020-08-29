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

    int   i0 = 0;
    int   j0 = 0;
    int   k0 = 0;
    int   jstride = 0;
    int   kstride = 0;
    int   nstride = 0;

    // Run our kernel in the device with the anticipation
    // that the data is alredy present in the device memory
    #pragma acc data create(i0, j0, k0, jstride, kstride, nstride) \
                     deviceptr(loGC_d, hiGC_d, data_d)
    {
        #pragma acc kernels
        {
        i0 = loGC_d[IAXIS_C];
        j0 = loGC_d[JAXIS_C];
        k0 = loGC_d[KAXIS_C];
        jstride =            hiGC_d[IAXIS_C] - i0 + 1;
        kstride = jstride * (hiGC_d[JAXIS_C] - j0 + 1);
        nstride = kstride * (hiGC_d[KAXIS_C] - k0 + 1);
        }

        #pragma acc parallel loop default(none)
        for (int i=0; i<nstride; ++i) {
            data_d[i]         += 2.1;
            data_d[i+nstride] += 2.1;
        }
    }
}

