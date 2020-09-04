#include "gpuKernel.h"

#include <cstdlib>
#include <iostream>
#include <sstream>

#include "Grid_REAL.h"
#include "FArray4D.h"

#include "constants.h"

using namespace orchestration;

void   gpuKernel::kernel(void* packet_d, const int streamId) {
    // Unmarshall data from given data packet
    char*  p = static_cast<char*>(packet_d);

    const int*  loGC_d = reinterpret_cast<int*>(p);
    p += MDIM * sizeof(int);
    const int*  hiGC_d = reinterpret_cast<int*>(p);
    p += MDIM * sizeof(int);
    Real*       data_d = reinterpret_cast<Real*>(p);

    int            i0 = 0;
    int            j0 = 0;
    int            k0 = 0;
    int            in = 0;
    int            jn = 0;
    int            kn = 0;
    unsigned int   jstride = 0;
    unsigned int   kstride = 0;
    unsigned int   nstride = 0;

    // Run our kernel in the device with the anticipation
    // that the data is alredy present in the device memory
    #pragma acc data create(i0, j0, k0, in, jn, kn, jstride, kstride, nstride) \
                     deviceptr(loGC_d, hiGC_d, data_d)
    {
        #pragma acc parallel async(streamId)
        {
            i0 = loGC_d[IAXIS_C];
            j0 = loGC_d[JAXIS_C];
            k0 = loGC_d[KAXIS_C];
            in = hiGC_d[IAXIS_C];
            jn = hiGC_d[JAXIS_C];
            kn = hiGC_d[KAXIS_C];
            jstride =           (in - i0 + 1);
            kstride = jstride * (jn - j0 + 1);
            nstride = kstride * (kn - k0 + 1);

            #pragma acc loop default(none)
            for         (int k=k0; k<=kn; ++k) {
                #pragma acc loop default(none)
                for     (int j=j0; j<=jn; ++j) {
                    #pragma acc loop default(none)
                    for (int i=i0; i<=in; ++i) {
                        data_d[(i-i0) + (j-j0)*jstride + (k-k0)*kstride          ] = 2.1;
                        data_d[(i-i0) + (j-j0)*jstride + (k-k0)*kstride + nstride] = 3.1;
                    }
                }
            }
        }
    }
}

