#include "gpuKernel.h"

#include <cstdlib>

#include "Grid_REAL.h"

using namespace orchestration;

void   gpuKernel::kernel(void* packet_d) {
    // Unmarshall data from given data packet
    char*  p = static_cast<char*>(packet_d);

    const std::size_t*  nCells_d = reinterpret_cast<std::size_t*>(p);
    p += sizeof(std::size_t);
    Real*               data_d   = reinterpret_cast<Real*>(p);

    // Run our kernel in the device with the anticipation
    // that the data is alredy present in the device memory
    #pragma acc data deviceptr(nCells_d, data_d)
    {
    #pragma acc parallel loop default(none)
    for (std::size_t i=0; i<*nCells_d; ++i) {
        data_d[i] += 2.1; 
    }
    }
}

