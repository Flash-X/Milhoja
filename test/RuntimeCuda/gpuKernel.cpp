#include "gpuKernel.h"

using namespace orchestration;

void   gpuKernel::kernel(Real* data_d, const std::size_t nCells) {
        #pragma acc data copyin(nCells) deviceptr(data_d)
        {
        #pragma acc parallel loop
        for (std::size_t i=0; i<nCells; ++i) {
            data_d[i] += 2.1;
        }
        }
}

