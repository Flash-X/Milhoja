#include "gpuKernel.h"

#include <stdio.h>

__global__
void gpuKernel::kernel_block(const std::size_t N, double* f,
                             CudaGpuArray* array,
                             const double* a,
                             const amrex::Dim3* loGC, const amrex::Dim3* hiGC) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < N) {
        if        (   (loGC->x != array->begin.x)
                   || (loGC->y != array->begin.y)
                   || (loGC->z != array->begin.z)) {
            printf("[gpuKernel::kernel_block] loGC != array.begin");
            return;
        } else if (   (hiGC->x != (array->end.x - 1))
                   || (hiGC->y != (array->end.y - 1))
                   || (hiGC->z != (array->end.z - 1)) ) {
            printf("[gpuKernel::kernel_block] hiGC != array.end");
            return;
        } else if (f != array->p) {
            printf("[gpuKernel::kernel_block] device pointers don't match");
            return;
        }

        f[i] *= *a;
    }
}

// TODO: At the moment, this routine does not block until all kernels that have
// been launched finish.  Therefore, if we like this idea, then the calling code
// will need to wait at some point until all kernels finish.
void gpuKernel::kernel_packet(CudaDataPacket& packet) {
    copyIn*      copyIn_p = reinterpret_cast<copyIn*>(packet.copyIn_p_);
    std::size_t  N = copyIn_p->nDataPerTile;

    copyIn*      copyIn_d = reinterpret_cast<copyIn*>(packet.copyIn_d_);
    double*      a_d = &(copyIn_d->coefficient);

    std::cout << "N data/block = " << N << std::endl;

    CudaGpuArray*   array_d = nullptr;
    for (std::size_t n=0; n<packet.nDataItems(); ++n) {
        Tile&  dataItem = packet[n];
        array_d = reinterpret_cast<CudaGpuArray*>(dataItem.CC1_array_d_);

        std::cout << "Apply kernel to block " << dataItem.gridIndex() << std::endl;
        kernel_block<<<(N+255)/256,256>>>(N, dataItem.CC1_d_, array_d, a_d,
                                          dataItem.loGC_d_, dataItem.hiGC_d_);
    }
}

