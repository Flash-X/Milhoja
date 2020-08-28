#ifndef GPU_KERNEL_H__
#define GPU_KERNEL_H__

#include <cstdlib>

#include "Grid_REAL.h"

//#include <AMReX_Dim3.H>
//
//#include "CudaGpuArray.h"
//#include "CudaDataPacket.h"

namespace gpuKernel {
    // Declare structure that holds all data that must be copied to device memory 
    // in the data packet, but that need not be transferred back.
//    struct copyIn {
//        std::size_t             nDataPerTile;
//        double                  coefficient;
//    };
//
//    __host__   void  kernel_packet(CudaDataPacket& packet);
//    __global__ void  kernel_block(const std::size_t N, double* f,
//                                  CudaGpuArray* array,
//                                  const double* a,
//                                  const amrex::Dim3* loGC, const amrex::Dim3* hiGC);

    void kernel(orchestration::Real* data_d, const std::size_t nCells);
}

#endif

