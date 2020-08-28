#ifndef GPU_KERNEL_H__
#define GPU_KERNEL_H__

namespace gpuKernel {
    // Declare structure that holds all data that must be copied to device memory 
    // in the data packet, but that need not be transferred back.
//    struct copyIn {
//        std::size_t             nDataPerTile;
//        double                  coefficient;
//    };

    void kernel(void* packet_d);
}

#endif

