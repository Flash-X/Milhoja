#include "DataPacket.h"

#ifdef USE_CUDA_BACKEND
#include "CudaDataPacket.h"
#endif

namespace orchestration {

/**
 *
 */
std::unique_ptr<DataPacket>   DataPacket::createPacket(void) {
#ifdef USE_CUDA_BACKEND
    return std::unique_ptr<DataPacket>{ new CudaDataPacket{} };
#else
#error "Selected runtime backend does not support data packets"
#endif
}

}

