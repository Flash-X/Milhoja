#include "DataPacket.h"

#ifdef USE_CUDA_BACKEND
#include "CudaDataPacket.h"
#endif

namespace orchestration {

/**
 *
 */
std::unique_ptr<DataPacket>   DataPacket::createPacket(std::shared_ptr<Tile>&& tileDesc) {
#ifdef USE_CUDA_BACKEND
    return std::unique_ptr<DataPacket>{ new CudaDataPacket{std::move(tileDesc)} };
#else
#error "Selected runtime backend does not support data packets"
#endif
}

}

