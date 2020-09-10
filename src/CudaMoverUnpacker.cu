#include "CudaMoverUnpacker.h"

#include <cassert>

namespace orchestration {

void CudaMoverUnpacker::increaseThreadCount(const unsigned int nThreads) {
    throw std::logic_error("[CudaMoverUnpacker::increaseThreadCount] "
                           "CudaMoverUnpackers do no have threads to awaken");
}

void CudaMoverUnpacker::enqueue(std::shared_ptr<DataItem>&& packet) {
    // Bring data back to host.  Use asynchronous transfer so that we can keep
    // the transfer off the default stream and therefore only wait on this
    // transfer.
    cudaStream_t  stream = *(packet->stream().object);
    cudaError_t   cErr = cudaMemcpyAsync(packet->hostPointer(), packet->gpuPointer(),
                                         packet->sizeInBytes(),
                                         cudaMemcpyDeviceToHost, stream);
    if (cErr != cudaSuccess) {
        std::string  errMsg = "[CudaMoverUnpacker::enqueue] ";
        errMsg += "Unable to execute D-to-H transfer\n";
        errMsg += "CUDA error - " + std::string(cudaGetErrorName(cErr)) + "\n";
        errMsg += std::string(cudaGetErrorString(cErr)) + "\n";
        throw std::runtime_error(errMsg);
    }
    cudaStreamSynchronize(stream);

    // Undo the packet => data transferred back to location in Grid's data
    // structures
    packet->unpack(); 

    // Transfer the ownership of the data item in the packet to the next team
    if (dataReceiver_) {
        dataReceiver_->enqueue(packet->getTile());
    }

    // This function must take over control of the packet from the calling code.
    // In this case, the data packet is now no longer needed.
    // TODO: Is this necessary and correct?
    packet.reset();
    assert(packet == nullptr);
    assert(packet.use_count() == 0);
}

void CudaMoverUnpacker::closeQueue(void) {
    if (dataReceiver_) {
        dataReceiver_->closeQueue();
    }
}

}

