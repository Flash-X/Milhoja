#include "MoverUnpacker.h"

#include <cassert>

#include "CudaDataPacket.h"

namespace orchestration {

void MoverUnpacker::increaseThreadCount(const unsigned int nThreads) {
    throw std::logic_error("[MoverUnpacker::increaseThreadCount] "
                           "MoverUnpackers do no have threads to awaken");
}

void MoverUnpacker::enqueue(std::shared_ptr<DataItem>&& dataItem) {
    CudaDataPacket*    packet = dynamic_cast<CudaDataPacket*>(dataItem.get());
    packet->transferFromDeviceToHost();

    // Transfer the ownership of the data item in the packet to the next team
    if (dataReceiver_) {
        dataReceiver_->enqueue(dataItem->getTile());
    }

    // This function must take over control of the packet from the calling code.
    // In this case, the data packet is now no longer needed.
    // TODO: Is this necessary and correct?
    dataItem.reset();
    assert(dataItem == nullptr);
    assert(dataItem.use_count() == 0);
}

void MoverUnpacker::closeQueue(void) {
    if (dataReceiver_) {
        dataReceiver_->closeQueue();
    }
}

}

