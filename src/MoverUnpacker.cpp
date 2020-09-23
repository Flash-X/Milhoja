#include "MoverUnpacker.h"

#include <cassert>

namespace orchestration {

void MoverUnpacker::increaseThreadCount(const unsigned int nThreads) {
    throw std::logic_error("[MoverUnpacker::increaseThreadCount] "
                           "MoverUnpackers do no have threads to awaken");
}

void MoverUnpacker::enqueue(std::shared_ptr<DataItem>&& packet) {
    packet->transferFromDeviceToHost();

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

void MoverUnpacker::closeQueue(void) {
    if (dataReceiver_) {
        dataReceiver_->closeQueue();
    }
}

}

