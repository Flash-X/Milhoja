#include <cassert>

#include "MoverUnpacker.h"

#include "DataPacket.h"

namespace orchestration {

void MoverUnpacker::increaseThreadCount(const unsigned int nThreads) {
    throw std::logic_error("[MoverUnpacker::increaseThreadCount] "
                           "MoverUnpackers do no have threads to awaken");
}

void MoverUnpacker::enqueue(std::shared_ptr<DataItem>&& dataItem) {
    // Unpacking only makes sense if the given data item is a packet.
    // Therefore, this ugly upcasting is reasonable.
    DataPacket*    packet = dynamic_cast<DataPacket*>(dataItem.get());
    // TODO: This is presently blocking as the transfer is effectively 
    // synchronous.  How to alter this so that the transfer is truly
    // asynchronous?
    packet->transferFromDeviceToHost();

    // Transfer the ownership of the data item in the packet to the next team
    if (dataReceiver_) {
        while (packet->nTiles() > 0) {
            dataReceiver_->enqueue( std::move(packet->popTile()) );
        }
    }

    // This function must take over control of the packet from the calling code.
    // In this case, the data packet is now no longer needed.
    // TODO: Is this necessary and correct?
    dataItem.reset();
    if ((dataItem != nullptr) || (dataItem.use_count() != 0)) {
        throw std::logic_error("[MoverUnpacker::enqueue] Packet not released");
    }
}

void MoverUnpacker::closeQueue(void) {
    if (dataReceiver_) {
        dataReceiver_->closeQueue();
    }
}

}

