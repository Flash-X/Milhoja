// WIP: Somehow NDEBUG is getting set and deactivating the asserts
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>

#include "MoverUnpacker.h"

#include "DataPacket.h"

namespace orchestration {

void MoverUnpacker::increaseThreadCount(const unsigned int nThreads) {
    throw std::logic_error("[MoverUnpacker::increaseThreadCount] "
                           "MoverUnpackers do no have threads to awaken");
}

void MoverUnpacker::enqueue(std::shared_ptr<DataItem>&& dataItem) {
    DataPacket*    packet = dynamic_cast<DataPacket*>(dataItem.get());
    packet->transferFromDeviceToHost();

    // Transfer the ownership of the data item in the packet to the next team
    if (dataReceiver_) {
        while (packet->nTiles() > 0) {
            dataReceiver_->enqueue( std::move(packet->popTile()) );
        }
        assert(packet->nTiles() == 0);
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

