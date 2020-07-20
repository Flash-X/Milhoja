#include "computeLaplacianEnergy_packet.h"
#include "computeLaplacianEnergy_block.h"

#include "DataPacket.h"

using namespace orchestration;

void ThreadRoutines::computeLaplacianEnergy_packet(const int tId, void* dataItem) {
    DataPacket*   packet = static_cast<DataPacket*>(dataItem);

    // The shared_ptrs in tileList will stay alive as long as the packet is
    // alive.  Therefore, we can pass the raw pointer content of the shared_ptr
    // to the per-block routine and know that it will not be a dangling pointer.
    for (unsigned int i=0; i<packet->tileList.size(); ++i) {
        computeLaplacianEnergy_block(tId, packet->tileList[i].get());
    }
}

