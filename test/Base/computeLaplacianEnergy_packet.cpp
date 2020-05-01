#include "computeLaplacianEnergy_packet.h"
#include "computeLaplacianEnergy_block.h"

void ThreadRoutines::computeLaplacianEnergy_packet(const int tId, void* dataItem) {
    DataPacket*   packet = static_cast<DataPacket*>(dataItem);

    for (unsigned int i=0; i<packet->tileList.size(); ++i) {
//        Tile    work = std::move(packet->tileList.front());
        Tile    work = packet->tileList.front();
        computeLaplacianEnergy_block(tId, &work);
        packet->tileList.pop();
    }
}

