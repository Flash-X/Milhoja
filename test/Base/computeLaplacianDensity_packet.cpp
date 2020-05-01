#include "computeLaplacianDensity_packet.h"
#include "computeLaplacianDensity_block.h"

void ThreadRoutines::computeLaplacianDensity_packet(const int tId, void* dataItem) {
    DataPacket*  packet = static_cast<DataPacket*>(dataItem);

    for (unsigned int i=0; i<packet->tileList.size(); ++i) {
//        Tile    work = std::move(packet->tileList.front());
        Tile    work = packet->tileList.front();
        computeLaplacianDensity_block(tId, &work);
        packet->tileList.pop();
    }
}

