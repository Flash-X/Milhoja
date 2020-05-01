#include "scaleEnergy_packet.h"
#include "scaleEnergy_block.h"

#include "DataPacket.h"

void ThreadRoutines::scaleEnergy_packet(const int tId, void* dataItem) {
    DataPacket*  packet = static_cast<DataPacket*>(dataItem);

    for (unsigned int i=0; i<packet->tileList.size(); ++i) {
//        Tile    work = std::move(packet->tileList.front());
        Tile    work = packet->tileList.front();
        scaleEnergy_block(tId, &work);
        packet->tileList.pop();
    }
}

