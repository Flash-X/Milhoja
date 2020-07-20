#include "scaleEnergy_packet.h"
#include "scaleEnergy_block.h"

#include "Tile.h"
#include "DataPacket.h"
using namespace orchestration;

void ThreadRoutines::scaleEnergy_packet(const int tId, void* dataItem) {
    DataPacket*  packet = static_cast<DataPacket*>(dataItem);

    for (unsigned int i=0; i<packet->tileList.size(); ++i) {
        scaleEnergy_block(tId, packet->tileList[i].get());
    }
}

