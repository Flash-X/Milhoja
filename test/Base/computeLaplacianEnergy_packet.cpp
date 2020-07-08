#include "computeLaplacianEnergy_packet.h"
#include "computeLaplacianEnergy_block.h"

#include "Tile.h"
#include "DataPacket.h"
using namespace orchestration;

void ThreadRoutines::computeLaplacianEnergy_packet(const int tId, void* dataItem) {
    DataPacket*   packet = static_cast<DataPacket*>(dataItem);

    for (unsigned int i=0; i<packet->tileList.size(); ++i) {
        Tile&    work = packet->tileList[i];
        computeLaplacianEnergy_block(tId, &work);
    }
}

