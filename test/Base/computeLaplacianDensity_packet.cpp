#include "computeLaplacianDensity_packet.h"
#include "computeLaplacianDensity_block.h"

#include "Tile.h"
#include "DataPacket.h"
using namespace orchestration;

void ThreadRoutines::computeLaplacianDensity_packet(const int tId, void* dataItem) {
    DataPacket*  packet = static_cast<DataPacket*>(dataItem);

    // TODO: Check if it is more efficient to use the iterator rather than
    // indexing.  Check when the the tile list is both small and large.
    for (unsigned int i=0; i<packet->tileList.size(); ++i) {
        Tile&    work = packet->tileList[i];
        computeLaplacianDensity_block(tId, &work);
    }
}

