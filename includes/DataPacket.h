#ifndef DATA_PACKET_H__
#define DATA_PACKET_H__

#include <queue>

struct DataPacket {
    int                id = -1;
    std::queue<Tile>   tileList;
};

#endif

