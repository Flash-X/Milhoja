#ifndef DATA_PACKET_H__
#define DATA_PACKET_H__

#include <deque>

#include "Tile.h"

// FIXME: This needs to be written as a full implementation and cleanly.
// TODO:  Should we preset the size of the deque to avoid automatic 
//        resizing when the runtime is active?
class DataPacket {
public:
    int                id = -1;
    std::deque<Tile>   tileList;

    DataPacket(void)  { clear(); }
    ~DataPacket(void) { clear(); }

    // TODO: Do we need to do something elementwise to make certain that 
    //       we are managing memory correctly?
    void clear(void) { id = -1; std::deque<Tile>().swap(this->tileList); }

private:
    // TODO: Do we need copy/move constructors/assignment?
//    DataPacket(DataPacket&) = delete;
//    DataPacket(const DataPacket&) = delete;
//    DataPacket(DataPacket&&) = delete;
//    DataPacket& operator=(DataPacket&) = delete;
//    DataPacket& operator=(const DataPacket&) = delete;
//    DataPacket& operator=(DataPacket&&) = delete;
};

#endif

