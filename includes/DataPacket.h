#ifndef DATA_PACKET_H__
#define DATA_PACKET_H__

#include "Grid_IntVect.h"
#include "Grid_RealVect.h"
#include "FArray1D.h"
#include "FArray4D.h"
#include "Tile.h"
#include "DataItem.h"

#include "CudaStream.h"

namespace orchestration {

enum class PacketDataLocation {NOT_ASSIGNED, CC1, CC2};

struct PacketContents {
    unsigned int  level    = 0;
    RealVect*     deltas   = nullptr;
    IntVect*      lo       = nullptr;
    IntVect*      hi       = nullptr;
    IntVect*      loGC     = nullptr;
    IntVect*      hiGC     = nullptr;
    FArray1D*     xCoords  = nullptr;
    FArray1D*     yCoords  = nullptr;
    FArray4D*     CC1      = nullptr;
    FArray4D*     CC2      = nullptr;
};

class DataPacket : public DataItem {
public:
    static std::unique_ptr<DataPacket>   createPacket(std::shared_ptr<Tile>&& tileDesc);

    virtual ~DataPacket(void)  { };

    DataPacket(DataPacket&)                  = delete;
    DataPacket(const DataPacket&)            = delete;
    DataPacket(DataPacket&& packet)          = delete;
    DataPacket& operator=(DataPacket&)       = delete;
    DataPacket& operator=(const DataPacket&) = delete;
    DataPacket& operator=(DataPacket&&)      = delete;

    virtual void                   initiateHostToDeviceTransfer(void) = 0;
    virtual void                   transferFromDeviceToHost(void) = 0;
    virtual CudaStream&            stream(void) = 0;

    virtual PacketDataLocation    getDataLocation(void) const = 0;
    virtual void                  setDataLocation(const PacketDataLocation location) = 0;
    virtual void                  setVariableMask(const int startVariable, 
                                                  const int endVariable) = 0;

    virtual std::shared_ptr<Tile>  getTile(void) = 0;
    virtual const PacketContents   gpuContents(void) const = 0;

protected:
    DataPacket(void)   { };
};

}

#endif

