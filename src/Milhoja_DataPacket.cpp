#include "Milhoja_DataPacket.h"

#include "Milhoja_RuntimeBackend.h"

namespace milhoja {

/**
 * Construct a DataPacket containing no Tile objects and with no resources
 * assigned to it.
 *
 * @todo The error checks below should be asserts.
 */
DataPacket::DataPacket(void)
      : packet_p_{nullptr},
        packet_d_{nullptr},
        copyInStart_p_{nullptr},
        copyInStart_d_{nullptr},
        copyInOutStart_p_{nullptr},
        copyInOutStart_d_{nullptr},
        tiles_{},
        stream_{},
        nCopyToGpuBytes_{0},
        nReturnToHostBytes_{0}
{
    std::string   errMsg = isNull();
    if (errMsg != "") {
        throw std::logic_error("[DataPacket::DataPacket] " + errMsg);
    }

    if (tiles_.size() != 0) {
        throw std::runtime_error("[DataPacket::DataPacket] tiles_ not empty");
    }
}

/**
 * Destroy object.  Under normal operations, this object should have been
 * consumed prior to the execution of the destructor and therefore it should
 * not own resources.  However, resources are released upon abnormal
 * termination.
 */
DataPacket::~DataPacket(void) {
    nullify();
}

/**
 * Set the DataPacket into a null/empty state.  This includes releasing any
 * resources owned by the object and setting data members to ridiculous values.
 */
void  DataPacket::nullify(void) {
    if (stream_.isValid()) {
        RuntimeBackend::instance().releaseStream(stream_);
    }
    RuntimeBackend::instance().releaseGpuMemory(&packet_p_, &packet_d_);

    nCopyToGpuBytes_    = 0;
    nReturnToHostBytes_ = 0;

    copyInStart_p_    = nullptr;
    copyInStart_d_    = nullptr;
    copyInOutStart_p_ = nullptr;
    copyInOutStart_d_ = nullptr;
}

/**
 * Determine if the packet is in the null/empty state.
 *
 * @return An empty string if yes; otherwise, an explanation of why it is not
 * null.
 *
 * @todo This could eventually disappear once the data packets are written by
 * code generator.
 */
std::string  DataPacket::isNull(void) const {
    if (stream_.isValid()) {
        return "Stream already acquired";
    } else if (packet_p_ != nullptr) {
        return "Pinned memory buffer has already been allocated";
    } else if (packet_d_ != nullptr) {
        return "Device memory buffer has already been allocated";
    } else if (nCopyToGpuBytes_ > 0) {
        return "Non-zero packet size";
    } else if (nReturnToHostBytes_ > 0) {
        return "Non-zero packet size";
    } else if (copyInStart_p_ != nullptr) {
        return "Pinned copy in buffer exists";
    } else if (copyInStart_d_ != nullptr) {
        return "GPU copy in buffer exists";
    } else if (copyInOutStart_p_ != nullptr) {
        return "Pinned copy back buffer exists";
    } else if (copyInOutStart_d_ != nullptr) {
        return "GPU copy back buffer exists";
    }

    return "";
}

/**
 * Add the given Tile to the DataPacket.  As part of this, the packet assumes
 * ownership of the Tile and the calling code's pointer is nullified.
 *
 * @todo The checks should be asserts.  Figure out how to get that working.
 */
void   DataPacket::addTile(std::shared_ptr<Tile>&& tileDesc) {
    tiles_.push_front( std::move(tileDesc) );
    if ((tileDesc != nullptr) || (tileDesc.use_count() != 0)) {
        throw std::runtime_error("[DataPacket::addTile] Ownership of tileDesc not transferred");
    }
}

/**
 * Obtain one of the Tiles included in the packet.  As part of this, the Tile is
 * removed from the packet and ownership of the Tile passes to the calling code.
 *
 * @todo The checks should be asserts.  Figure out how to get that working.
 */
std::shared_ptr<Tile>  DataPacket::popTile(void) {
    if (tiles_.size() == 0) {
        throw std::invalid_argument("[DataPacket::popTile] No tiles to pop");
    }

    std::shared_ptr<Tile>   tileDesc{ std::move(tiles_.front()) };
    if (   (tiles_.front() != nullptr)
        || (tiles_.front().use_count() != 0)) {
        throw std::runtime_error("[DataPacket::popTile] Ownership of tileDesc not transferred");
    } 

    tiles_.pop_front();
    if ((tileDesc == nullptr) || (tileDesc.use_count() == 0)) {
        throw std::runtime_error("[DataPacket::popTile] Bad tileDesc");
    }

    return tileDesc;
}
}

