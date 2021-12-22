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
      : location_{PacketDataLocation::NOT_ASSIGNED},
        packet_p_{nullptr},
        packet_d_{nullptr},
        copyInStart_p_{nullptr},
        copyInStart_d_{nullptr},
        copyInOutStart_p_{nullptr},
        copyInOutStart_d_{nullptr},
        tiles_{},
        contents_p_{nullptr},
        contents_d_{nullptr},
        pinnedPtrs_{nullptr},
        stream_{},
        nCopyToGpuBytes_{0},
        nReturnToHostBytes_{0},
        startVariable_{-1},
        endVariable_{-1}
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

    if (pinnedPtrs_) {
        delete [] pinnedPtrs_;
        pinnedPtrs_ = nullptr;
    }

    location_ = PacketDataLocation::NOT_ASSIGNED;

    startVariable_ = -1;
    endVariable_   = -1;

    nCopyToGpuBytes_    = 0;
    nReturnToHostBytes_ = 0;

    contents_p_       = nullptr;
    contents_d_       = nullptr;
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
    } else if (location_ != PacketDataLocation::NOT_ASSIGNED) {
        return "Data location already assigned";
    } else if (startVariable_ >= 0) {
        return "Start variable already set";
    } else if (endVariable_ >= 0) {
        return "End variable already set";
    } else if (nCopyToGpuBytes_ > 0) {
        return "Non-zero packet size";
    } else if (nReturnToHostBytes_ > 0) {
        return "Non-zero packet size";
    } else if (contents_p_ != nullptr) {
        return "Pinned contents exist";
    } else if (contents_d_ != nullptr) {
        return "GPU contents exist";
    } else if (copyInStart_p_ != nullptr) {
        return "Pinned copy in buffer exists";
    } else if (copyInStart_d_ != nullptr) {
        return "GPU copy in buffer exists";
    } else if (copyInOutStart_p_ != nullptr) {
        return "Pinned copy back buffer exists";
    } else if (copyInOutStart_d_ != nullptr) {
        return "GPU copy back buffer exists";
    } else if (pinnedPtrs_ != nullptr) {
        return "Pinned pointers exist";
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

/**
 * Obtain the location of the correct cell-centered data.
 */
PacketDataLocation    DataPacket::getDataLocation(void) const {
    return location_;
}

/**
 * Specify the location of the correct cell-centered data so that the next
 * runtime element to use the packet knows where to look.
 */
void   DataPacket::setDataLocation(const PacketDataLocation location) {
    location_ = location;
}

/**
 * We assume that all variable indices are non-negative and that the end index
 * is greater than or equal to the start index.  Other than this, there is no
 * error checking of the variables performed here.
 *
 * @todo The present interface allows task functions to specify the variable
 * mask.  Since concrete DataPackets are now coupled to task functions, it seems
 * like this should be known when the concrete classes are designed.  Remove the
 * setVariableMask from the public interface.
 */
void   DataPacket::setVariableMask(const int startVariable,
                                   const int endVariable) {
    if        (startVariable < 0) {
        throw std::logic_error("[DataPacket::setVariableMask] "
                               "Starting variable index is negative");
    } else if (endVariable < 0) {
        throw std::logic_error("[DataPacket::setVariableMask] "
                               "Ending variable index is negative");
    } else if (startVariable > endVariable) {
        throw std::logic_error("[DataPacket::setVariableMask] "
                               "Starting variable > ending variable");
    }

    startVariable_ = startVariable;
    endVariable_ = endVariable;
}

}

