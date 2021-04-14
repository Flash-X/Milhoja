#include "DataPacket.h"

#include "Backend.h"

#include "Flash.h"

namespace orchestration {

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
        startVariable_{UNK_VARS_BEGIN_C - 1},
        endVariable_{UNK_VARS_BEGIN_C - 1}
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
        Backend::instance().releaseStream(stream_);
    }
    Backend::instance().releaseGpuMemory(&packet_p_, &packet_d_);

    if (pinnedPtrs_) {
        delete [] pinnedPtrs_;
        pinnedPtrs_ = nullptr;
    }

    location_ = PacketDataLocation::NOT_ASSIGNED;

    startVariable_ = UNK_VARS_BEGIN_C - 1;
    endVariable_   = UNK_VARS_BEGIN_C - 1;

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
    } else if (startVariable_ >= UNK_VARS_BEGIN_C) {
        return "Start variable already set";
    } else if (endVariable_ >= UNK_VARS_BEGIN_C) {
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
 * @todo The present interface allows task functions to specify the variable
 * mask.  Since concrete DataPackets are now coupled to task functions, it seems
 * like this should be known when the concrete classes are designed.  Remove the
 * setVariableMask from the public interface.
 */
void   DataPacket::setVariableMask(const int startVariable,
                                   const int endVariable) {
    if        (startVariable < UNK_VARS_BEGIN_C) {
        throw std::logic_error("[DataPacket::setVariableMask] "
                               "Starting variable is invalid");
    } else if (endVariable > UNK_VARS_END_C) {
        throw std::logic_error("[DataPacket::setVariableMask] "
                               "Ending variable is invalid");
    } else if (startVariable > endVariable) {
        throw std::logic_error("[DataPacket::setVariableMask] "
                               "Starting variable > ending variable");
    }

    startVariable_ = startVariable;
    endVariable_ = endVariable;
}

/**
 * The runtime calls this member function automatically once a DataPacket
 * has arrived in the host memory again.  It is responsible for unpacking
 * the contents and in particular for copying cell-centered data back to the
 * host-side Grid data structures that hold solution data.  The data is copied
 * back in accord with the variable masks set in the DataPacket to avoid
 * inadvertently overwriting variables that were updated in parallel by other
 * actions.
 *
 * All memory and stream resources are released.
 *
 * While the packet is consumed once the function finishes, the list of Tiles
 * that were included in the packet is preserved.  This is necessary so that
 * runtime elements such as MoverUnpacker can enqueue the Tiles with its data
 * subscriber.
 */
void  DataPacket::unpack(void) {
    if (tiles_.size() <= 0) {
        throw std::logic_error("[DataPacket::unpack] "
                               "Empty data packet");
    } else if (!stream_.isValid()) {
        throw std::logic_error("[DataPacket::unpack] "
                               "Stream not acquired");
    } else if (pinnedPtrs_ == nullptr) {
        throw std::logic_error("[DataPacket::unpack] "
                               "No pinned pointers set");
    } else if (   (startVariable_ < UNK_VARS_BEGIN_C )
               || (startVariable_ > UNK_VARS_END_C )
               || (endVariable_   < UNK_VARS_BEGIN_C )
               || (endVariable_   > UNK_VARS_END_C)) {
        throw std::logic_error("[DataPacket::unpack] "
                               "Invalid variable mask");
    }

    // Release stream as soon as possible
    Backend::instance().releaseStream(stream_);
    assert(!stream_.isValid());

    for (std::size_t n=0; n<tiles_.size(); ++n) {
        Tile*   tileDesc_h = tiles_[n].get();

        Real*         data_h = tileDesc_h->dataPtr();
        const Real*   data_p = nullptr;
        switch (location_) {
            case PacketDataLocation::CC1:
                data_p = pinnedPtrs_[n].CC1_data;
                break;
            case PacketDataLocation::CC2:
                data_p = pinnedPtrs_[n].CC2_data;
                break;
            default:
                throw std::logic_error("[DataPacket::unpack] Data not in CC1 or CC2");
        }

        if (data_h == nullptr) {
            throw std::logic_error("[DataPacket::unpack] "
                                   "Invalid pointer to data in host memory");
        } else if (data_p == nullptr) {
            throw std::runtime_error("[DataPacket::unpack] "
                                     "Invalid pointer to data in pinned memory");
        }

        // The code here imposes requirements on the variable indices.  See Flash.h
        // for more information.  If this code is changed, please make sure to
        // adjust Flash.h appropriately.
        assert(UNK_VARS_BEGIN_C == 0);
        assert(UNK_VARS_END_C == (NUNKVAR - 1));
        std::size_t  offset =   N_ELEMENTS_PER_CC_PER_VARIABLE
                              * static_cast<std::size_t>(startVariable_);
        Real*        start_h = data_h + offset;
        const Real*  start_p = data_p + offset;
        std::size_t  nBytes =  (endVariable_ - startVariable_ + 1)
                              * N_ELEMENTS_PER_CC_PER_VARIABLE
                              * sizeof(Real);
        std::memcpy((void*)start_h, (void*)start_p, nBytes);
    }

    // The packet is consumed upon unpacking.
    nullify();
}

}

