#include "DataPacket.h"

#include "Backend.h"

#ifdef USE_CUDA_BACKEND
#include "CudaDataPacket.h"
#endif

#include "Flash.h"

namespace orchestration {

/**
 *
 */
std::unique_ptr<DataPacket>   DataPacket::createPacket(void) {
#ifdef USE_CUDA_BACKEND
    return std::unique_ptr<DataPacket>{ new CudaDataPacket{} };
#else
#error "Selected runtime backend does not support data packets"
#endif
}

DataPacket::DataPacket(void)
      : location_{PacketDataLocation::NOT_ASSIGNED},
        packet_p_{nullptr},
        packet_d_{nullptr},
        tiles_{},
        nTiles_d_{nullptr},
        contents_p_{nullptr},
        contents_d_{nullptr},
        stream_{},
        nBytesPerPacket_{0},
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

DataPacket::~DataPacket(void) {
    nullify();
}

/**
 *
 */
void  DataPacket::nullify(void) {
    if (stream_.isValid()) {
        Backend::instance().releaseStream(stream_);
    }
    Backend::instance().releaseGpuMemory(&packet_p_, &packet_d_);

    location_ = PacketDataLocation::NOT_ASSIGNED;

    startVariable_ = UNK_VARS_BEGIN_C - 1;
    endVariable_   = UNK_VARS_BEGIN_C - 1;

    nBytesPerPacket_ = 0;

    nTiles_d_   = nullptr;
    contents_p_ = nullptr;
    contents_d_ = nullptr;
}

/**
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
    } else if (nBytesPerPacket_ > 0) {
        return "Non-zero packet size";
    } else if (nTiles_d_ != nullptr) {
        return "N tiles exist in GPU";
    } else if (contents_p_ != nullptr) {
        return "Pinned contents exist";
    } else if (contents_d_ != nullptr) {
        return "GPU contents exist";
    }

    return "";
}

/**
 *
 */
void   DataPacket::addTile(std::shared_ptr<Tile>&& tileDesc) {
    tiles_.push_front( std::move(tileDesc) );
    if ((tileDesc != nullptr) || (tileDesc.use_count() != 0)) {
        throw std::runtime_error("[DataPacket::addTile] Ownership of tileDesc not transferred");
    }
}

/**
 *
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
 *
 */
PacketDataLocation    DataPacket::getDataLocation(void) const {
    return location_;
}

/**
 *
 */
void   DataPacket::setDataLocation(const PacketDataLocation location) {
    location_ = location;
}

/**
 *
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
 *
 */
void  DataPacket::unpack(void) {
    if (tiles_.size() <= 0) {
        throw std::logic_error("[DataPacket::unpack] "
                               "Empty data packet");
    } else if (packet_p_ == nullptr) {
        throw std::logic_error("[DataPacket::unpack] "
                               "No pointer to start of packet in pinned memory");
    } else if (packet_d_ == nullptr) {
        throw std::logic_error("[DataPacket::unpack] "
                               "No pointer to start of packet in GPU memory");
    } else if (!stream_.isValid()) {
        throw std::logic_error("[DataPacket::unpack] "
                               "Stream not acquired");
    } else if (contents_p_ == nullptr) {
        throw std::logic_error("[DataPacket::unpack] "
                               "No pinned packet contents");
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

    PacketContents*   tilePtrs_p = contents_p_;
    for (std::size_t n=0; n<tiles_.size(); ++n, ++tilePtrs_p) {
        Tile*   tileDesc_h = tiles_[n].get();

        Real*         data_h = tileDesc_h->dataPtr();
        const Real*   data_p = nullptr;
        switch (location_) {
            case PacketDataLocation::CC1:
                data_p = tilePtrs_p->CC1_data_p;
                break;
            case PacketDataLocation::CC2:
                data_p = tilePtrs_p->CC2_data_p;
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

    // The packet is consumed upon unpacking.  However, we still keep the
    // contents intact so that runtime elements such as MoverUnpacker can 
    // enqueue the tiles with its data subscriber.
    nullify();
}

}

