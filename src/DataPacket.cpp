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
        startVariable_{UNK_VARS_BEGIN_C - 1},
        endVariable_{UNK_VARS_BEGIN_C - 1},
        packet_p_{nullptr},
        packet_d_{nullptr},
        tiles_{},
        nTiles_d_{nullptr},
        contents_p_{nullptr},
        contents_d_{nullptr},
        stream_{},
        nBytesPerPacket_{0}
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

}

