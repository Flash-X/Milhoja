#include "CudaDataPacket.h"

#include <cstring>
#include <stdexcept>

#include "CudaStreamManager.h"
#include "CudaMemoryManager.h"

namespace orchestration {

/**
 *
 */
CudaDataPacket::CudaDataPacket(std::shared_ptr<Tile>&& tileDesc)
    : DataItem{},
      tileDesc_{ std::move(tileDesc) },
      data_p_{nullptr},
      packet_p_{nullptr},
      packet_d_{nullptr},
      stream_{}
{ }

/**
 *
 */
CudaDataPacket::~CudaDataPacket(void) {
    if (stream_.object != nullptr) {
        CudaStreamManager::instance().releaseStream(stream_);
        assert(stream_.object == nullptr);
        assert(stream_.id == CudaStream::NULL_STREAM_ID);
    }

    CudaMemoryManager::instance().releaseMemory(&packet_p_, &packet_d_);
    assert(packet_p_ == nullptr);
    assert(packet_d_ == nullptr);

    data_p_  = nullptr;
}

/**
 *
 */
std::size_t  CudaDataPacket::nSubItems(void) const {
    throw std::logic_error("[CudaDataPacket::nSubItems] Subitems not yet implemented");
}

/**
 *
 */
void   CudaDataPacket::addSubItem(std::shared_ptr<DataItem>&& dataItem) {
    throw std::logic_error("[CudaDataPacket::addSubItem] Subitems not yet implemented");
}

/**
 *
 */
std::shared_ptr<DataItem>  CudaDataPacket::popSubItem(void) {
    throw std::logic_error("[CudaDataPacket::popSubItem] Subitems not yet implemented");
}

/**
 *
 */
DataItem*  CudaDataPacket::getSubItem(const std::size_t i) {
    throw std::logic_error("[CudaDataPacket::getSubItem] Subitems not yet implemented");
}

/**
 *
 */
void  CudaDataPacket::pack(void) {
    if (tileDesc_ == nullptr) {
        throw std::logic_error("[CudaDataPacket::pack] "
                               "Null tile descriptor given");
    } else if ((stream_.object != nullptr) || (stream_.id != CudaStream::NULL_STREAM_ID)) {
        throw std::logic_error("[CudaDataPacket::pack] "
                               "CUDA stream already acquired");
    } else if (packet_p_) {
        throw std::logic_error("[CudaDataPacket::pack] "
                               "Pinned memory buffer has already been allocated");
    } else if (packet_d_) {
        throw std::logic_error("[CudaDataPacket::pack] "
                               "Device memory buffer has already been allocated");
    } else if (data_p_) {
        throw std::logic_error("[CudaDataPacket::pack] "
                               "Block buffere already allocated in pinned memory");
    }

    // For the present purpose of development, fail if no streams available
    stream_ = CudaStreamManager::instance().requestStream(false);
    assert(stream_.object != nullptr);
    assert(stream_.id != CudaStream::NULL_STREAM_ID);

    // Allocate memory in pinned and device memory on demand for now
    CudaMemoryManager::instance().requestMemory(N_BYTES_PER_PACKET,
                                                &packet_p_,
                                                &packet_d_);

    const IntVect    loGC = tileDesc_->loGC();
    const IntVect    hiGC = tileDesc_->hiGC();
    Real*            data_h = tileDesc_->dataPtr();
    Real*            data_d = nullptr;
    if (data_h == nullptr) {
        throw std::logic_error("[CudaDataPacket::pack] "
                               "Invalid pointer to data in host memory");
    }

    // Pointer to the next free byte in the current data packets
    assert(sizeof(char) == 1);   // Should be true by C++ standard
    char*   ptr_p = static_cast<char*>(packet_p_);
    char*   ptr_d = static_cast<char*>(packet_d_);

    // Pack data for single tile data packet
    std::memcpy((void*)ptr_p, (void*)&loGC, sizeof(IntVect));
    ptr_p += sizeof(IntVect);
    ptr_d += sizeof(IntVect);

    std::memcpy((void*)ptr_p, (void*)&hiGC, sizeof(IntVect));
    ptr_p += sizeof(IntVect);
    ptr_d += sizeof(IntVect);

    data_p_ = reinterpret_cast<Real*>(ptr_p);
    data_d  = reinterpret_cast<Real*>(ptr_d);
    std::memcpy((void*)ptr_p, (void*)data_h, BLOCK_SIZE_BYTES);
    ptr_p += BLOCK_SIZE_BYTES;
    ptr_d += BLOCK_SIZE_BYTES;

    // Create an FArray4D object in host memory but that already points
    // to where its data will be in device memory (i.e. the device object
    // will already be attached to its data in device memory).
    // The object in host memory should never be used then.
    // IMPORTANT: When this local object is destroyed, we don't want it to
    // affect the use of the copies (e.g. release memory).
    FArray4D   f_d{data_d, loGC, hiGC, NUNKVAR};
    std::memcpy((void*)ptr_p, (void*)&f_d, sizeof(FArray4D));
}

/**
 *
 */
void  CudaDataPacket::unpack(void) {
    if (tileDesc_ == nullptr) {
        throw std::logic_error("[CudaDataPacket::unpack] "
                               "Null tile descriptor given");
    } else if (packet_p_ == nullptr) {
        throw std::logic_error("[CudaDataPacket::unpack] "
                               "No pointer to start of packet in pinned memory");
    } else if (packet_d_ == nullptr) {
        throw std::logic_error("[CudaDataPacket::unpack] "
                               "No pointer to start of packet in GPU memory");
    } else if ((stream_.object == nullptr) || (stream_.id == CudaStream::NULL_STREAM_ID)) {
        throw std::logic_error("[CudaDataPacket::unpack] "
                               "CUDA stream not acquired");
    } else if (data_p_ == nullptr) {
        throw std::logic_error("[CudaDataPacket::unpack] "
                               "No pointer to data in pinned memory");
    }

    Real*   data_h = tileDesc_->dataPtr();
    if (data_h == nullptr) {
        throw std::logic_error("[CudaDataPacket::unpack] "
                               "Invalid pointer to data in host memory");
    }
    std::memcpy((void*)data_h, (void*)data_p_, BLOCK_SIZE_BYTES);

    CudaMemoryManager::instance().releaseMemory(&packet_p_, &packet_d_);

    data_p_  = nullptr;

    CudaStreamManager::instance().releaseStream(stream_);
    assert(stream_.object == nullptr);
    assert(stream_.id == CudaStream::NULL_STREAM_ID);
}

}

