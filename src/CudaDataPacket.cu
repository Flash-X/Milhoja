#include "CudaDataPacket.h"

#include <cstring>
#include <stdexcept>

#include "Grid_RealVect.h"
#include "Grid_Axis.h"
#include "Grid_Edge.h"
#include "Grid.h"
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
      contents_d_{},
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

    contents_d_.level   = 0;
    contents_d_.deltas  = nullptr;
    contents_d_.lo      = nullptr;
    contents_d_.hi      = nullptr;
    contents_d_.loGC    = nullptr;
    contents_d_.hiGC    = nullptr;
    contents_d_.xCoords = nullptr;
    contents_d_.yCoords = nullptr;
    contents_d_.data    = nullptr;
    contents_d_.scratch = nullptr;
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
    } else if (   (contents_d_.deltas  != nullptr)
               || (contents_d_.lo      != nullptr)
               || (contents_d_.hi      != nullptr)
               || (contents_d_.loGC    != nullptr)
               || (contents_d_.hiGC    != nullptr)
               || (contents_d_.xCoords != nullptr)
               || (contents_d_.yCoords != nullptr)
               || (contents_d_.data    != nullptr)
               || (contents_d_.scratch != nullptr)) {
        throw std::logic_error("[CudaDataPacket::pack] "
                               "Contents object not nulled");
    }

    Grid&   grid = Grid::instance();

    // For the present purpose of development, fail if no streams available
    stream_ = CudaStreamManager::instance().requestStream(true);
    // TODO: Turn these in to actual checks
    assert(stream_.object != nullptr);
    assert(stream_.id != CudaStream::NULL_STREAM_ID);

    // Allocate memory in pinned and device memory on demand for now
    CudaMemoryManager::instance().requestMemory(N_BYTES_PER_PACKET,
                                                &packet_p_,
                                                &packet_d_);

    const unsigned int  level  = tileDesc_->level();
    const RealVect      deltas = tileDesc_->deltas();
    const IntVect       lo     = tileDesc_->lo();
    const IntVect       hi     = tileDesc_->hi();
    const IntVect       loGC   = tileDesc_->loGC();
    const IntVect       hiGC   = tileDesc_->hiGC();
    const FArray1D      xCoords = grid.getCellCoords(Axis::I, Edge::Center,
                                                     level, lo, hi); 
    const FArray1D      yCoords = grid.getCellCoords(Axis::J, Edge::Center,
                                                     level, lo, hi); 
    Real*               data_h = tileDesc_->dataPtr();
    Real*               data_d = nullptr;
    Real*               data_scratch_d = nullptr;
    const Real*         xCoords_h = xCoords.dataPtr();
    const Real*         yCoords_h = yCoords.dataPtr();
    if (data_h == nullptr) {
        throw std::logic_error("[CudaDataPacket::pack] "
                               "Invalid pointer to data in host memory");
    }

    // Pointer to the next free byte in the current data packets
    assert(sizeof(char) == 1);   // Should be true by C++ standard
    char*   ptr_p = static_cast<char*>(packet_p_);
    char*   ptr_d = static_cast<char*>(packet_d_);

    // TODO: I think that we should put in padding so that all objects 
    //       are byte aligned in the device's memory.
    contents_d_.level = level;
    contents_d_.deltas = reinterpret_cast<RealVect*>(ptr_d);
    std::memcpy((void*)ptr_p, (void*)&deltas, DELTA_SIZE_BYTES);
    ptr_p += DELTA_SIZE_BYTES;
    ptr_d += DELTA_SIZE_BYTES;

    // Pack data for single tile data packet
    contents_d_.lo = reinterpret_cast<IntVect*>(ptr_d);
    std::memcpy((void*)ptr_p, (void*)&lo, POINT_SIZE_BYTES);
    ptr_p += POINT_SIZE_BYTES;
    ptr_d += POINT_SIZE_BYTES;

    contents_d_.hi = reinterpret_cast<IntVect*>(ptr_d);
    std::memcpy((void*)ptr_p, (void*)&hi, POINT_SIZE_BYTES);
    ptr_p += POINT_SIZE_BYTES;
    ptr_d += POINT_SIZE_BYTES;

    contents_d_.loGC = reinterpret_cast<IntVect*>(ptr_d);
    std::memcpy((void*)ptr_p, (void*)&loGC, POINT_SIZE_BYTES);
    ptr_p += POINT_SIZE_BYTES;
    ptr_d += POINT_SIZE_BYTES;

    contents_d_.hiGC = reinterpret_cast<IntVect*>(ptr_d);
    std::memcpy((void*)ptr_p, (void*)&hiGC, POINT_SIZE_BYTES);
    ptr_p += POINT_SIZE_BYTES;
    ptr_d += POINT_SIZE_BYTES;

    data_p_ = reinterpret_cast<Real*>(ptr_p);
    data_d  = reinterpret_cast<Real*>(ptr_d);
    std::memcpy((void*)ptr_p, (void*)data_h, BLOCK_SIZE_BYTES);
    ptr_p += BLOCK_SIZE_BYTES;
    ptr_d += BLOCK_SIZE_BYTES;

    data_scratch_d = reinterpret_cast<Real*>(ptr_d);
    ptr_p += BLOCK_SIZE_BYTES;
    ptr_d += BLOCK_SIZE_BYTES;

    contents_h_.xCoordsData = reinterpret_cast<Real*>(ptr_p);
    contents_d_.xCoordsData = reinterpret_cast<Real*>(ptr_d);
    std::memcpy((void*)ptr_p, (void*)xCoords_h, COORDS_X_SIZE_BYTES);
    ptr_p += COORDS_X_SIZE_BYTES;
    ptr_d += COORDS_X_SIZE_BYTES;

    contents_h_.yCoordsData = reinterpret_cast<Real*>(ptr_p);
    contents_d_.yCoordsData = reinterpret_cast<Real*>(ptr_d);
    std::memcpy((void*)ptr_p, (void*)yCoords_h, COORDS_Y_SIZE_BYTES);
    ptr_p += COORDS_Y_SIZE_BYTES;
    ptr_d += COORDS_Y_SIZE_BYTES;

    contents_d_.xCoords = reinterpret_cast<FArray1D*>(ptr_d);
    FArray1D   xCoordArray_d{contents_d_.xCoordsData, lo.I()};
    std::memcpy((void*)ptr_p, (void*)&xCoordArray_d, ARRAY1_SIZE_BYTES);
    ptr_p += ARRAY1_SIZE_BYTES;
    ptr_d += ARRAY1_SIZE_BYTES;

    contents_d_.yCoords = reinterpret_cast<FArray1D*>(ptr_d);
    FArray1D   yCoordArray_d{contents_d_.yCoordsData, lo.J()};
    std::memcpy((void*)ptr_p, (void*)&yCoordArray_d, ARRAY1_SIZE_BYTES);
    ptr_p += ARRAY1_SIZE_BYTES;
    ptr_d += ARRAY1_SIZE_BYTES;
 
    // Create an FArray4D object in host memory but that already points
    // to where its data will be in device memory (i.e. the device object
    // will already be attached to its data in device memory).
    // The object in host memory should never be used then.
    // IMPORTANT: When this local object is destroyed, we don't want it to
    // affect the use of the copies (e.g. release memory).
    contents_d_.data = reinterpret_cast<FArray4D*>(ptr_d);
    FArray4D   f_d{data_d, loGC, hiGC, NUNKVAR};
    std::memcpy((void*)ptr_p, (void*)&f_d, ARRAY4_SIZE_BYTES);
    ptr_p += ARRAY4_SIZE_BYTES;
    ptr_d += ARRAY4_SIZE_BYTES;

    contents_d_.scratch = reinterpret_cast<FArray4D*>(ptr_d);
    FArray4D   scratch_d{data_scratch_d, loGC, hiGC, NUNKVAR};
    std::memcpy((void*)ptr_p, (void*)&scratch_d, ARRAY4_SIZE_BYTES);
    ptr_p += ARRAY4_SIZE_BYTES;
    ptr_d += ARRAY4_SIZE_BYTES;
    // Use pointers to determine size of packet and compare against
    // N_BYTES_PER_PACKET
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
    assert(packet_p_ == nullptr);
    assert(packet_d_ == nullptr);

    data_p_  = nullptr;

    contents_d_.level   = 0;
    contents_d_.deltas  = nullptr;
    contents_d_.lo      = nullptr;
    contents_d_.hi      = nullptr;
    contents_d_.loGC    = nullptr;
    contents_d_.hiGC    = nullptr;
    contents_d_.xCoords = nullptr;
    contents_d_.yCoords = nullptr;
    contents_d_.data    = nullptr;
    contents_d_.scratch = nullptr;

    CudaStreamManager::instance().releaseStream(stream_);
    assert(stream_.object == nullptr);
    assert(stream_.id == CudaStream::NULL_STREAM_ID);
}

}

