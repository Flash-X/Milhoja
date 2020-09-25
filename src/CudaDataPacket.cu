#include "CudaDataPacket.h"

#include <cstring>
#include <stdexcept>

#include "Grid_RealVect.h"
#include "Grid_Axis.h"
#include "Grid_Edge.h"
#include "Grid.h"
#include "CudaStreamManager.h"
#include "CudaMemoryManager.h"

#include "Flash.h"

namespace orchestration {

/**
 *
 */
CudaDataPacket::CudaDataPacket(std::shared_ptr<Tile>&& tileDesc)
    : DataItem{},
      tileDesc_{ std::move(tileDesc) },
      CC1_data_p_{nullptr},
      CC2_data_p_{nullptr},
      location_{PacketDataLocation::NOT_ASSIGNED},
      startVariable_{UNK_VARS_BEGIN_C - 1},
      endVariable_{UNK_VARS_BEGIN_C - 1},
      packet_p_{nullptr},
      packet_d_{nullptr},
      contents_d_{},
      stream_{}
{
    std::string   errMsg = isNull();
    if (errMsg != "") {
        throw std::logic_error("[CudaDataPacket::CudaDataPacket] " + errMsg);
    }
}

/**
 *
 */
CudaDataPacket::~CudaDataPacket(void) {
    nullify();
}

/**
 *
 */
void  CudaDataPacket::nullify(void) {
    if (stream_.object != nullptr) {
        CudaStreamManager::instance().releaseStream(stream_);
        assert(stream_.object == nullptr);
        assert(stream_.id == CudaStream::NULL_STREAM_ID);
    }

    CudaMemoryManager::instance().releaseMemory(&packet_p_, &packet_d_);
    assert(packet_p_ == nullptr);
    assert(packet_d_ == nullptr);

    CC1_data_p_  = nullptr;
    CC2_data_p_  = nullptr;
    location_ = PacketDataLocation::NOT_ASSIGNED;

    startVariable_ = UNK_VARS_BEGIN_C - 1;
    endVariable_   = UNK_VARS_BEGIN_C - 1;

    contents_d_.level   = 0;
    contents_d_.deltas  = nullptr;
    contents_d_.lo      = nullptr;
    contents_d_.hi      = nullptr;
    contents_d_.loGC    = nullptr;
    contents_d_.hiGC    = nullptr;
    contents_d_.xCoords = nullptr;
    contents_d_.yCoords = nullptr;
    contents_d_.CC1     = nullptr;
    contents_d_.CC2     = nullptr;
}

/**
 *
 */
std::string  CudaDataPacket::isNull(void) const {
    if ((stream_.object != nullptr) || (stream_.id != CudaStream::NULL_STREAM_ID)) {
        return "CUDA stream already acquired";
    } else if (packet_p_ != nullptr) {
        return "Pinned memory buffer has already been allocated";
    } else if (packet_d_ != nullptr) {
        return "Device memory buffer has already been allocated";
    } else if ((CC1_data_p_ != nullptr) || (CC2_data_p_ != nullptr)) {
        return "Block buffers already allocated in pinned memory";
    } else if (location_ != PacketDataLocation::NOT_ASSIGNED) {
        return "Data location already assigned";
    } else if (   (contents_d_.deltas  != nullptr)
               || (contents_d_.lo      != nullptr)
               || (contents_d_.hi      != nullptr)
               || (contents_d_.loGC    != nullptr)
               || (contents_d_.hiGC    != nullptr)
               || (contents_d_.xCoords != nullptr)
               || (contents_d_.yCoords != nullptr)
               || (contents_d_.CC1     != nullptr)
               || (contents_d_.CC2     != nullptr)) {
        return "Contents object not nulled";
    }

    return "";
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
PacketDataLocation    CudaDataPacket::getDataLocation(void) const {
    return location_;
}

/**
 *
 */
void   CudaDataPacket::setDataLocation(const PacketDataLocation location) {
    location_ = location;
}

/**
 *
 */
void   CudaDataPacket::setVariableMask(const int sVar,
                                       const int eVar) {
    if        (sVar < UNK_VARS_BEGIN_C) {
        throw std::logic_error("[CudaDataPacket::setVariableMask] "
                               "Starting variable is invalid");
    } else if (eVar > UNK_VARS_END_C) {
        throw std::logic_error("[CudaDataPacket::setVariableMask] "
                               "Ending variable is invalid");
    } else if (sVar > eVar) {
        throw std::logic_error("[CudaDataPacket::setVariableMask] "
                               "Starting variable > ending variable");
    }

    startVariable_ = sVar;
    endVariable_ = eVar;
}

/**
 *
 */
void  CudaDataPacket::pack(void) {
    std::string   errMsg = isNull();
    if (errMsg != "") {
        throw std::logic_error("[CudaDataPacket::pack] " + errMsg);
    } else if (tileDesc_ == nullptr) {
        throw std::logic_error("[CudaDataPacket::pack] Null tile descriptor given");
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
    const Real*         xCoords_h = xCoords.dataPtr();
    const Real*         yCoords_h = yCoords.dataPtr();
    Real*               xCoords_data_d = nullptr;
    Real*               yCoords_data_d = nullptr;
    Real*               data_h = tileDesc_->dataPtr();
    Real*               CC1_data_d = nullptr;
    Real*               CC2_data_d = nullptr;
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

    location_ = PacketDataLocation::CC1;
    CC1_data_p_ = reinterpret_cast<Real*>(ptr_p);
    CC1_data_d  = reinterpret_cast<Real*>(ptr_d);
    std::memcpy((void*)ptr_p, (void*)data_h, BLOCK_SIZE_BYTES);
    ptr_p += BLOCK_SIZE_BYTES;
    ptr_d += BLOCK_SIZE_BYTES;

    CC2_data_p_ = reinterpret_cast<Real*>(ptr_p);
    CC2_data_d  = reinterpret_cast<Real*>(ptr_d);
    ptr_p += BLOCK_SIZE_BYTES;
    ptr_d += BLOCK_SIZE_BYTES;

    xCoords_data_d = reinterpret_cast<Real*>(ptr_d);
    std::memcpy((void*)ptr_p, (void*)xCoords_h, COORDS_X_SIZE_BYTES);
    ptr_p += COORDS_X_SIZE_BYTES;
    ptr_d += COORDS_X_SIZE_BYTES;

    yCoords_data_d = reinterpret_cast<Real*>(ptr_d);
    std::memcpy((void*)ptr_p, (void*)yCoords_h, COORDS_Y_SIZE_BYTES);
    ptr_p += COORDS_Y_SIZE_BYTES;
    ptr_d += COORDS_Y_SIZE_BYTES;

    contents_d_.xCoords = reinterpret_cast<FArray1D*>(ptr_d);
    FArray1D   xCoordArray_d{xCoords_data_d, lo.I()};
    std::memcpy((void*)ptr_p, (void*)&xCoordArray_d, ARRAY1_SIZE_BYTES);
    ptr_p += ARRAY1_SIZE_BYTES;
    ptr_d += ARRAY1_SIZE_BYTES;

    contents_d_.yCoords = reinterpret_cast<FArray1D*>(ptr_d);
    FArray1D   yCoordArray_d{yCoords_data_d, lo.J()};
    std::memcpy((void*)ptr_p, (void*)&yCoordArray_d, ARRAY1_SIZE_BYTES);
    ptr_p += ARRAY1_SIZE_BYTES;
    ptr_d += ARRAY1_SIZE_BYTES;
 
    // Create an FArray4D object in host memory but that already points
    // to where its data will be in device memory (i.e. the device object
    // will already be attached to its data in device memory).
    // The object in host memory should never be used then.
    // IMPORTANT: When this local object is destroyed, we don't want it to
    // affect the use of the copies (e.g. release memory).
    contents_d_.CC1 = reinterpret_cast<FArray4D*>(ptr_d);
    FArray4D   CC1_d{CC1_data_d, loGC, hiGC, NUNKVAR};
    std::memcpy((void*)ptr_p, (void*)&CC1_d, ARRAY4_SIZE_BYTES);
    ptr_p += ARRAY4_SIZE_BYTES;
    ptr_d += ARRAY4_SIZE_BYTES;

    contents_d_.CC2 = reinterpret_cast<FArray4D*>(ptr_d);
    FArray4D   CC2_d{CC2_data_d, loGC, hiGC, NUNKVAR};
    std::memcpy((void*)ptr_p, (void*)&CC2_d, ARRAY4_SIZE_BYTES);
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
    } else if ((CC1_data_p_ == nullptr) || (CC2_data_p_ == nullptr)) {
        throw std::logic_error("[CudaDataPacket::unpack] "
                               "No pointer to data in pinned memory");
    } else if (   (startVariable_ < UNK_VARS_BEGIN_C )
               || (startVariable_ > UNK_VARS_END_C )
               || (endVariable_   < UNK_VARS_BEGIN_C )
               || (endVariable_   > UNK_VARS_END_C)) {
        throw std::logic_error("[CudaDataPacket::unpack] "
                               "Invalid variable mask");
    }

    // Release stream as soon as possible
    CudaStreamManager::instance().releaseStream(stream_);
    assert(stream_.object == nullptr);
    assert(stream_.id == CudaStream::NULL_STREAM_ID);

    Real*   data_h = tileDesc_->dataPtr();
    if (data_h == nullptr) {
        throw std::logic_error("[CudaDataPacket::unpack] "
                               "Invalid pointer to data in host memory");
    }

    Real*   data_p = nullptr;
    switch (location_) {
        case PacketDataLocation::CC1:
            data_p = CC1_data_p_;
            break;
        case PacketDataLocation::CC2:
            data_p = CC2_data_p_;
            break;
        default:
            throw std::logic_error("[CudaDataPacket::unpack] Data not in CC1 or CC2");
    }

    // The code here imposes requirements on the variable indices.  See Flash.h
    // for more information.  If this code is changed, please make sure to
    // adjust Flash.h appropriately.
    assert(UNK_VARS_BEGIN_C == 0);
    assert(UNK_VARS_END_C == (NUNKVAR - 1));
    std::size_t  offset =   N_CELLS_PER_VARIABLE
                          * static_cast<std::size_t>(startVariable_);
    Real*        start_h = data_h + offset;
    Real*        start_p = data_p + offset;
    std::size_t  nBytes =  (endVariable_ - startVariable_ + 1)
                          * N_CELLS_PER_VARIABLE
                          * sizeof(Real);
    std::memcpy((void*)start_h, (void*)start_p, nBytes);

    // The packet is consumed upon unpacking
    nullify();
}

/**
 *
 */
void  CudaDataPacket::initiateHostToDeviceTransfer(void) {
    pack();

    cudaStream_t  stream = *(stream_.object);
    cudaError_t cErr = cudaMemcpyAsync(packet_d_, packet_p_,
                                       N_BYTES_PER_PACKET,
                                       cudaMemcpyHostToDevice,
                                       stream);
    if (cErr != cudaSuccess) {
        std::string  errMsg = "[CudaDataPacket::initiateHostToDeviceTransfer] ";
        errMsg += "Unable to execute H-to-D transfer\n";
        errMsg += "CUDA error - " + std::string(cudaGetErrorName(cErr)) + "\n";
        errMsg += std::string(cudaGetErrorString(cErr)) + "\n";
        throw std::runtime_error(errMsg);
    }
}

/**
 *
 */
void  CudaDataPacket::transferFromDeviceToHost(void) {
    // Bring data back to host.  Use asynchronous transfer so that we can keep
    // the transfer off the default stream and therefore only wait on this
    // transfer.
    cudaStream_t  stream = *(stream_.object);
    cudaError_t   cErr = cudaMemcpyAsync(packet_p_, packet_d_,
                                         N_BYTES_PER_PACKET,
                                         cudaMemcpyDeviceToHost,
                                         stream);
    if (cErr != cudaSuccess) {
        std::string  errMsg = "[CudaDataPacket::transferFromDeviceToHost] ";
        errMsg += "Unable to execute D-to-H transfer\n";
        errMsg += "CUDA error - " + std::string(cudaGetErrorName(cErr)) + "\n";
        errMsg += std::string(cudaGetErrorString(cErr)) + "\n";
        throw std::runtime_error(errMsg);
    }
    cudaStreamSynchronize(stream);

    unpack();
}

}

