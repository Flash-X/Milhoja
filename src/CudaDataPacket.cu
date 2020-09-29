// WIP: Somehow NDEBUG is getting set and deactivating the asserts
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>

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
CudaDataPacket::CudaDataPacket(void)
    : DataPacket{},
      location_{PacketDataLocation::NOT_ASSIGNED},
      startVariable_{UNK_VARS_BEGIN_C - 1},
      endVariable_{UNK_VARS_BEGIN_C - 1},
      packet_p_{nullptr},
      packet_d_{nullptr},
      contents_{},
      stream_{},
      nBytesPerPacket_{0}
{
    std::string   errMsg = isNull();
    if (errMsg != "") {
        throw std::logic_error("[CudaDataPacket::CudaDataPacket] " + errMsg);
    }

    if (contents_.size() != 0) {
        throw std::runtime_error("[CudaDataPacket::CudaDataPacket] contents_ not empty");
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

    location_ = PacketDataLocation::NOT_ASSIGNED;

    startVariable_ = UNK_VARS_BEGIN_C - 1;
    endVariable_   = UNK_VARS_BEGIN_C - 1;

    nBytesPerPacket_ = 0;
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
    } else if (location_ != PacketDataLocation::NOT_ASSIGNED) {
        return "Data location already assigned";
    } else if (nBytesPerPacket_ > 0) {
        return "Non-zero packet size";
    }

    return "";
}

/**
 *
 */
std::size_t   CudaDataPacket::nTiles(void) const {
    return contents_.size();

}

/**
 *
 */
void   CudaDataPacket::addTile(std::shared_ptr<Tile>&& tileDesc) {
    contents_.push_front( PacketContents() );
    contents_.front().tileDesc_h = std::move(tileDesc);
    if ((tileDesc != nullptr) || (tileDesc.use_count() != 0)) {
        throw std::runtime_error("[CudaDataPacket::addTile] Ownership of tileDesc not transferred");
    }
}

/**
 *
 */
std::shared_ptr<Tile>  CudaDataPacket::popTile(void) {
    if (contents_.size() == 0) {
        throw std::invalid_argument("[CudaDataPacket::popTile] No tiles to pop");
    }

    std::shared_ptr<Tile>   tileDesc{ std::move(contents_.front().tileDesc_h) };
    if (   (contents_.front().tileDesc_h != nullptr)
        || (contents_.front().tileDesc_h.use_count() != 0)) {
        throw std::runtime_error("[CudaDataPacket::popTile] Ownership of tileDesc not transferred");
    } 
    
    contents_.pop_front();
    if ((tileDesc == nullptr) || (tileDesc.use_count() == 0)) {
        throw std::runtime_error("[CudaDataPacket::popTile] Bad tileDesc");
    }

    return tileDesc;
}

/**
 *
 */
const PacketContents&   CudaDataPacket::tilePointers(const std::size_t n) const {
#ifdef DEBUG_RUNTIME
    return contents_.at(n);
#else
    return contents_[n];
#endif
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
void   CudaDataPacket::setVariableMask(const int startVariable,
                                       const int endVariable) {
    if        (startVariable < UNK_VARS_BEGIN_C) {
        throw std::logic_error("[CudaDataPacket::setVariableMask] "
                               "Starting variable is invalid");
    } else if (endVariable > UNK_VARS_END_C) {
        throw std::logic_error("[CudaDataPacket::setVariableMask] "
                               "Ending variable is invalid");
    } else if (startVariable > endVariable) {
        throw std::logic_error("[CudaDataPacket::setVariableMask] "
                               "Starting variable > ending variable");
    }

    startVariable_ = startVariable;
    endVariable_ = endVariable;
}

/**
 *
 */
void  CudaDataPacket::pack(void) {
    std::string   errMsg = isNull();
    if (errMsg != "") {
        throw std::logic_error("[CudaDataPacket::pack] " + errMsg);
    } else if (contents_.size() == 0) {
        throw std::logic_error("[CudaDataPacket::pack] No tiles added");
    }

    Grid&   grid = Grid::instance();

    // TODO: Deltas should just be placed into the packet once.
    std::size_t    nTiles = contents_.size();
    nBytesPerPacket_ =   nTiles * (         1 * DELTA_SIZE_BYTES
                                   +        4 * POINT_SIZE_BYTES
                                   + N_BLOCKS * BLOCK_SIZE_BYTES
                                   +        2 * ARRAY4_SIZE_BYTES
                                   +        1 * COORDS_X_SIZE_BYTES
                                   +        1 * COORDS_Y_SIZE_BYTES
                                   +        2 * ARRAY1_SIZE_BYTES);

    // For the present purpose of development, fail if no streams available
    stream_ = CudaStreamManager::instance().requestStream(true);
    if ((stream_.object == nullptr) || (stream_.id == CudaStream::NULL_STREAM_ID)) {
        throw std::runtime_error("[CudaDataPacket::pack] Unable to acquire stream");
    }

    // Allocate memory in pinned and device memory on demand for now
    CudaMemoryManager::instance().requestMemory(nBytesPerPacket_,
                                                &packet_p_,
                                                &packet_d_);

    // Pointer to the next free byte in the current data packets
    // Should be true by C++ standard
    static_assert(sizeof(char) == 1, "Invalid char size");
    char*   ptr_p = static_cast<char*>(packet_p_);
    char*   ptr_d = static_cast<char*>(packet_d_);
    for (std::size_t n=0; n<contents_.size(); ++n) {
        PacketContents&   tilePtrs = contents_[n];
        if ((tilePtrs.tileDesc_h == nullptr) || (tilePtrs.tileDesc_h.use_count() == 0)) {
            throw std::runtime_error("[CudaDataPacket::pack] Bad tileDesc");
        }
 
        const unsigned int  level  = tilePtrs.tileDesc_h->level();
        const RealVect      deltas = tilePtrs.tileDesc_h->deltas();
        const IntVect       lo     = tilePtrs.tileDesc_h->lo();
        const IntVect       hi     = tilePtrs.tileDesc_h->hi();
        const IntVect       loGC   = tilePtrs.tileDesc_h->loGC();
        const IntVect       hiGC   = tilePtrs.tileDesc_h->hiGC();
        const FArray1D      xCoordsGC = grid.getCellCoords(Axis::I, Edge::Center,
                                                           level, loGC, hiGC); 
        const FArray1D      yCoordsGC = grid.getCellCoords(Axis::J, Edge::Center,
                                                           level, loGC, hiGC); 
        const Real*         xCoordsGC_h = xCoordsGC.dataPtr();
        const Real*         yCoordsGC_h = yCoordsGC.dataPtr();
        Real*               xCoordsGC_data_d = nullptr;
        Real*               yCoordsGC_data_d = nullptr;
        Real*               data_h = tilePtrs.tileDesc_h->dataPtr();
        Real*               CC1_data_d = nullptr;
        Real*               CC2_data_d = nullptr;
        if (data_h == nullptr) {
            throw std::logic_error("[CudaDataPacket::pack] "
                                   "Invalid pointer to data in host memory");
        }

        // TODO: I think that we should put in padding so that all objects 
        //       are byte aligned in the device's memory.
        tilePtrs.level = level;
        tilePtrs.deltas_d = static_cast<RealVect*>((void*)ptr_d);
        std::memcpy((void*)ptr_p, (void*)&deltas, DELTA_SIZE_BYTES);
        ptr_p += DELTA_SIZE_BYTES;
        ptr_d += DELTA_SIZE_BYTES;

        // Pack data for single tile data packet
        tilePtrs.lo_d = static_cast<IntVect*>((void*)ptr_d);
        std::memcpy((void*)ptr_p, (void*)&lo, POINT_SIZE_BYTES);
        ptr_p += POINT_SIZE_BYTES;
        ptr_d += POINT_SIZE_BYTES;

        tilePtrs.hi_d = static_cast<IntVect*>((void*)ptr_d);
        std::memcpy((void*)ptr_p, (void*)&hi, POINT_SIZE_BYTES);
        ptr_p += POINT_SIZE_BYTES;
        ptr_d += POINT_SIZE_BYTES;

        tilePtrs.loGC_d = static_cast<IntVect*>((void*)ptr_d);
        std::memcpy((void*)ptr_p, (void*)&loGC, POINT_SIZE_BYTES);
        ptr_p += POINT_SIZE_BYTES;
        ptr_d += POINT_SIZE_BYTES;

        tilePtrs.hiGC_d = static_cast<IntVect*>((void*)ptr_d);
        std::memcpy((void*)ptr_p, (void*)&hiGC, POINT_SIZE_BYTES);
        ptr_p += POINT_SIZE_BYTES;
        ptr_d += POINT_SIZE_BYTES;

        location_ = PacketDataLocation::CC1;
        tilePtrs.CC1_data_p = static_cast<Real*>((void*)ptr_p);
        CC1_data_d  = static_cast<Real*>((void*)ptr_d);
        std::memcpy((void*)ptr_p, (void*)data_h, BLOCK_SIZE_BYTES);
        ptr_p += BLOCK_SIZE_BYTES;
        ptr_d += BLOCK_SIZE_BYTES;

        tilePtrs.CC2_data_p = static_cast<Real*>((void*)ptr_p);
        CC2_data_d  = static_cast<Real*>((void*)ptr_d);
        ptr_p += BLOCK_SIZE_BYTES;
        ptr_d += BLOCK_SIZE_BYTES;

        xCoordsGC_data_d = static_cast<Real*>((void*)ptr_d);
        std::memcpy((void*)ptr_p, (void*)xCoordsGC_h, COORDS_X_SIZE_BYTES);
        ptr_p += COORDS_X_SIZE_BYTES;
        ptr_d += COORDS_X_SIZE_BYTES;

        yCoordsGC_data_d = static_cast<Real*>((void*)ptr_d);
        std::memcpy((void*)ptr_p, (void*)yCoordsGC_h, COORDS_Y_SIZE_BYTES);
        ptr_p += COORDS_Y_SIZE_BYTES;
        ptr_d += COORDS_Y_SIZE_BYTES;

        tilePtrs.xCoords_d = static_cast<FArray1D*>((void*)ptr_d);
        FArray1D   xCoordGCArray_d{xCoordsGC_data_d, loGC.I()};
        std::memcpy((void*)ptr_p, (void*)&xCoordGCArray_d, ARRAY1_SIZE_BYTES);
        ptr_p += ARRAY1_SIZE_BYTES;
        ptr_d += ARRAY1_SIZE_BYTES;

        tilePtrs.yCoords_d = static_cast<FArray1D*>((void*)ptr_d);
        FArray1D   yCoordGCArray_d{yCoordsGC_data_d, loGC.J()};
        std::memcpy((void*)ptr_p, (void*)&yCoordGCArray_d, ARRAY1_SIZE_BYTES);
        ptr_p += ARRAY1_SIZE_BYTES;
        ptr_d += ARRAY1_SIZE_BYTES;
 
        // Create an FArray4D object in host memory but that already points
        // to where its data will be in device memory (i.e. the device object
        // will already be attached to its data in device memory).
        // The object in host memory should never be used then.
        // IMPORTANT: When this local object is destroyed, we don't want it to
        // affect the use of the copies (e.g. release memory).
        tilePtrs.CC1_d = static_cast<FArray4D*>((void*)ptr_d);
        FArray4D   CC1_d{CC1_data_d, loGC, hiGC, NUNKVAR};
        std::memcpy((void*)ptr_p, (void*)&CC1_d, ARRAY4_SIZE_BYTES);
        ptr_p += ARRAY4_SIZE_BYTES;
        ptr_d += ARRAY4_SIZE_BYTES;

        tilePtrs.CC2_d = static_cast<FArray4D*>((void*)ptr_d);
        FArray4D   CC2_d{CC2_data_d, loGC, hiGC, NUNKVAR};
        std::memcpy((void*)ptr_p, (void*)&CC2_d, ARRAY4_SIZE_BYTES);
        ptr_p += ARRAY4_SIZE_BYTES;
        ptr_d += ARRAY4_SIZE_BYTES;
    }

    // TODO: Use pointers to determine size of packet and compare against
    // nBytesPerPacket
}

/**
 *
 */
void  CudaDataPacket::unpack(void) {
    if (contents_.size() <= 0) {
        throw std::logic_error("[CudaDataPacket::unpack] "
                               "Empty data packet");
    } else if (packet_p_ == nullptr) {
        throw std::logic_error("[CudaDataPacket::unpack] "
                               "No pointer to start of packet in pinned memory");
    } else if (packet_d_ == nullptr) {
        throw std::logic_error("[CudaDataPacket::unpack] "
                               "No pointer to start of packet in GPU memory");
    } else if ((stream_.object == nullptr) || (stream_.id == CudaStream::NULL_STREAM_ID)) {
        throw std::logic_error("[CudaDataPacket::unpack] "
                               "CUDA stream not acquired");
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

    for (std::size_t n=0; n<contents_.size(); ++n) {
        const PacketContents&   tilePtrs = contents_[n];

        Real*   data_h = tilePtrs.tileDesc_h->dataPtr();
        Real*   data_p = nullptr;
        switch (location_) {
            case PacketDataLocation::CC1:
                data_p = tilePtrs.CC1_data_p;
                break;
            case PacketDataLocation::CC2:
                data_p = tilePtrs.CC2_data_p;
                break;
            default:
                throw std::logic_error("[CudaDataPacket::unpack] Data not in CC1 or CC2");
        }

        if (data_h == nullptr) {
            throw std::logic_error("[CudaDataPacket::unpack] "
                                   "Invalid pointer to data in host memory");
        } else if (data_p == nullptr) {
            throw std::runtime_error("[CudaDataPacket::unpack] "
                                     "Invalid pointer to data in pinned memory");
        }

        // The code here imposes requirements on the variable indices.  See Flash.h
        // for more information.  If this code is changed, please make sure to
        // adjust Flash.h appropriately.
        assert(UNK_VARS_BEGIN_C == 0);
        assert(UNK_VARS_END_C == (NUNKVAR - 1));
        std::size_t  offset =   N_ELEMENTS_PER_BLOCK_PER_VARIABLE
                              * static_cast<std::size_t>(startVariable_);
        Real*        start_h = data_h + offset;
        Real*        start_p = data_p + offset;
        std::size_t  nBytes =  (endVariable_ - startVariable_ + 1)
                              * N_ELEMENTS_PER_BLOCK_PER_VARIABLE
                              * sizeof(Real);
        std::memcpy((void*)start_h, (void*)start_p, nBytes);
    }

    // The packet is consumed upon unpacking.  However, we still keep the
    // contents intact so that runtime elements such as MoverUnpacker can 
    // enqueue the tiles with its data subscriber.
    nullify();
}

/**
 *
 */
void  CudaDataPacket::initiateHostToDeviceTransfer(void) {
    pack();

    cudaStream_t  stream = *(stream_.object);
    cudaError_t cErr = cudaMemcpyAsync(packet_d_, packet_p_,
                                       nBytesPerPacket_,
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
                                         nBytesPerPacket_,
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

