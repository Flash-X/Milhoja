#ifndef USE_CUDA_BACKEND
#error "This file need not be compiled if the CUDA backend isn't used"
#endif

#include "CudaDataPacket.h"

#include <cassert>
#include <cstring>
#include <stdexcept>

#include "Grid_REAL.h"
#include "Grid_IntVect.h"
#include "Grid_RealVect.h"
#include "Grid_Axis.h"
#include "Grid_Edge.h"
#include "Grid.h"
#include "FArray1D.h"
#include "FArray4D.h"
#include "CudaStreamManager.h"
#include "CudaMemoryManager.h"

#include "Driver.h"

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
      tiles_{},
      nTiles_d_{nullptr},
      contents_p_{nullptr},
      contents_d_{nullptr},
      stream_{},
      nBytesPerPacket_{0},
      dt_d_{nullptr}
{
    std::string   errMsg = isNull();
    if (errMsg != "") {
        throw std::logic_error("[CudaDataPacket::CudaDataPacket] " + errMsg);
    }

    if (tiles_.size() != 0) {
        throw std::runtime_error("[CudaDataPacket::CudaDataPacket] tiles_ not empty");
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
    if (stream_.cudaStream != nullptr) {
        CudaStreamManager::instance().releaseStream(stream_);
        assert(stream_.cudaStream == nullptr);
    }

    CudaMemoryManager::instance().releaseMemory(&packet_p_, &packet_d_);
    assert(packet_p_ == nullptr);
    assert(packet_d_ == nullptr);

    location_ = PacketDataLocation::NOT_ASSIGNED;

    startVariable_ = UNK_VARS_BEGIN_C - 1;
    endVariable_   = UNK_VARS_BEGIN_C - 1;

    nBytesPerPacket_ = 0;

    nTiles_d_   = nullptr;
    contents_p_ = nullptr;
    contents_d_ = nullptr;
    dt_d_       = nullptr;
}

/**
 *
 */
std::string  CudaDataPacket::isNull(void) const {
    if (stream_.cudaStream != nullptr) {
        return "CUDA stream already acquired";
    } else if (packet_p_ != nullptr) {
        return "Pinned memory buffer has already been allocated";
    } else if (packet_d_ != nullptr) {
        return "Device memory buffer has already been allocated";
    } else if (location_ != PacketDataLocation::NOT_ASSIGNED) {
        return "Data location already assigned";
    } else if (nBytesPerPacket_ > 0) {
        return "Non-zero packet size";
    } else if (nTiles_d_ != nullptr) {
        return "N tiles exist in GPU";
    } else if (dt_d_ != nullptr) {
        return "dt already exists in GPU";
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
std::size_t   CudaDataPacket::nTiles(void) const {
    return tiles_.size();
}

/**
 *
 */
void   CudaDataPacket::addTile(std::shared_ptr<Tile>&& tileDesc) {
    tiles_.push_front( std::move(tileDesc) );
    if ((tileDesc != nullptr) || (tileDesc.use_count() != 0)) {
        throw std::runtime_error("[CudaDataPacket::addTile] Ownership of tileDesc not transferred");
    }
}

/**
 *
 */
std::shared_ptr<Tile>  CudaDataPacket::popTile(void) {
    if (tiles_.size() == 0) {
        throw std::invalid_argument("[CudaDataPacket::popTile] No tiles to pop");
    }

    std::shared_ptr<Tile>   tileDesc{ std::move(tiles_.front()) };
    if (   (tiles_.front() != nullptr)
        || (tiles_.front().use_count() != 0)) {
        throw std::runtime_error("[CudaDataPacket::popTile] Ownership of tileDesc not transferred");
    } 
    
    tiles_.pop_front();
    if ((tileDesc == nullptr) || (tileDesc.use_count() == 0)) {
        throw std::runtime_error("[CudaDataPacket::popTile] Bad tileDesc");
    }

    return tileDesc;
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
    } else if (tiles_.size() == 0) {
        throw std::logic_error("[CudaDataPacket::pack] No tiles added");
    }

    Grid&   grid = Grid::instance();

    // TODO: Deltas should just be placed into the packet once.
    std::size_t    nTiles = tiles_.size();
    nBytesPerPacket_ =            sizeof(std::size_t) 
                       +                        DRIVER_DT_SIZE_BYTES
                       + nTiles * sizeof(PacketContents)
                       + nTiles * (         1 * DELTA_SIZE_BYTES
                                   +        4 * POINT_SIZE_BYTES
                                   + N_BLOCKS * CC_BLOCK_SIZE_BYTES
                                   + N_BLOCKS * ARRAY4_SIZE_BYTES
#if NFLUXES > 0
                                   +            FCX_BLOCK_SIZE_BYTES
                                   +            FCY_BLOCK_SIZE_BYTES
                                   +            FCZ_BLOCK_SIZE_BYTES
                                   +        3 * ARRAY4_SIZE_BYTES
#endif
                                   +        1 * COORDS_X_SIZE_BYTES
                                   +        1 * COORDS_Y_SIZE_BYTES
                                   +        1 * COORDS_Z_SIZE_BYTES
                                   +        3 * ARRAY1_SIZE_BYTES);

    stream_ = CudaStreamManager::instance().requestStream(true);
    if (stream_.cudaStream == nullptr) {
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

    nTiles_d_ = static_cast<std::size_t*>((void*)ptr_d); 
    std::memcpy((void*)ptr_p, (void*)&nTiles, sizeof(std::size_t));
    ptr_p += sizeof(std::size_t);
    ptr_d += sizeof(std::size_t);
    
    dt_d_ = static_cast<Real*>((void*)ptr_d); 
    std::memcpy((void*)ptr_p, (void*)&Driver::dt, DRIVER_DT_SIZE_BYTES);
    ptr_p += sizeof(DRIVER_DT_SIZE_BYTES);
    ptr_d += sizeof(DRIVER_DT_SIZE_BYTES);

    contents_p_ = static_cast<PacketContents*>((void*)ptr_p);
    contents_d_ = static_cast<PacketContents*>((void*)ptr_d);
    ptr_p += nTiles * sizeof(PacketContents);
    ptr_d += nTiles * sizeof(PacketContents);

    PacketContents*   tilePtrs_p = contents_p_;
    for (std::size_t n=0; n<nTiles; ++n, ++tilePtrs_p) {
        Tile*   tileDesc_h = tiles_[n].get();
        if (tileDesc_h == nullptr) {
            throw std::runtime_error("[CudaDataPacket::pack] Bad tileDesc");
        }
 
        const unsigned int  level  = tileDesc_h->level();
        const RealVect      deltas = tileDesc_h->deltas();
        const IntVect       lo     = tileDesc_h->lo();
        const IntVect       hi     = tileDesc_h->hi();
        const IntVect       loGC   = tileDesc_h->loGC();
        const IntVect       hiGC   = tileDesc_h->hiGC();
        const FArray1D      xCoordsGC = grid.getCellCoords(Axis::I, Edge::Center,
                                                           level, loGC, hiGC); 
        const FArray1D      yCoordsGC = grid.getCellCoords(Axis::J, Edge::Center,
                                                           level, loGC, hiGC); 
        const FArray1D      zCoordsGC = grid.getCellCoords(Axis::K, Edge::Center,
                                                           level, loGC, hiGC); 
        const Real*         xCoordsGC_h = xCoordsGC.dataPtr();
        const Real*         yCoordsGC_h = yCoordsGC.dataPtr();
        const Real*         zCoordsGC_h = zCoordsGC.dataPtr();
        Real*               xCoordsGC_data_d = nullptr;
        Real*               yCoordsGC_data_d = nullptr;
        Real*               zCoordsGC_data_d = nullptr;
        Real*               data_h = tileDesc_h->dataPtr();
        Real*               CC1_data_d = nullptr;
        Real*               CC2_data_d = nullptr;
#if NFLUXES > 0
        Real*               FCX_data_d = nullptr;
        Real*               FCY_data_d = nullptr;
        Real*               FCZ_data_d = nullptr;
#endif
        if (data_h == nullptr) {
            throw std::logic_error("[CudaDataPacket::pack] "
                                   "Invalid pointer to data in host memory");
        }

        // TODO: I think that we should put in padding so that all objects 
        //       are byte aligned in the device's memory.
        tilePtrs_p->level = level;
        tilePtrs_p->deltas_d = static_cast<RealVect*>((void*)ptr_d);
        std::memcpy((void*)ptr_p, (void*)&deltas, DELTA_SIZE_BYTES);
        ptr_p += DELTA_SIZE_BYTES;
        ptr_d += DELTA_SIZE_BYTES;

        // Pack data for single tile data packet
        tilePtrs_p->lo_d = static_cast<IntVect*>((void*)ptr_d);
        std::memcpy((void*)ptr_p, (void*)&lo, POINT_SIZE_BYTES);
        ptr_p += POINT_SIZE_BYTES;
        ptr_d += POINT_SIZE_BYTES;

        tilePtrs_p->hi_d = static_cast<IntVect*>((void*)ptr_d);
        std::memcpy((void*)ptr_p, (void*)&hi, POINT_SIZE_BYTES);
        ptr_p += POINT_SIZE_BYTES;
        ptr_d += POINT_SIZE_BYTES;

        tilePtrs_p->loGC_d = static_cast<IntVect*>((void*)ptr_d);
        std::memcpy((void*)ptr_p, (void*)&loGC, POINT_SIZE_BYTES);
        ptr_p += POINT_SIZE_BYTES;
        ptr_d += POINT_SIZE_BYTES;

        tilePtrs_p->hiGC_d = static_cast<IntVect*>((void*)ptr_d);
        std::memcpy((void*)ptr_p, (void*)&hiGC, POINT_SIZE_BYTES);
        ptr_p += POINT_SIZE_BYTES;
        ptr_d += POINT_SIZE_BYTES;

        location_ = PacketDataLocation::CC1;
        tilePtrs_p->CC1_data_p = static_cast<Real*>((void*)ptr_p);
        CC1_data_d  = static_cast<Real*>((void*)ptr_d);
        std::memcpy((void*)ptr_p, (void*)data_h, CC_BLOCK_SIZE_BYTES);
        ptr_p += CC_BLOCK_SIZE_BYTES;
        ptr_d += CC_BLOCK_SIZE_BYTES;

        tilePtrs_p->CC2_data_p = static_cast<Real*>((void*)ptr_p);
        CC2_data_d  = static_cast<Real*>((void*)ptr_d);
        ptr_p += CC_BLOCK_SIZE_BYTES;
        ptr_d += CC_BLOCK_SIZE_BYTES;

        xCoordsGC_data_d = static_cast<Real*>((void*)ptr_d);
        std::memcpy((void*)ptr_p, (void*)xCoordsGC_h, COORDS_X_SIZE_BYTES);
        ptr_p += COORDS_X_SIZE_BYTES;
        ptr_d += COORDS_X_SIZE_BYTES;

        yCoordsGC_data_d = static_cast<Real*>((void*)ptr_d);
        std::memcpy((void*)ptr_p, (void*)yCoordsGC_h, COORDS_Y_SIZE_BYTES);
        ptr_p += COORDS_Y_SIZE_BYTES;
        ptr_d += COORDS_Y_SIZE_BYTES;

        zCoordsGC_data_d = static_cast<Real*>((void*)ptr_d);
        std::memcpy((void*)ptr_p, (void*)zCoordsGC_h, COORDS_Z_SIZE_BYTES);
        ptr_p += COORDS_Z_SIZE_BYTES;
        ptr_d += COORDS_Z_SIZE_BYTES;

        tilePtrs_p->xCoords_d = static_cast<FArray1D*>((void*)ptr_d);
        FArray1D   xCoordGCArray_d{xCoordsGC_data_d, loGC.I()};
        std::memcpy((void*)ptr_p, (void*)&xCoordGCArray_d, ARRAY1_SIZE_BYTES);
        ptr_p += ARRAY1_SIZE_BYTES;
        ptr_d += ARRAY1_SIZE_BYTES;

        tilePtrs_p->yCoords_d = static_cast<FArray1D*>((void*)ptr_d);
        FArray1D   yCoordGCArray_d{yCoordsGC_data_d, loGC.J()};
        std::memcpy((void*)ptr_p, (void*)&yCoordGCArray_d, ARRAY1_SIZE_BYTES);
        ptr_p += ARRAY1_SIZE_BYTES;
        ptr_d += ARRAY1_SIZE_BYTES;
 
        tilePtrs_p->zCoords_d = static_cast<FArray1D*>((void*)ptr_d);
        FArray1D   zCoordGCArray_d{zCoordsGC_data_d, loGC.K()};
        std::memcpy((void*)ptr_p, (void*)&zCoordGCArray_d, ARRAY1_SIZE_BYTES);
        ptr_p += ARRAY1_SIZE_BYTES;
        ptr_d += ARRAY1_SIZE_BYTES;
 
        // Create an FArray4D object in host memory but that already points
        // to where its data will be in device memory (i.e. the device object
        // will already be attached to its data in device memory).
        // The object in host memory should never be used then.
        // IMPORTANT: When this local object is destroyed, we don't want it to
        // affect the use of the copies (e.g. release memory).
        tilePtrs_p->CC1_d = static_cast<FArray4D*>((void*)ptr_d);
        FArray4D   CC1_d{CC1_data_d, loGC, hiGC, NUNKVAR};
        std::memcpy((void*)ptr_p, (void*)&CC1_d, ARRAY4_SIZE_BYTES);
        ptr_p += ARRAY4_SIZE_BYTES;
        ptr_d += ARRAY4_SIZE_BYTES;

        tilePtrs_p->CC2_d = static_cast<FArray4D*>((void*)ptr_d);
        FArray4D   CC2_d{CC2_data_d, loGC, hiGC, NUNKVAR};
        std::memcpy((void*)ptr_p, (void*)&CC2_d, ARRAY4_SIZE_BYTES);
        ptr_p += ARRAY4_SIZE_BYTES;
        ptr_d += ARRAY4_SIZE_BYTES;

#if NFLUXES > 0
        FCX_data_d  = static_cast<Real*>((void*)ptr_d);
        ptr_p += FCX_BLOCK_SIZE_BYTES;
        ptr_d += FCX_BLOCK_SIZE_BYTES;

        FCY_data_d  = static_cast<Real*>((void*)ptr_d);
        ptr_p += FCY_BLOCK_SIZE_BYTES;
        ptr_d += FCY_BLOCK_SIZE_BYTES;

        FCZ_data_d  = static_cast<Real*>((void*)ptr_d);
        ptr_p += FCZ_BLOCK_SIZE_BYTES;
        ptr_d += FCZ_BLOCK_SIZE_BYTES;

        tilePtrs_p->FCX_d = static_cast<FArray4D*>((void*)ptr_d);
        IntVect    fHi = IntVect{LIST_NDIM(hi.I()+1, hi.J(), hi.K())};
        FArray4D   FCX_d{FCX_data_d, lo, fHi, NFLUXES};
        std::memcpy((void*)ptr_p, (void*)&FCX_d, ARRAY4_SIZE_BYTES);
        ptr_p += ARRAY4_SIZE_BYTES;
        ptr_d += ARRAY4_SIZE_BYTES;

        tilePtrs_p->FCY_d = static_cast<FArray4D*>((void*)ptr_d);
        fHi = IntVect{LIST_NDIM(hi.I(), hi.J()+1, hi.K())};
        FArray4D   FCY_d{FCY_data_d, lo, fHi, NFLUXES};
        std::memcpy((void*)ptr_p, (void*)&FCY_d, ARRAY4_SIZE_BYTES);
        ptr_p += ARRAY4_SIZE_BYTES;
        ptr_d += ARRAY4_SIZE_BYTES;

        // TODO: Is is necessary to always have the correct number of faces in
        // these flux arrays for dimensions above NDIM?  Is it even necessary to
        // make flux arrays available above NDIM?
        tilePtrs_p->FCZ_d = static_cast<FArray4D*>((void*)ptr_d);
        fHi = IntVect{LIST_NDIM(hi.I(), hi.J(), hi.K()+1)};
        FArray4D   FCZ_d{FCZ_data_d, lo, fHi, NFLUXES};
        std::memcpy((void*)ptr_p, (void*)&FCZ_d, ARRAY4_SIZE_BYTES);
        ptr_p += ARRAY4_SIZE_BYTES;
        ptr_d += ARRAY4_SIZE_BYTES;
#endif
    }

    // TODO: Use pointers to determine size of packet and compare against
    // nBytesPerPacket
}

/**
 *
 */
void  CudaDataPacket::unpack(void) {
    if (tiles_.size() <= 0) {
        throw std::logic_error("[CudaDataPacket::unpack] "
                               "Empty data packet");
    } else if (packet_p_ == nullptr) {
        throw std::logic_error("[CudaDataPacket::unpack] "
                               "No pointer to start of packet in pinned memory");
    } else if (packet_d_ == nullptr) {
        throw std::logic_error("[CudaDataPacket::unpack] "
                               "No pointer to start of packet in GPU memory");
    } else if (stream_.cudaStream == nullptr) {
        throw std::logic_error("[CudaDataPacket::unpack] "
                               "CUDA stream not acquired");
    } else if (contents_p_ == nullptr) {
        throw std::logic_error("[CudaDataPacket::unpack] "
                               "No pinned packet contents");
    } else if (   (startVariable_ < UNK_VARS_BEGIN_C )
               || (startVariable_ > UNK_VARS_END_C )
               || (endVariable_   < UNK_VARS_BEGIN_C )
               || (endVariable_   > UNK_VARS_END_C)) {
        throw std::logic_error("[CudaDataPacket::unpack] "
                               "Invalid variable mask");
    }

    // Release stream as soon as possible
    CudaStreamManager::instance().releaseStream(stream_);
    assert(stream_.cudaStream == nullptr);

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
        std::size_t  offset =   N_ELEMENTS_PER_CC_PER_VARIABLE
                              * static_cast<std::size_t>(startVariable_);
        Real*              start_h = data_h + offset;
        const Real*        start_p = data_p + offset;
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

/**
 *
 */
void  CudaDataPacket::initiateHostToDeviceTransfer(void) {
    pack();

    cudaError_t cErr = cudaMemcpyAsync(packet_d_, packet_p_,
                                       nBytesPerPacket_,
                                       cudaMemcpyHostToDevice,
                                       stream_.cudaStream);
    if (cErr != cudaSuccess) {
        std::string  errMsg = "[CudaDataPacket::initiateHostToDeviceTransfer] ";
        errMsg += "Unable to initiate H-to-D transfer\n";
        errMsg += "CUDA error - " + std::string(cudaGetErrorName(cErr)) + "\n";
        errMsg += std::string(cudaGetErrorString(cErr)) + "\n";
        throw std::runtime_error(errMsg);
    }
}

/**
 *  \todo Determine how to implement this as an asynchronous transfer.
 */
void  CudaDataPacket::transferFromDeviceToHost(void) {
    // Bring data back to host.  Use asynchronous transfer so that we can keep
    // the transfer off the default stream and therefore only wait on this
    // transfer.
    cudaError_t   cErr = cudaMemcpyAsync(packet_p_, packet_d_,
                                         nBytesPerPacket_,
                                         cudaMemcpyDeviceToHost,
                                         stream_.cudaStream);
    if (cErr != cudaSuccess) {
        std::string  errMsg = "[CudaDataPacket::transferFromDeviceToHost] ";
        errMsg += "Unable to execute D-to-H transfer\n";
        errMsg += "CUDA error - " + std::string(cudaGetErrorName(cErr)) + "\n";
        errMsg += std::string(cudaGetErrorString(cErr)) + "\n";
        throw std::runtime_error(errMsg);
    }
    cudaStreamSynchronize(stream_.cudaStream);

    unpack();
}

}

