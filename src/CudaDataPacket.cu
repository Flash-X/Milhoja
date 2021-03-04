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
      buffer_p_{nullptr},
      buffer_d_{nullptr},
      packet_p_{nullptr},
      packet_d_{nullptr},
      tiles_{},
      nTiles_d_{nullptr},
      contents_p_{nullptr},
      contents_d_{nullptr},
      stream_{},
      streamsExtra_{nullptr},
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

    if (streamsExtra_) {
        // FIXME: Hardcoded size!
        for (unsigned int i=0; i<2; ++i) {
            if (streamsExtra_[i].cudaStream != nullptr) {
                CudaStreamManager::instance().releaseStream(streamsExtra_[i]);
                assert(streamsExtra_[i].cudaStream == nullptr);
            }
        }

        delete [] streamsExtra_;
        streamsExtra_ = nullptr;
    }

    CudaMemoryManager::instance().releaseMemory(&buffer_p_, &buffer_d_);
    assert(buffer_p_ == nullptr);
    assert(buffer_d_ == nullptr);
    packet_p_ = nullptr;
    packet_d_ = nullptr;

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
    } else if (streamsExtra_ != nullptr) {
        return "Extra stream already acquired";
    } else if (buffer_p_ != nullptr) {
        return "Pinned memory buffer has already been allocated";
    } else if (buffer_d_ != nullptr) {
        return "Device memory buffer has already been allocated";
    } else if (packet_p_ != nullptr) {
        return "Pinned packet already exists";
    } else if (packet_d_ != nullptr) {
        return "Device packet already exists";
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
                       + nTiles * (  1 * DELTA_SIZE_BYTES
                                   + 4 * POINT_SIZE_BYTES
                                   + 1 * CC_BLOCK_SIZE_BYTES
#if NFLUXES == 0
                                   + 2 * ARRAY4_SIZE_BYTES);
#else
                                   + 5 * ARRAY4_SIZE_BYTES);
#endif
    std::size_t    bufferSizeBytes =            nBytesPerPacket_
                                     + nTiles * BUFFER_SIZE_PER_TILE;

    stream_ = CudaStreamManager::instance().requestStream(true);
    if (stream_.cudaStream == nullptr) {
        throw std::runtime_error("[CudaDataPacket::pack] Unable to acquire stream");
    }

    // TODO: This is an ugly workaround for now.  If we want eager acquisition
    // of stream resources in order to prevent possible deadlocks on
    // dynamic requesting of streams, then this needs to be made more
    // configurable.
#if NDIM == 3
    // FIXME: Hardcoded size!
    streamsExtra_ = new Stream[2];
    for (unsigned int i=0; i<2; ++i) {
        streamsExtra_[i] = CudaStreamManager::instance().requestStream(true);
        if (streamsExtra_[i].cudaStream == nullptr) {
            throw std::runtime_error("[CudaDataPacket::pack] Unable to acquire extra stream");
        }
    }
#endif

    // Allocate memory in pinned and device memory on demand for now
    CudaMemoryManager::instance().requestMemory(bufferSizeBytes,
                                                &buffer_p_,
                                                &buffer_d_);

    // Scratch data allocated at top of allocated memory block.
    // Maintain a pointer to the next bit of scratch memory available.
    char*   scratch_d = static_cast<char*>(buffer_d_);

    // Set pointers to start of data packet.
    // TODO: Note that we can request less pinned memory.
    packet_p_ = buffer_p_;
    packet_d_ = static_cast<void*>(scratch_d + BUFFER_SIZE_PER_TILE * nTiles);

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

        // FIXME: For Sedov, we aren't copying CC2 data back;
        tilePtrs_p->CC2_data_p = nullptr;
        CC2_data_d  = static_cast<Real*>((void*)scratch_d);
        scratch_d += CC_BLOCK_SIZE_BYTES;

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
        FCX_data_d  = static_cast<Real*>((void*)scratch_d);
        scratch_d += FCX_BLOCK_SIZE_BYTES;

        FCY_data_d  = static_cast<Real*>((void*)scratch_d);
        scratch_d += FCY_BLOCK_SIZE_BYTES;

        FCZ_data_d  = static_cast<Real*>((void*)scratch_d);
        scratch_d += FCZ_BLOCK_SIZE_BYTES;

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

    if (streamsExtra_) {
        // FIXME: Hardcoded size!
        for (unsigned int i=0; i<2; ++i) {
            if (streamsExtra_[i].cudaStream != nullptr) {
                CudaStreamManager::instance().releaseStream(streamsExtra_[i]);
                assert(streamsExtra_[i].cudaStream == nullptr);
            }
        }

        delete [] streamsExtra_;
        streamsExtra_ = nullptr;
    }

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
 *  Initiate an asychronous transfer of the packet from the device to the host
 *  on the packet's stream.  As part of this, launch on the same stream the given
 *  callback for handling the unpacking and other auxiliary work that must occur
 *  once the packet is back in pinned memory.
 *
 * \param  callback - the routine that will be registered with the CUDA runtime
 *                    so that the routine can unpack the packet (likely using
 *                    unpack) and perform other desired actions.
 * \param  callbackData - the data that must be passed to the callback so that
 *                        it can carry out its work.  This resource just passes
 *                        through this routine so that this routine has no
 *                        responsibility in managing the resources.
 */
void  CudaDataPacket::initiateDeviceToHostTransfer(cudaHostFn_t callback,
                                                   void* callbackData) {
    cudaError_t   cErr = cudaMemcpyAsync(packet_p_, packet_d_,
                                         nBytesPerPacket_,
                                         cudaMemcpyDeviceToHost,
                                         stream_.cudaStream);
    if (cErr != cudaSuccess) {
        std::string  errMsg = "[CudaDataPacket::initiateDeviceToHostTransfer] ";
        errMsg += "Unable to initiate D-to-H transfer\n";
        errMsg += "CUDA error - " + std::string(cudaGetErrorName(cErr)) + "\n";
        errMsg += std::string(cudaGetErrorString(cErr)) + "\n";
        throw std::runtime_error(errMsg);
    }

    cErr = cudaLaunchHostFunc(stream_.cudaStream, callback, callbackData); 
    if (cErr != cudaSuccess) {
        std::string  errMsg = "[CudaDataPacket::initiateDeviceToHostTransfer] ";
        errMsg += "Unable to register D-to-H callback function\n";
        errMsg += "CUDA error - " + std::string(cudaGetErrorName(cErr)) + "\n";
        errMsg += std::string(cudaGetErrorString(cErr)) + "\n";
        throw std::runtime_error(errMsg);
    }
}

}

