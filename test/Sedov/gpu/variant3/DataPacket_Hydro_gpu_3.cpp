#include "DataPacket_Hydro_gpu_3.h"

#include <cassert>
#include <cstring>
#include <stdexcept>

#include "Grid_IntVect.h"
#include "Grid_RealVect.h"
#include "Grid.h"
#include "FArray4D.h"
#include "Backend.h"

#include "Driver.h"

#include "Flash.h"

#if NFLUXES <= 0
#error "Sedov problem should include fluxes"
#endif

namespace orchestration {

/**
 * Construct a DataPacket containing no Tile objects and with no resources
 * assigned to it.
 */
DataPacket_Hydro_gpu_3::DataPacket_Hydro_gpu_3(void)
    : DataPacket{},
#if NDIM == 3
      stream2_{},
      stream3_{},
#endif
      nTiles_h_{0},
      nTiles_p_{nullptr},
      nTiles_d_{nullptr},
      dt_h_{-1.0},
      dt_p_{nullptr},
      dt_d_{nullptr},
      deltas_start_p_{nullptr},
      deltas_start_d_{nullptr},
      lo_start_p_{nullptr},
      lo_start_d_{nullptr},
      hi_start_p_{nullptr},
      hi_start_d_{nullptr}
{
}

/**
 * Destroy DataPacket.  Under normal circumstances, the DataPacket should have
 * been consumed and therefore own no resources.
 */
DataPacket_Hydro_gpu_3::~DataPacket_Hydro_gpu_3(void) {
#if NDIM == 3
    if (stream2_.isValid() || stream3_.isValid()) {
        throw std::logic_error("[DataPacket_Hydro_gpu_3::~DataPacket_Hydro_gpu_3] "
                               "One or more extra streams not released");
    }
#endif
}

/**
 *
 */
std::unique_ptr<DataPacket>   DataPacket_Hydro_gpu_3::clone(void) const {
    return std::unique_ptr<DataPacket>{new DataPacket_Hydro_gpu_3{}};
}

/**
 *
 */
int   DataPacket_Hydro_gpu_3::nTiles_host(void) const {
    return nTiles_h_;
}

/**
 *
 */
int*  DataPacket_Hydro_gpu_3::nTiles_devptr(void) const {
    return static_cast<int*>(nTiles_d_);
}

/**
 *
 */
Real   DataPacket_Hydro_gpu_3::dt_host(void) const {
    return dt_h_;
}

/**
 *
 */
Real*  DataPacket_Hydro_gpu_3::dt_devptr(void) const {
    return static_cast<Real*>(dt_d_);
}

/**
 *
 */
Real*  DataPacket_Hydro_gpu_3::deltas_devptr(void) const {
    return static_cast<Real*>(deltas_start_d_);
}

/**
 *
 */
int*   DataPacket_Hydro_gpu_3::lo_devptr(void) const {
    return static_cast<int*>(lo_start_d_);
}

/**
 *
 */
int*   DataPacket_Hydro_gpu_3::hi_devptr(void) const {
    return static_cast<int*>(hi_start_d_);
}

#if NDIM == 3 && defined(ENABLE_OPENACC_OFFLOAD)
/**
 * Refer to the documentation of this member function for DataPacket.
 */
void  DataPacket_Hydro_gpu_3::releaseExtraQueue(const unsigned int id) {
    if        (id == 2) {
        if (!stream2_.isValid()) {
            throw std::logic_error("[DataPacket_Hydro_gpu_3::releaseExtraQueue] "
                                   "Second queue invalid or already released");
        } else {
            Backend::instance().releaseStream(stream2_);
        }
    } else if (id == 3) {
        if (!stream3_.isValid()) {
            throw std::logic_error("[DataPacket_Hydro_gpu_3::releaseExtraQueue] "
                                   "Third queue invalid or already released");
        } else {
            Backend::instance().releaseStream(stream3_);
        }
    } else {
        throw std::invalid_argument("[DataPacket_Hydro_gpu_3::releaseExtraQueue] "
                                    "Invalid id");
    }
}
#endif

#if NDIM == 3 && defined(ENABLE_OPENACC_OFFLOAD)
/**
 * Refer to the documentation of this member function for DataPacket.
 */
int  DataPacket_Hydro_gpu_3::extraAsynchronousQueue(const unsigned int id) {
    if        (id == 2) {
        if (!stream2_.isValid()) {
            throw std::logic_error("[DataPacket_Hydro_gpu_3::extraAsynchronousQueue] "
                                   "Second queue invalid");
        } else {
            return stream2_.accAsyncQueue;
        }
    } else if (id == 3) {
        if (!stream3_.isValid()) {
            throw std::logic_error("[DataPacket_Hydro_gpu_3::extraAsynchronousQueue] "
                                   "Third queue invalid");
        } else {
            return stream3_.accAsyncQueue;
        }
    } else {
        throw std::invalid_argument("[DataPacket_Hydro_gpu_3::extraAsynchronousQueue] Invalid id");
    }
}
#endif

/**
 *
 */
void  DataPacket_Hydro_gpu_3::pack(void) {
    std::string   errMsg = isNull();
    if (errMsg != "") {
        throw std::logic_error("[DataPacket_Hydro_gpu_3::pack] " + errMsg);
    } else if (tiles_.size() == 0) {
        throw std::logic_error("[DataPacket_Hydro_gpu_3::pack] No tiles added");
    }

    Grid&   grid = Grid::instance();

    //----- SCRATCH SECTION
    // First block of memory should be allocated as acratch space
    // for use by GPU.  No need for this to be transferred at all.
    // 
    // GAMC is read from, but not written to by the task function that uses this
    // packet.  In addition, GAME is neither read from nor written to.
    // Therefore, GAME need not be included in the packet.  As part of this,
    // the variable order in memory was setup so that GAME is last.
    //
    // The CC scratch (a.k.a. CC2) should just be 3D.  However, the GPU version
    // of the static Fortran routine hy::computeSoundSpeedHll_oacc_summit, which
    // is used by all three task function variants is written for a 4D block
    // that uses only the first variable.  This is necessary for the second
    // version, where CC2 is used first as scratch and then as the storage block
    // for the updated solution.  When this file is generated by the code generator,
    // it should use the 3D version.
//    std::size_t  nCc1Variables = NUNKVAR - 1;
//    std::size_t  nCc2Variables = 1;
//    std::size_t  cc1BlockSizeBytes =   nCc1Variables
//                                     * N_ELEMENTS_PER_CC_PER_VARIABLE
//                                     * sizeof(Real);
//    std::size_t  cc2BlockSizeBytes =   nCc2Variables
//                                     * N_ELEMENTS_PER_CC_PER_VARIABLE
//                                     * sizeof(Real);
//
//    unsigned int nScratchArrays = 2;
//    std::size_t  nScratchPerTileBytes  =  cc2BlockSizeBytes
//                                        + FCX_BLOCK_SIZE_BYTES;
//#if NDIM >= 2
//    nScratchPerTileBytes += FCY_BLOCK_SIZE_BYTES;
//    ++nScratchArrays;
//#endif
//#if NDIM == 3
//    nScratchPerTileBytes += FCZ_BLOCK_SIZE_BYTES;
//    ++nScratchArrays;
//#endif

    //----- OBTAIN NON-TILE-SPECIFIC HOST-SIDE DATA
    // TODO: Check for correctness of cast here of elsewhere?
    // This cast is necessary since Fortran code consumes the packet.
    nTiles_h_ = static_cast<int>(tiles_.size());
    dt_h_     = Driver::dt;

    //----- COMPUTE SIZES/OFFSETS & ACQUIRE MEMORY
    std::size_t    sz_nTiles = sizeof(int);
    std::size_t    sz_dt     = sizeof(Real);

    std::size_t    sz_deltas = MDIM * sizeof(Real);
    std::size_t    sz_lo     = MDIM * sizeof(int);
    std::size_t    sz_hi     = MDIM * sizeof(int);
 
    std::size_t    nCopyInBytes =   sz_nTiles
                                  + sz_dt
                                  + nTiles_h_
                                    * (  sz_deltas
                                       + sz_lo
                                       + sz_hi);

    nCopyToGpuBytes_    = nCopyInBytes;
    nReturnToHostBytes_ = 0;
    std::size_t  nBytesPerPacket = nCopyInBytes;

    // ACQUIRE PINNED AND GPU MEMORY & SPECIFY STRUCTURE
    // Scratch only needed on GPU side
    // At present, this call makes certain that each data packet is
    // appropriately byte aligned.
    Backend::instance().requestGpuMemory(nBytesPerPacket, &packet_p_,
                                         nBytesPerPacket, &packet_d_);
//    Backend::instance().requestGpuMemory(nBytesPerPacket - nTiles * nScratchPerTileBytes,
//                                         &packet_p_, nBytesPerPacket, &packet_d_);

    static_assert(sizeof(char) == 1, "Invalid char size");
    char*   ptr_p = static_cast<char*>(packet_p_);
    char*   ptr_d = static_cast<char*>(packet_d_);

    // Copy in section
    copyInStart_p_ = ptr_p;
    copyInStart_d_ = ptr_d;

    //----- BYTE-ALIGN COPY-IN SECTION
    // Order from largest to smallest in data type size
    // 
    //-- REALS (8-byte)
    // non-tile-specific reals
    dt_p_ = static_cast<void*>(ptr_p);
    dt_d_ = static_cast<void*>(ptr_d);
    ptr_p += sz_dt;
    ptr_d += sz_dt;

    // tile-specific reals
    deltas_start_p_ = static_cast<void*>(ptr_p);
    deltas_start_d_ = static_cast<void*>(ptr_d);
    ptr_p += nTiles_h_ * sz_deltas;
    ptr_d += nTiles_h_ * sz_deltas;

    //-- INTEGERS (4-byte)
    // non-tile-specific integers
    nTiles_p_ = static_cast<void*>(ptr_p);
    nTiles_d_ = static_cast<void*>(ptr_d);
    ptr_p += sz_nTiles;
    ptr_d += sz_nTiles;

    // tile-specific integers
    lo_start_p_ = static_cast<void*>(ptr_p);
    lo_start_d_ = static_cast<void*>(ptr_d);
    ptr_p += nTiles_h_ * sz_lo;
    ptr_d += nTiles_h_ * sz_lo;

    hi_start_p_ = static_cast<void*>(ptr_p);
    hi_start_d_ = static_cast<void*>(ptr_d);
    ptr_p += nTiles_h_ * sz_hi;
    ptr_d += nTiles_h_ * sz_hi;

    // No copy in/out data
    copyInOutStart_p_ = copyInStart_p_ + nCopyInBytes;
    copyInOutStart_d_ = copyInStart_d_ + nCopyInBytes;

    // No copy-out data

    // TODO:  Acquire memory first and do copy to pinned buffer.   Only then
    // acquire stream.  In this way, we can copy data even if there are no
    // streams available.  Determine if a deadlock can occur due to acquiring
    // streams and memory separately.
    stream_ = Backend::instance().requestStream(true);
    if (!stream_.isValid()) {
        throw std::runtime_error("[DataPacket_Hydro_gpu_3::pack] Unable to acquire stream");
    }
//#if NDIM == 3
//    stream2_ = Backend::instance().requestStream(true);
//    stream3_ = Backend::instance().requestStream(true);
//    if (!stream2_.isValid() || !stream3_.isValid()) {
//        throw std::runtime_error("[DataPacket_Hydro_gpu_3::pack] Unable to acquire extra streams");
//    }
//#endif

    // Define high-level structure
    location_ = PacketDataLocation::CC1;

    //----- SCRATCH SECTION
    // Nothing to include nor record

    //----- COPY IN SECTION
    // Non-tile-specific data
    std::memcpy(nTiles_p_, static_cast<void*>(&nTiles_h_), sz_nTiles);
    std::memcpy(dt_p_,     static_cast<void*>(&dt_h_),     sz_dt);

    // Tile-specific metadata
    char*   char_ptr;
    for (std::size_t n=0; n<nTiles_h_; ++n) {
        Tile*   tileDesc_h = tiles_[n].get();
        if (tileDesc_h == nullptr) {
            throw std::runtime_error("[DataPacket_Hydro_gpu_3::pack] Bad tileDesc");
        }

        const RealVect      deltas = tileDesc_h->deltas();
        const IntVect       lo     = tileDesc_h->lo();
        const IntVect       hi     = tileDesc_h->hi();
        const IntVect       loGC   = tileDesc_h->loGC();
        const IntVect       hiGC   = tileDesc_h->hiGC();
        Real*               data_h = tileDesc_h->dataPtr();
        if (data_h == nullptr) {
            throw std::logic_error("[DataPacket_Hydro_gpu_3::pack] "
                                   "Invalid pointer to data in host memory");
        }

//        // Put data in copy in/out section
//        // We are not including GAME, which is the last variable in each block
//        std::memcpy((void*)CC_data_p, (void*)data_h, cc1BlockSizeBytes);
//        pinnedPtrs_[n].CC1_data = static_cast<Real*>((void*)CC_data_p);
//        // Data will always be copied back from CC1
//        pinnedPtrs_[n].CC2_data = nullptr;

        Real    deltas_h[MDIM] = {deltas[0], deltas[1], deltas[2]};
        char_ptr = static_cast<char*>(deltas_start_p_) + n * sz_deltas;
        std::memcpy(static_cast<void*>(char_ptr),
                    static_cast<void*>(deltas_h),
                    sz_deltas);

        // Global index space is 0-based in runtime; 1-based in Fortran code.
        // Translate here so that it is immediately ready for use with Fortran.
        int     lo_h[MDIM] = {lo[0]+1, lo[1]+1, lo[2]+1};
        char_ptr = static_cast<char*>(lo_start_p_) + n * sz_lo;
        std::memcpy(static_cast<void*>(char_ptr),
                    static_cast<void*>(lo_h),
                    sz_lo);

        // Global index space is 0-based in runtime; 1-based in Fortran code.
        // Translate here so that it is immediately ready for use with Fortran.
        int     hi_h[MDIM] = {hi[0]+1, hi[1]+1, hi[2]+1};
        char_ptr = static_cast<char*>(hi_start_p_) + n * sz_hi;
        std::memcpy(static_cast<void*>(char_ptr),
                    static_cast<void*>(hi_h),
                    sz_hi);
    }
}

}

