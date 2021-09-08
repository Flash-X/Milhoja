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
      hi_start_d_{nullptr},
      loGC_start_p_{nullptr},
      loGC_start_d_{nullptr},
      hiGC_start_p_{nullptr},
      hiGC_start_d_{nullptr},
      U_start_p_{nullptr},
      U_start_d_{nullptr},
      auxC_start_d_{nullptr},
      faceX_start_d_{nullptr},
      faceY_start_d_{nullptr},
      faceZ_start_d_{nullptr}
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
void  DataPacket_Hydro_gpu_3::tileSize_host(int* nxb,
                                            int* nyb,
                                            int* nzb,
                                            int* nvar) const {
    *nxb  = NXB + 2 * NGUARD * K1D;
    *nyb  = NYB + 2 * NGUARD * K2D;
    *nzb  = NZB + 2 * NGUARD * K3D;
    // We are not including GAME in U, which is the last variable in each block
    *nvar = NUNKVAR - 1;
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

/**
 *
 */
int*   DataPacket_Hydro_gpu_3::loGC_devptr(void) const {
    return static_cast<int*>(loGC_start_d_);
}

/**
 *
 */
int*   DataPacket_Hydro_gpu_3::hiGC_devptr(void) const {
    return static_cast<int*>(hiGC_start_d_);
}

/**
 *
 */
Real*  DataPacket_Hydro_gpu_3::U_devptr(void) const {
    return static_cast<Real*>(U_start_d_);
}

/**
 *
 */
Real*  DataPacket_Hydro_gpu_3::scratchAuxC_devptr(void) const {
    return static_cast<Real*>(auxC_start_d_);
}

/**
 *
 */
Real*  DataPacket_Hydro_gpu_3::scratchFaceX_devptr(void) const {
    return static_cast<Real*>(faceX_start_d_);
}

/**
 *
 */
Real*  DataPacket_Hydro_gpu_3::scratchFaceY_devptr(void) const {
    return static_cast<Real*>(faceY_start_d_);
}

/**
 *
 */
Real*  DataPacket_Hydro_gpu_3::scratchFaceZ_devptr(void) const {
    return static_cast<Real*>(faceZ_start_d_);
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

    //----- OBTAIN NON-TILE-SPECIFIC HOST-SIDE DATA
    // TODO: Check for correctness of cast here of elsewhere?
    // This cast is necessary since Fortran code consumes the packet.
    nTiles_h_ = static_cast<int>(tiles_.size());
    dt_h_     = Driver::dt;

    int   nxb_h  = -1;
    int   nyb_h  = -1;
    int   nzb_h  = -1;
    int   nvar_h = -1;
    tileSize_host(&nxb_h, &nyb_h, &nzb_h, &nvar_h);

    //----- COMPUTE SIZES/OFFSETS & ACQUIRE MEMORY
    std::size_t    sz_nTiles = sizeof(int);
    std::size_t    sz_dt     = sizeof(Real);

    std::size_t    sz_deltas = MDIM * sizeof(Real);
    std::size_t    sz_lo     = MDIM * sizeof(int);
    std::size_t    sz_hi     = MDIM * sizeof(int);
    std::size_t    sz_loGC   = MDIM * sizeof(int);
    std::size_t    sz_hiGC   = MDIM * sizeof(int);
    std::size_t    sz_U      =   nxb_h * nyb_h * nzb_h * nvar_h * sizeof(Real);
    std::size_t    sz_auxC   =   nxb_h + nyb_h + nzb_h *          sizeof(Real);
    // TODO: Can't this be sized smaller for simpleUnsplit?
    std::size_t    sz_FX     =   (nxb_h + 1) *  nyb_h      *  nzb_h      * sizeof(Real);
    std::size_t    sz_FY     =    nxb_h      * (nyb_h + 1) *  nzb_h      * sizeof(Real);
    std::size_t    sz_FZ     =    nxb_h      *  nyb_h      * (nzb_h + 1) * sizeof(Real);

    std::size_t    nScratchBytes = nTiles_h_ * (sz_auxC + sz_FX);
#if NDIM >= 2
    nScratchBytes += nTiles_h_ * sz_FY;
#endif
#if NDIM == 3
    nScratchBytes += nTiles_h_ * sz_FZ;
#endif
    std::size_t    nScratchBytes_padded 
                        = static_cast<std::size_t>(ceil(nScratchBytes / 8.0)) * 8;

    std::size_t    nCopyInBytes =   sz_nTiles
                                  + sz_dt
                                  + nTiles_h_
                                    * (  sz_deltas
                                       + sz_lo   + sz_hi
                                       + sz_loGC + sz_hiGC);
    std::size_t    nCopyInBytes_padded 
                        = static_cast<std::size_t>(ceil(nCopyInBytes / 8.0)) * 8;

    std::size_t    nCopyInOutBytes = nTiles_h_ * sz_U;
    std::size_t    nCopyInOutBytes_padded 
                        = static_cast<std::size_t>(ceil(nCopyInOutBytes / 8.0)) * 8;

    std::size_t    nCopyOutBytes   = 0;
    std::size_t    nCopyOutBytes_padded 
                        = static_cast<std::size_t>(ceil(nCopyOutBytes / 8.0)) * 8;

    nCopyToGpuBytes_    = nCopyInBytes_padded    + nCopyInOutBytes;
    nReturnToHostBytes_ = nCopyInOutBytes_padded + nCopyOutBytes;
    std::size_t  nBytesPerPacket =   nScratchBytes_padded
                                   + nCopyInBytes_padded
                                   + nCopyInOutBytes_padded
                                   + nCopyOutBytes_padded;

    // ACQUIRE PINNED AND GPU MEMORY & SPECIFY STRUCTURE
    // Scratch only needed on GPU side
    // At present, this call makes certain that each data packet is
    // appropriately byte aligned.
    Backend::instance().requestGpuMemory(nBytesPerPacket - nScratchBytes_padded,
                                         &packet_p_, nBytesPerPacket, &packet_d_);

    //----- BYTE-ALIGN SCRATCH SECTION
    // Order from largest to smallest in data type size
    //
    char*   ptr_d = static_cast<char*>(packet_d_);

    auxC_start_d_  = static_cast<void*>(ptr_d);
    ptr_d += nTiles_h_ * sz_auxC;

    faceX_start_d_ = static_cast<void*>(ptr_d);
    ptr_d += nTiles_h_ * sz_FX;
#if NDIM >= 2
    faceY_start_d_ = static_cast<void*>(ptr_d);
    ptr_d += nTiles_h_ * sz_FY;
#endif
#if NDIM == 3
    faceZ_start_d_ = static_cast<void*>(ptr_d);
    ptr_d += nTiles_h_ * sz_FZ;
#endif
 
    //----- BYTE-ALIGN COPY-IN SECTION
    // Order from largest to smallest in data type size
    // 
    static_assert(sizeof(char) == 1, "Invalid char size");
    copyInStart_p_ = static_cast<char*>(packet_p_);
    copyInStart_d_ =   static_cast<char*>(packet_d_)
                     + nScratchBytes_padded;

    char*   ptr_p = copyInStart_p_;
    ptr_d         = copyInStart_d_;

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

    loGC_start_p_ = static_cast<void*>(ptr_p);
    loGC_start_d_ = static_cast<void*>(ptr_d);
    ptr_p += nTiles_h_ * sz_loGC;
    ptr_d += nTiles_h_ * sz_loGC;

    hiGC_start_p_ = static_cast<void*>(ptr_p);
    hiGC_start_d_ = static_cast<void*>(ptr_d);
    ptr_p += nTiles_h_ * sz_hiGC;
    ptr_d += nTiles_h_ * sz_hiGC;

    //----- BYTE-ALIGN COPY-IN/-OUT SECTION
    // Order from largest to smallest in data type size
    // 
    // Pad copy-in section if necessary to get correct byte alignment
    copyInOutStart_p_ = copyInStart_p_ + nCopyInBytes_padded;
    copyInOutStart_d_ = copyInStart_d_ + nCopyInBytes_padded;

    ptr_p = static_cast<char*>(copyInOutStart_p_);
    ptr_d = static_cast<char*>(copyInOutStart_d_);

    U_start_p_ = static_cast<void*>(ptr_p);
    U_start_d_ = static_cast<void*>(ptr_d);
    ptr_p += nTiles_h_ * sz_U;
    ptr_d += nTiles_h_ * sz_U;

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


    // Store for later unpacking the location in pinned memory of the different
    // blocks.
    if (pinnedPtrs_) {
        throw std::logic_error("[DataPacket_Hydro_gpu_3::pack] Pinned pointers already exist");
    }
    pinnedPtrs_ = new BlockPointersPinned[nTiles_h_];

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

        // Global index space is 0-based in runtime; 1-based in Fortran code.
        // Translate here so that it is immediately ready for use with Fortran.
        int     loGC_h[MDIM] = {loGC[0]+1, loGC[1]+1, loGC[2]+1};
        char_ptr = static_cast<char*>(loGC_start_p_) + n * sz_loGC;
        std::memcpy(static_cast<void*>(char_ptr),
                    static_cast<void*>(loGC_h),
                    sz_loGC);

        // Global index space is 0-based in runtime; 1-based in Fortran code.
        // Translate here so that it is immediately ready for use with Fortran.
        int     hiGC_h[MDIM] = {hiGC[0]+1, hiGC[1]+1, hiGC[2]+1};
        char_ptr = static_cast<char*>(hiGC_start_p_) + n * sz_hiGC;
        std::memcpy(static_cast<void*>(char_ptr),
                    static_cast<void*>(hiGC_h),
                    sz_hiGC);

        char_ptr = static_cast<char*>(U_start_p_) + n * sz_U;
        std::memcpy(static_cast<void*>(char_ptr),
                    static_cast<void*>(data_h),
                    sz_U);
        pinnedPtrs_[n].data_h     = data_h;
        pinnedPtrs_[n].CC1_data_p = static_cast<Real*>((void*)char_ptr);
        // Data will always be copied back from CC1
        pinnedPtrs_[n].CC2_data_p = nullptr;
    }
}

}

