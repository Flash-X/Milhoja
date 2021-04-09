#include "DataPacket_Hydro_gpu_1.h"

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
 *
 */
DataPacket_Hydro_gpu_1::DataPacket_Hydro_gpu_1(void)
    : DataPacket{},
#if NDIM == 3
      stream2_{},
      stream3_{},
#endif
      dt_d_{nullptr}
{
}

/**
 *
 */
DataPacket_Hydro_gpu_1::~DataPacket_Hydro_gpu_1(void) {
#if NDIM == 3
    if (stream2_.isValid() || stream3_.isValid()) {
        throw std::logic_error("[DataPacket_Hydro_gpu_1::~DataPacket_Hydro_gpu_1] "
                               "One or more extra streams not released");
    }
#endif
}

/**
 *
 */
std::unique_ptr<DataPacket>   DataPacket_Hydro_gpu_1::clone(void) const {
    return std::unique_ptr<DataPacket>{new DataPacket_Hydro_gpu_1{}};
}

#if NDIM == 3
/**
 * Do not call this member function before calling pack() or more than once on
 * the same queue.
 */
void  DataPacket_Hydro_gpu_1::releaseExtraQueue(const unsigned int id) {
    if        (id == 2) {
        if (!stream2_.isValid()) {
            throw std::logic_error("[DataPacket_Hydro_gpu_1::releaseExtraQueue] "
                                   "Second queue invalid or already released");
        } else {
            Backend::instance().releaseStream(stream2_);
        }
    } else if (id == 3) {
        if (!stream3_.isValid()) {
            throw std::logic_error("[DataPacket_Hydro_gpu_1::releaseExtraQueue] "
                                   "Third queue invalid or already released");
        } else {
            Backend::instance().releaseStream(stream3_);
        }
    } else {
        throw std::invalid_argument("[DataPacket_Hydro_gpu_1::releaseExtraQueue] "
                                    "Invalid id");
    }
}
#endif

#if NDIM == 3
/**
 * Pack must be called before calling this member function.  It cannot be called
 * after calling releaseExtraStream on the same ID.
 *
 * Calling code is allowed to acquire the same queue as many times as desired.
 * It is the calling code's responsibility to use the queues correctly.
 */
int  DataPacket_Hydro_gpu_1::extraAsynchronousQueue(const unsigned int id) {
    if        (id == 2) {
        if (!stream2_.isValid()) {
            throw std::logic_error("[DataPacket_Hydro_gpu_1::extraAsynchronousQueue] "
                                   "Second queue invalid");
        } else {
            return stream2_.accAsyncQueue;
        }
    } else if (id == 3) {
        if (!stream3_.isValid()) {
            throw std::logic_error("[DataPacket_Hydro_gpu_1::extraAsynchronousQueue] "
                                   "Third queue invalid");
        } else {
            return stream3_.accAsyncQueue;
        }
    } else {
        throw std::invalid_argument("[DataPacket_Hydro_gpu_1::extraAsynchronousQueue] Invalid id");
    }
}
#endif

/**
 *
 */
void  DataPacket_Hydro_gpu_1::pack(void) {
    std::string   errMsg = isNull();
    if (errMsg != "") {
        throw std::logic_error("[DataPacket_Hydro_gpu_1::pack] " + errMsg);
    } else if (tiles_.size() == 0) {
        throw std::logic_error("[DataPacket_Hydro_gpu_1::pack] No tiles added");
    }

    Grid&   grid = Grid::instance();

    //----- SCRATCH SECTION
    // First block of memory should be allocated as acratch space
    // for use by GPU.  No need for this to be transferred at all.
    unsigned int nScratchArrays = 2;
    std::size_t  nScratchPerTileBytes  =   CC_BLOCK_SIZE_BYTES
                                        + FCX_BLOCK_SIZE_BYTES;
#if NDIM >= 2
    nScratchPerTileBytes += FCY_BLOCK_SIZE_BYTES;
    ++nScratchArrays;
#endif
#if NDIM == 3
    nScratchPerTileBytes += FCZ_BLOCK_SIZE_BYTES;
    ++nScratchArrays;
#endif

    //----- COPY IN SECTION
    // Data needed in GPU that is not tile-specific
    std::size_t  nTiles = tiles_.size();
    std::size_t  nCopyInBytes =            sizeof(std::size_t)
                                +          DRIVER_DT_SIZE_BYTES
                                + nTiles * sizeof(PacketContents);
    // Tile metadata including array objects that wrap the scratch
    // blocks for use by the GPU.
    std::size_t  nBlockMetadataPerTileBytes  =                     1  * DELTA_SIZE_BYTES
                                               +                   2  * POINT_SIZE_BYTES
                                               + (nScratchArrays + 1) * ARRAY4_SIZE_BYTES;
    // No copy-in block data

    //----- COPY IN/OUT SECTION
    // All computation on CC data effectively done in place
    std::size_t  nCopyInOutDataPerTileBytes = CC_BLOCK_SIZE_BYTES;

    //----- COPY OUT SECTION
    // No copy-out data
    nCopyToGpuBytes_ =            nCopyInBytes
                       + nTiles * nBlockMetadataPerTileBytes
                       + nTiles * nCopyInOutDataPerTileBytes;
    nReturnToHostBytes_ = nTiles * nCopyInOutDataPerTileBytes;
    std::size_t  nBytesPerPacket =   nTiles * nScratchPerTileBytes
                                   +          nCopyInBytes
                                   + nTiles * nBlockMetadataPerTileBytes
                                   + nTiles * nCopyInOutDataPerTileBytes;

    stream_ = Backend::instance().requestStream(true);
    if (!stream_.isValid()) {
        throw std::runtime_error("[DataPacket_Hydro_gpu_1::pack] Unable to acquire stream");
    }
#if NDIM == 3
    stream2_ = Backend::instance().requestStream(true);
    stream3_ = Backend::instance().requestStream(true);
    if (!stream2_.isValid() || !stream3_.isValid()) {
        throw std::runtime_error("[DataPacket_Hydro_gpu_1::pack] Unable to acquire extra streams");
    }
#endif

    // ACQUIRE PINNED AND GPU MEMORY & SPECIFY STRUCTURE
    // Scratch only needed on GPU side
    Backend::instance().requestGpuMemory(nBytesPerPacket - nTiles * nScratchPerTileBytes,
                                         &packet_p_, nBytesPerPacket, &packet_d_);

    // Define high-level structure
    location_ = PacketDataLocation::CC1;

    char*  scratchStart_d    = static_cast<char*>(packet_d_);
    copyInStart_p_           = static_cast<char*>(packet_p_);
    copyInStart_d_           =            scratchStart_d
                               + nTiles * nScratchPerTileBytes;
    copyInOutStart_p_        =            copyInStart_p_
                               +          nCopyInBytes
                               + nTiles * nBlockMetadataPerTileBytes;
    copyInOutStart_d_        =            copyInStart_d_
                               +          nCopyInBytes
                               + nTiles * nBlockMetadataPerTileBytes;

    // Store for later unpacking the location in pinned memory of the different
    // blocks.
    if (pinnedPtrs_) {
        throw std::logic_error("[DataPacket_Hydro_gpu_1::pack] Pinned pointers already exist");
    }
    pinnedPtrs_ = new BlockPointersPinned[nTiles];

    //----- SCRATCH SECTION
    // Nothing to include nor record

    //----- COPY IN SECTION
    // Pointer to the next free byte in the current data packets
    // Should be true by C++ standard
    static_assert(sizeof(char) == 1, "Invalid char size");
    char*   ptr_p = copyInStart_p_;
    char*   ptr_d = copyInStart_d_;

    // Non-tile-specific data
    std::memcpy((void*)ptr_p, (void*)&nTiles, sizeof(std::size_t));
    ptr_p += sizeof(std::size_t);
    ptr_d += sizeof(std::size_t);

    dt_d_ = static_cast<Real*>((void*)ptr_d); 
    std::memcpy((void*)ptr_p, (void*)&Driver::dt, DRIVER_DT_SIZE_BYTES);
    ptr_p += sizeof(DRIVER_DT_SIZE_BYTES);
    ptr_d += sizeof(DRIVER_DT_SIZE_BYTES);

    // TODO: The PacketContents are a means to avoid putting related, ugly unpacking
    //       code in the patch code.  This certainly aids in developing and
    //       maintaining that code.  However, once the patch code is written
    //       by a code generator, we need not manage the contents like this nor
    //       put it in the data packet.
    contents_p_ = static_cast<PacketContents*>((void*)ptr_p);
    contents_d_ = static_cast<PacketContents*>((void*)ptr_d);
    ptr_p += nTiles * sizeof(PacketContents);
    ptr_d += nTiles * sizeof(PacketContents);

    char* CC_data_p     = copyInOutStart_p_;
    char* CC_data_d     = copyInOutStart_d_;
    char* CC_scratch_d  = scratchStart_d;
    char* FCX_scratch_d = CC_scratch_d +  CC_BLOCK_SIZE_BYTES;
#if NDIM >= 2
    char* FCY_scratch_d = FCX_scratch_d + FCX_BLOCK_SIZE_BYTES;
#endif
#if NDIM == 3
    char* FCZ_scratch_d = FCY_scratch_d + FCY_BLOCK_SIZE_BYTES;
#endif

    // Tile-specific metadata
    PacketContents*   tilePtrs_p = contents_p_;
    for (std::size_t n=0; n<nTiles; ++n, ++tilePtrs_p) {
        Tile*   tileDesc_h = tiles_[n].get();
        if (tileDesc_h == nullptr) {
            throw std::runtime_error("[DataPacket_Hydro_gpu_1::pack] Bad tileDesc");
        }

        const RealVect      deltas = tileDesc_h->deltas();
        const IntVect       lo     = tileDesc_h->lo();
        const IntVect       hi     = tileDesc_h->hi();
        const IntVect       loGC   = tileDesc_h->loGC();
        const IntVect       hiGC   = tileDesc_h->hiGC();
        Real*               data_h = tileDesc_h->dataPtr();
        if (data_h == nullptr) {
            throw std::logic_error("[DataPacket_Hydro_gpu_1::pack] "
                                   "Invalid pointer to data in host memory");
        }

        // Put data in copy in/out section
        // TODO: Can we transfer over all CC1 data in a single memcpy call?
        std::memcpy((void*)CC_data_p, (void*)data_h, CC_BLOCK_SIZE_BYTES);
        pinnedPtrs_[n].CC1_data = static_cast<Real*>((void*)CC_data_p);
        // Data will always be copied back from CC1
        pinnedPtrs_[n].CC2_data = nullptr;

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

        // Create an FArray4D object in host memory but that already points
        // to where its data will be in device memory (i.e. the device object
        // will already be attached to its data in device memory).
        // The object in host memory should never be used then.
        // IMPORTANT: When this local object is destroyed, we don't want it to
        // affect the use of the copies (e.g. release memory).
        tilePtrs_p->CC1_d = static_cast<FArray4D*>((void*)ptr_d);
        FArray4D   CC_d{static_cast<Real*>((void*)CC_data_d),
                        loGC, hiGC, NUNKVAR};
        std::memcpy((void*)ptr_p, (void*)&CC_d, ARRAY4_SIZE_BYTES);
        ptr_p += ARRAY4_SIZE_BYTES;
        ptr_d += ARRAY4_SIZE_BYTES;

        tilePtrs_p->CC2_d = static_cast<FArray4D*>((void*)ptr_d);
        FArray4D   CC_sc_d{static_cast<Real*>((void*)CC_scratch_d),
                           loGC, hiGC, NUNKVAR};
        std::memcpy((void*)ptr_p, (void*)&CC_sc_d, ARRAY4_SIZE_BYTES);
        ptr_p += ARRAY4_SIZE_BYTES;
        ptr_d += ARRAY4_SIZE_BYTES;

        tilePtrs_p->FCX_d = static_cast<FArray4D*>((void*)ptr_d);
        IntVect    fHi = IntVect{LIST_NDIM(hi.I()+1, hi.J(), hi.K())};
        FArray4D   FCX_d{static_cast<Real*>((void*)FCX_scratch_d),
                         lo, fHi, NFLUXES};
        std::memcpy((void*)ptr_p, (void*)&FCX_d, ARRAY4_SIZE_BYTES);
        ptr_p += ARRAY4_SIZE_BYTES;
        ptr_d += ARRAY4_SIZE_BYTES;

        CC_data_p     += CC_BLOCK_SIZE_BYTES;
        CC_data_d     += CC_BLOCK_SIZE_BYTES;
        CC_scratch_d  += nScratchPerTileBytes;
        FCX_scratch_d += nScratchPerTileBytes;

#if NDIM >= 2
        tilePtrs_p->FCY_d = static_cast<FArray4D*>((void*)ptr_d);
        fHi = IntVect{LIST_NDIM(hi.I(), hi.J()+1, hi.K())};
        FArray4D   FCY_d{static_cast<Real*>((void*)FCY_scratch_d),
                         lo, fHi, NFLUXES};
        std::memcpy((void*)ptr_p, (void*)&FCY_d, ARRAY4_SIZE_BYTES);
        ptr_p += ARRAY4_SIZE_BYTES;
        ptr_d += ARRAY4_SIZE_BYTES;

        FCY_scratch_d += nScratchPerTileBytes;
#else
        tilePtrs_p->FCY_d = nullptr;
#endif

#if NDIM == 3
        tilePtrs_p->FCZ_d = static_cast<FArray4D*>((void*)ptr_d);
        fHi = IntVect{LIST_NDIM(hi.I(), hi.J(), hi.K()+1)};
        FArray4D   FCZ_d{static_cast<Real*>((void*)FCZ_scratch_d),
                         lo, fHi, NFLUXES};

        std::memcpy((void*)ptr_p, (void*)&FCZ_d, ARRAY4_SIZE_BYTES);
        ptr_p += ARRAY4_SIZE_BYTES;
        ptr_d += ARRAY4_SIZE_BYTES;

        FCZ_scratch_d += nScratchPerTileBytes;
#else
        tilePtrs_p->FCZ_d = nullptr;
#endif
    }
}

}

