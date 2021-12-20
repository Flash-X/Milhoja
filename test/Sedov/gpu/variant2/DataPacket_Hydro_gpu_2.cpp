#include "DataPacket_Hydro_gpu_2.h"

#include <cassert>
#include <cstring>
#include <stdexcept>

#include <milhoja.h>
#include <Grid_IntVect.h>
#include <Grid_RealVect.h>
#include <Grid.h>
#include <FArray4D.h>
#include <Backend.h>

#include "Sedov.h"
#include "Driver.h"

#if NFLUXES <= 0
#error "Sedov problem should include fluxes"
#endif

namespace orchestration {

/**
 * Construct a DataPacket containing no Tile objects and with no resources
 * assigned to it.
 */
DataPacket_Hydro_gpu_2::DataPacket_Hydro_gpu_2(void)
    : DataPacket{},
      dt_d_{nullptr},
      N_ELEMENTS_PER_CC_PER_VARIABLE{0},
      N_ELEMENTS_PER_FCX_PER_VARIABLE{0},
      N_ELEMENTS_PER_FCX{0},
      N_ELEMENTS_PER_FCY_PER_VARIABLE{0},
      N_ELEMENTS_PER_FCY{0},
      N_ELEMENTS_PER_FCZ_PER_VARIABLE{0},
      N_ELEMENTS_PER_FCZ{0},
      DRIVER_DT_SIZE_BYTES{0},
      DELTA_SIZE_BYTES{0},
      FCX_BLOCK_SIZE_BYTES{0},
      FCY_BLOCK_SIZE_BYTES{0},
      FCZ_BLOCK_SIZE_BYTES{0},
      POINT_SIZE_BYTES{0},
      ARRAY4_SIZE_BYTES{0}
{
    unsigned int    nxb, nyb, nzb;
    Grid::instance().getBlockSize(&nxb, &nyb, &nzb);

    N_ELEMENTS_PER_CC_PER_VARIABLE =   (nxb + 2 * NGUARD * K1D)
                                     * (nyb + 2 * NGUARD * K2D)
                                     * (nzb + 2 * NGUARD * K3D);

    N_ELEMENTS_PER_FCX_PER_VARIABLE = (nxb + 1) * nyb * nzb;
    N_ELEMENTS_PER_FCX = N_ELEMENTS_PER_FCX_PER_VARIABLE * NFLUXES;

    N_ELEMENTS_PER_FCY_PER_VARIABLE = nxb * (nyb + 1) * nzb;
    N_ELEMENTS_PER_FCY = N_ELEMENTS_PER_FCY_PER_VARIABLE * NFLUXES;

    N_ELEMENTS_PER_FCZ_PER_VARIABLE = nxb * nyb * (nzb + 1);
    N_ELEMENTS_PER_FCZ = N_ELEMENTS_PER_FCZ_PER_VARIABLE * NFLUXES;

    DRIVER_DT_SIZE_BYTES =          sizeof(Real);
    DELTA_SIZE_BYTES     =          sizeof(RealVect);
    FCX_BLOCK_SIZE_BYTES = N_ELEMENTS_PER_FCX
                                  * sizeof(Real);
    FCY_BLOCK_SIZE_BYTES = N_ELEMENTS_PER_FCY
                                  * sizeof(Real);
    FCZ_BLOCK_SIZE_BYTES = N_ELEMENTS_PER_FCZ
                                  * sizeof(Real);
    POINT_SIZE_BYTES     =          sizeof(IntVect);
    ARRAY4_SIZE_BYTES    =          sizeof(FArray4D);
}

/**
 * Destroy DataPacket.  Under normal circumstances, the DataPacket should have
 * been consumed and therefore own no resources.
 */
DataPacket_Hydro_gpu_2::~DataPacket_Hydro_gpu_2(void) {
    for (unsigned int i=0; i<N_STREAMS; ++i) {
        if (streams_[i].isValid()) {
            throw std::logic_error("[DataPacket_Hydro_gpu_2::~DataPacket_Hydro_gpu_2] "
                                   "One or more extra streams not released");
        }
    }
}

/**
 *
 */
std::unique_ptr<DataPacket>   DataPacket_Hydro_gpu_2::clone(void) const {
    return std::unique_ptr<DataPacket>{new DataPacket_Hydro_gpu_2{}};
}

#ifdef ENABLE_OPENACC_OFFLOAD
/**
 * Refer to the documentation of this member function for DataPacket.
 */
void  DataPacket_Hydro_gpu_2::releaseExtraQueue(const unsigned int id) {
    if ((id < 2) || (id > N_STREAMS + 1)) {
        throw std::invalid_argument("[DataPacket_Hydro_gpu_2::releaseExtraQueue] "
                                    "Invalid id");
    } else if (!streams_[id-2].isValid()) {
        throw std::logic_error("[DataPacket_Hydro_gpu_2::releaseExtraQueue] "
                               "Extra queue invalid or already released");
    }

    Backend::instance().releaseStream(streams_[id-2]);
}
#endif

#ifdef ENABLE_OPENACC_OFFLOAD
/**
 * Refer to the documentation of this member function for DataPacket.
 */
int  DataPacket_Hydro_gpu_2::extraAsynchronousQueue(const unsigned int id) {
    if ((id < 2) || (id > N_STREAMS + 1)) {
        throw std::invalid_argument("[DataPacket_Hydro_gpu_2::extraAsynchronousQueue] "
                                    "Invalid id");
    } else if (!streams_[id-2].isValid()) {
        throw std::logic_error("[DataPacket_Hydro_gpu_2::extraAsynchronousQueue] "
                               "Second queue invalid");
    }

    return streams_[id-2].accAsyncQueue;
}
#endif

/**
 *
 */
void  DataPacket_Hydro_gpu_2::pack(void) {
    std::string   errMsg = isNull();
    if (errMsg != "") {
        throw std::logic_error("[DataPacket_Hydro_gpu_2::pack] " + errMsg);
    } else if (tiles_.size() == 0) {
        throw std::logic_error("[DataPacket_Hydro_gpu_2::pack] No tiles added");
    }

    Grid&   grid = Grid::instance();

    //----- SCRATCH SECTION
    // First block of memory should be allocated as acratch space
    // for use by GPU.  No need for this to be transferred at all.
    // 
    // GAMC is read from, but not written to by the task function that uses this
    // packet.  In addition, GAME is neither read from nor written to.
    // Therefore, GAME need not be included in the packet at all.  GAMC should
    // be included in the copyInOut section (i.e. CC1), but not in the copyOut
    // section (i.e. CC2).  As part of this, the variable order in memory was
    // setup so that GAME is the last variable; GAMC, the penultimate.
    std::size_t  nCc1Variables = NUNKVAR - 1;
    std::size_t  nCc2Variables = NUNKVAR - 2;
    std::size_t  cc1BlockSizeBytes =   nCc1Variables
                                     * N_ELEMENTS_PER_CC_PER_VARIABLE
                                     * sizeof(Real);
    std::size_t  cc2BlockSizeBytes =   nCc2Variables
                                     * N_ELEMENTS_PER_CC_PER_VARIABLE
                                     * sizeof(Real);

    std::size_t  nScratchPerTileBytes = FCX_BLOCK_SIZE_BYTES;
    unsigned int nScratchArrays = 1;
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
                                               + (nScratchArrays + 2) * ARRAY4_SIZE_BYTES;
    std::size_t  nCopyInDataPerTileBytes = cc1BlockSizeBytes;

    //----- COPY IN/OUT SECTION
    // No copy-in/out data

    //----- COPY OUT SECTION
    std::size_t  nCopyOutDataPerTileBytes = cc2BlockSizeBytes;

    nCopyToGpuBytes_ =            nCopyInBytes
                       + nTiles * nBlockMetadataPerTileBytes
                       + nTiles * nCopyInDataPerTileBytes;
    nReturnToHostBytes_ = nTiles * nCopyOutDataPerTileBytes;
    std::size_t  nBytesPerPacket =   nTiles * nScratchPerTileBytes
                                   +          nCopyInBytes
                                   + nTiles * nBlockMetadataPerTileBytes
                                   + nTiles * nCopyInDataPerTileBytes
                                   + nTiles * nCopyOutDataPerTileBytes;

    stream_ = Backend::instance().requestStream(true);
    if (!stream_.isValid()) {
        throw std::runtime_error("[DataPacket_Hydro_gpu_2::pack] Unable to acquire stream");
    }
    for (unsigned int i=0; i<N_STREAMS; ++i) {
        streams_[i] = Backend::instance().requestStream(true);
        if (!streams_[i].isValid()) {
            throw std::runtime_error("[DataPacket_Hydro_gpu_2::pack] Unable to acquire extra stream");
        }
    }

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
                               + nTiles * nBlockMetadataPerTileBytes
                               + nTiles * nCopyInDataPerTileBytes;
    copyInOutStart_d_        =            copyInStart_d_
                               +          nCopyInBytes
                               + nTiles * nBlockMetadataPerTileBytes
                               + nTiles * nCopyInDataPerTileBytes;
    char* copyOutStart_p = copyInOutStart_p_;
    char* copyOutStart_d = copyInOutStart_d_;

    // Store for later unpacking the location in pinned memory of the different
    // blocks.
    if (pinnedPtrs_) {
        throw std::logic_error("[DataPacket_Hydro_gpu_2::pack] Pinned pointers already exist");
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

    char* CC1_data_p    =            copyInStart_p_
                          +          nCopyInBytes
                          + nTiles * nBlockMetadataPerTileBytes;
    char* CC1_data_d    =            copyInStart_d_
                          +          nCopyInBytes
                          + nTiles * nBlockMetadataPerTileBytes;
    char* CC2_data_p    = copyOutStart_p;
    char* CC2_data_d    = copyOutStart_d;
    char* FCX_scratch_d = scratchStart_d;
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
            throw std::runtime_error("[DataPacket_Hydro_gpu_2::pack] Bad tileDesc");
        }

        const RealVect      deltas = tileDesc_h->deltas();
        const IntVect       lo     = tileDesc_h->lo();
        const IntVect       hi     = tileDesc_h->hi();
        const IntVect       loGC   = tileDesc_h->loGC();
        const IntVect       hiGC   = tileDesc_h->hiGC();
        Real*               data_h = tileDesc_h->dataPtr();
        if (data_h == nullptr) {
            throw std::logic_error("[DataPacket_Hydro_gpu_2::pack] "
                                   "Invalid pointer to data in host memory");
        }

        // Put data in copy in/out section
        // We are not including GAME, which is the last variable in each block
        std::memcpy((void*)CC1_data_p, (void*)data_h, cc1BlockSizeBytes);
        pinnedPtrs_[n].CC1_data = static_cast<Real*>((void*)CC1_data_p);
        pinnedPtrs_[n].CC2_data = static_cast<Real*>((void*)CC2_data_p);

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
        FArray4D   CC1_d{static_cast<Real*>((void*)CC1_data_d),
                         loGC, hiGC, nCc1Variables};
        std::memcpy((void*)ptr_p, (void*)&CC1_d, ARRAY4_SIZE_BYTES);
        ptr_p += ARRAY4_SIZE_BYTES;
        ptr_d += ARRAY4_SIZE_BYTES;

        tilePtrs_p->CC2_d = static_cast<FArray4D*>((void*)ptr_d);
        FArray4D   CC2_d{static_cast<Real*>((void*)CC2_data_d),
                         loGC, hiGC, nCc2Variables};
        std::memcpy((void*)ptr_p, (void*)&CC2_d, ARRAY4_SIZE_BYTES);
        ptr_p += ARRAY4_SIZE_BYTES;
        ptr_d += ARRAY4_SIZE_BYTES;

        tilePtrs_p->FCX_d = static_cast<FArray4D*>((void*)ptr_d);
        IntVect    fHi = IntVect{LIST_NDIM(hi.I()+1, hi.J(), hi.K())};
        FArray4D   FCX_d{static_cast<Real*>((void*)FCX_scratch_d),
                         lo, fHi, NFLUXES};
        std::memcpy((void*)ptr_p, (void*)&FCX_d, ARRAY4_SIZE_BYTES);
        ptr_p += ARRAY4_SIZE_BYTES;
        ptr_d += ARRAY4_SIZE_BYTES;

        CC1_data_p    += cc1BlockSizeBytes;
        CC1_data_d    += cc1BlockSizeBytes;
        CC2_data_p    += cc2BlockSizeBytes;
        CC2_data_d    += cc2BlockSizeBytes;
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

/**
 * The runtime calls this member function automatically once a DataPacket
 * has arrived in the host memory again.  It is responsible for unpacking
 * the contents and in particular for copying cell-centered data back to the
 * host-side Grid data structures that hold solution data.  The data is copied
 * back in accord with the variable masks set in the DataPacket to avoid
 * inadvertently overwriting variables that were updated in parallel by other
 * actions.
 *
 * All memory and stream resources are released.
 *
 * While the packet is consumed once the function finishes, the list of Tiles
 * that were included in the packet is preserved.  This is necessary so that
 * runtime elements such as MoverUnpacker can enqueue the Tiles with its data
 * subscriber.
 *
 * @todo Should unpacking be made more generic so that the CC blocks need not 
 *       start always with the first data variable.  What if the packet just
 *       needs to include variables 3-5 (out of 10 for example)?
 */
void  DataPacket_Hydro_gpu_2::unpack(void) {
    if (tiles_.size() <= 0) {
        throw std::logic_error("[DataPacket_Hydro_gpu_2::unpack] "
                               "Empty data packet");
    } else if (!stream_.isValid()) {
        throw std::logic_error("[DataPacket_Hydro_gpu_2::unpack] "
                               "Stream not acquired");
    } else if (pinnedPtrs_ == nullptr) {
        throw std::logic_error("[DataPacket_Hydro_gpu_2::unpack] "
                               "No pinned pointers set");
    } else if (   (startVariable_ < UNK_VARS_BEGIN )
               || (startVariable_ > UNK_VARS_END )
               || (endVariable_   < UNK_VARS_BEGIN )
               || (endVariable_   > UNK_VARS_END)) {
        throw std::logic_error("[DataPacket_Hydro_gpu_2::unpack] "
                               "Invalid variable mask");
    }

    // Release stream as soon as possible
    Backend::instance().releaseStream(stream_);
    assert(!stream_.isValid());

    for (std::size_t n=0; n<tiles_.size(); ++n) {
        Tile*   tileDesc_h = tiles_[n].get();

        Real*         data_h = tileDesc_h->dataPtr();
        const Real*   data_p = nullptr;
        switch (location_) {
            case PacketDataLocation::CC1:
                data_p = pinnedPtrs_[n].CC1_data;
                break;
            case PacketDataLocation::CC2:
                data_p = pinnedPtrs_[n].CC2_data;
                break;
            default:
                throw std::logic_error("[DataPacket_Hydro_gpu_2::unpack] Data not in CC1 or CC2");
        }

        if (data_h == nullptr) {
            throw std::logic_error("[DataPacket_Hydro_gpu_2::unpack] "
                                   "Invalid pointer to data in host memory");
        } else if (data_p == nullptr) {
            throw std::runtime_error("[DataPacket_Hydro_gpu_2::unpack] "
                                     "Invalid pointer to data in pinned memory");
        }

        // The code here imposes requirements on the variable indices.
        assert(UNK_VARS_BEGIN == 0);
        assert(UNK_VARS_END == (NUNKVAR - 1));
        std::size_t  offset =   N_ELEMENTS_PER_CC_PER_VARIABLE
                              * static_cast<std::size_t>(startVariable_);
        Real*        start_h = data_h + offset;
        const Real*  start_p = data_p + offset;
        std::size_t  nBytes =  (endVariable_ - startVariable_ + 1)
                              * N_ELEMENTS_PER_CC_PER_VARIABLE
                              * sizeof(Real);
        std::memcpy((void*)start_h, (void*)start_p, nBytes);
    }

    // The packet is consumed upon unpacking.
    nullify();
}

}

