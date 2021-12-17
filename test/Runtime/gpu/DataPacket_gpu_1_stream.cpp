#include "DataPacket_gpu_1_stream.h"

#include <cassert>
#include <cstring>

#include <milhoja.h>
#include <Grid_IntVect.h>
#include <Grid_RealVect.h>
#include <Grid.h>
#include <FArray4D.h>
#include <Backend.h>

#include "Base.h"

namespace orchestration {

DataPacket_gpu_1_stream::DataPacket_gpu_1_stream(void)
    : N_ELEMENTS_PER_CC_PER_VARIABLE{0},
      N_ELEMENTS_PER_CC{0},
      DELTA_SIZE_BYTES{0},
      CC_BLOCK_SIZE_BYTES{0},
      POINT_SIZE_BYTES{0},
      ARRAY4_SIZE_BYTES{0}
{
    unsigned int  nxb, nyb, nzb;
    Grid::instance().getBlockSize(&nxb, &nyb, &nzb);

    N_ELEMENTS_PER_CC_PER_VARIABLE =   (nxb + 2 * NGUARD * K1D)
                                     * (nyb + 2 * NGUARD * K2D)
                                     * (nzb + 2 * NGUARD * K3D);
    N_ELEMENTS_PER_CC  = N_ELEMENTS_PER_CC_PER_VARIABLE * NUNKVAR;

    DELTA_SIZE_BYTES     =          sizeof(RealVect);
    CC_BLOCK_SIZE_BYTES  = N_ELEMENTS_PER_CC
                                  * sizeof(Real);
    POINT_SIZE_BYTES     =          sizeof(IntVect);
    ARRAY4_SIZE_BYTES    =          sizeof(FArray4D);
}

/**
 *
 */
std::unique_ptr<DataPacket>   DataPacket_gpu_1_stream::clone(void) const {
    return std::unique_ptr<DataPacket>{new DataPacket_gpu_1_stream{}};
}

/**
 *
 */
void  DataPacket_gpu_1_stream::pack(void) {
    std::string   errMsg = isNull();
    if (errMsg != "") {
        throw std::logic_error("[DataPacket_gpu_1_stream::pack] " + errMsg);
    } else if (tiles_.size() == 0) {
        throw std::logic_error("[DataPacket_gpu_1_stream::pack] No tiles added");
    }

    Grid&   grid = Grid::instance();

    //----- SCRATCH SECTION
    // No scratch

    //----- COPY IN SECTION
    // Data needed in GPU that is not tile-specific
    location_ = PacketDataLocation::CC1;

    std::size_t  nTiles = tiles_.size();
    std::size_t  nCopyInBytes =            sizeof(std::size_t)
                                + nTiles * sizeof(PacketContents);
    std::size_t  nBlockMetadataPerTileBytes  =   1 * DELTA_SIZE_BYTES
                                               + 2 * POINT_SIZE_BYTES
                                               + 2 * ARRAY4_SIZE_BYTES;
    std::size_t  nCopyInDataPerTileBytes = CC_BLOCK_SIZE_BYTES;

    //----- COPY IN/OUT SECTION
    // No copy-in/out data

    //----- COPY OUT SECTION
    std::size_t  nCopyOutDataPerTileBytes = CC_BLOCK_SIZE_BYTES;

    nCopyToGpuBytes_ =            nCopyInBytes
                       + nTiles * nBlockMetadataPerTileBytes
                       + nTiles * nCopyInDataPerTileBytes;
    nReturnToHostBytes_ = nTiles * nCopyOutDataPerTileBytes;
    std::size_t  nBytesPerPacket =            nCopyInBytes
                                   + nTiles * nBlockMetadataPerTileBytes
                                   + nTiles * nCopyInDataPerTileBytes
                                   + nTiles * nCopyOutDataPerTileBytes;

    stream_ = Backend::instance().requestStream(true);
    if (!stream_.isValid()) {
        throw std::runtime_error("[DataPacket_gpu_1_stream::pack] Unable to acquire stream");
    }

    // Allocate memory in pinned and device memory on demand for now
    Backend::instance().requestGpuMemory(nBytesPerPacket, &packet_p_,
                                         nBytesPerPacket, &packet_d_);

    // Define high-level structure
    copyInStart_p_       = static_cast<char*>(packet_p_);
    copyInStart_d_       = static_cast<char*>(packet_d_);
    copyInOutStart_p_    = copyInStart_p_ + nCopyToGpuBytes_;
    copyInOutStart_d_    = copyInStart_d_ + nCopyToGpuBytes_;
    char* copyOutStart_p = copyInOutStart_p_;
    char* copyOutStart_d = copyInOutStart_d_;

    // Store for later unpacking the location in pinned memory of the different
    // blocks.
    if (pinnedPtrs_) {
        throw std::logic_error("[DataPacket_gpu_1_stream::pack] Pinned pointers already exist");
    }
    pinnedPtrs_ = new BlockPointersPinned[nTiles];

    //----- SCRATCH SECTION
    // No scratch

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

    contents_p_ = static_cast<PacketContents*>((void*)ptr_p);
    contents_d_ = static_cast<PacketContents*>((void*)ptr_d);
    ptr_p += nTiles * sizeof(PacketContents);
    ptr_d += nTiles * sizeof(PacketContents);

    char* CC1_data_p = copyInStart_p_ + nCopyInBytes + nTiles*nBlockMetadataPerTileBytes;
    char* CC1_data_d = copyInStart_d_ + nCopyInBytes + nTiles*nBlockMetadataPerTileBytes;
    char* CC2_data_p = copyOutStart_p;
    char* CC2_data_d = copyOutStart_d;

    PacketContents*   tilePtrs_p = contents_p_;
    for (std::size_t n=0; n<nTiles; ++n, ++tilePtrs_p) {
        Tile*   tileDesc_h = tiles_[n].get();
        if (tileDesc_h == nullptr) {
            throw std::runtime_error("[DataPacket_gpu_1_stream::pack] Bad tileDesc");
        }
 
        const RealVect      deltas = tileDesc_h->deltas();
        const IntVect       lo     = tileDesc_h->lo();
        const IntVect       hi     = tileDesc_h->hi();
        const IntVect       loGC   = tileDesc_h->loGC();
        const IntVect       hiGC   = tileDesc_h->hiGC();
        Real*               data_h = tileDesc_h->dataPtr();
        if (data_h == nullptr) {
            throw std::logic_error("[DataPacket_gpu_1_stream::pack] "
                                   "Invalid pointer to data in host memory");
        }

        pinnedPtrs_[n].CC1_data = static_cast<Real*>((void*)CC1_data_p);
        pinnedPtrs_[n].CC2_data = static_cast<Real*>((void*)CC2_data_p);
        std::memcpy((void*)CC1_data_p, (void*)data_h, CC_BLOCK_SIZE_BYTES);

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
                         loGC, hiGC, NUNKVAR};
        std::memcpy((void*)ptr_p, (void*)&CC1_d, ARRAY4_SIZE_BYTES);
        ptr_p += ARRAY4_SIZE_BYTES;
        ptr_d += ARRAY4_SIZE_BYTES;

        tilePtrs_p->CC2_d = static_cast<FArray4D*>((void*)ptr_d);
        FArray4D   CC2_d{static_cast<Real*>((void*)CC2_data_d),
                         loGC, hiGC, NUNKVAR};
        std::memcpy((void*)ptr_p, (void*)&CC2_d, ARRAY4_SIZE_BYTES);
        ptr_p += ARRAY4_SIZE_BYTES;
        ptr_d += ARRAY4_SIZE_BYTES;

        CC1_data_p += CC_BLOCK_SIZE_BYTES;
        CC1_data_d += CC_BLOCK_SIZE_BYTES;
        CC2_data_p += CC_BLOCK_SIZE_BYTES;
        CC2_data_d += CC_BLOCK_SIZE_BYTES;
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
void  DataPacket_gpu_1_stream::unpack(void) {
    if (tiles_.size() <= 0) {
        throw std::logic_error("[DataPacket_gpu_1_stream::unpack] "
                               "Empty data packet");
    } else if (!stream_.isValid()) {
        throw std::logic_error("[DataPacket_gpu_1_stream::unpack] "
                               "Stream not acquired");
    } else if (pinnedPtrs_ == nullptr) {
        throw std::logic_error("[DataPacket_gpu_1_stream::unpack] "
                               "No pinned pointers set");
    } else if (   (startVariable_ < UNK_VARS_BEGIN )
               || (startVariable_ > UNK_VARS_END )
               || (endVariable_   < UNK_VARS_BEGIN )
               || (endVariable_   > UNK_VARS_END)) {
        throw std::logic_error("[DataPacket_gpu_1_stream::unpack] "
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
                throw std::logic_error("[DataPacket_gpu_1_stream::unpack] Data not in CC1 or CC2");
        }

        if (data_h == nullptr) {
            throw std::logic_error("[DataPacket_gpu_1_stream::unpack] "
                                   "Invalid pointer to data in host memory");
        } else if (data_p == nullptr) {
            throw std::runtime_error("[DataPacket_gpu_1_stream::unpack] "
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

