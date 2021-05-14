#include "DataPacket_gpu_2_stream.h"

#include <cassert>
#include <cstring>

#include "Grid_IntVect.h"
#include "Grid_RealVect.h"
#include "Grid.h"
#include "FArray4D.h"
#include "Backend.h"

#include "Flash.h"

namespace orchestration {

/**
 *
 */
DataPacket_gpu_2_stream::~DataPacket_gpu_2_stream(void) {
    if (stream2_.isValid()) {
        throw std::logic_error("[DataPacket_gpu_2_stream::~DataPacket_gpu_2_stream] "
                               "Extra stream not released");
    }
}

/**
 *
 */
std::unique_ptr<DataPacket>   DataPacket_gpu_2_stream::clone(void) const {
    return std::unique_ptr<DataPacket>{new DataPacket_gpu_2_stream{}};
}

#ifdef ENABLE_OPENACC_OFFLOAD
/**
 * Do not call this member function before calling pack() or more than once.
 */
void  DataPacket_gpu_2_stream::releaseExtraQueue(const unsigned int id) {
    if (id != 2) {
        throw std::invalid_argument("[DataPacket_gpu_2_stream::releaseExtraQueue] Invalid id");
    } else if (!stream2_.isValid()) {
        throw std::logic_error("[DataPacket_gpu_2_stream::releaseExtraQueue] No stream");
    }

    Backend::instance().releaseStream(stream2_);
}
#endif

#ifdef ENABLE_OPENACC_OFFLOAD
/**
 * Pack must be called before calling this member function.  It cannot be called
 * after calling releaseExtraStream on the same ID.
 */
int  DataPacket_gpu_2_stream::extraAsynchronousQueue(const unsigned int id) {
    if (id != 2) {
        throw std::invalid_argument("[DataPacket_gpu_2_stream::extraAsynchronousQueue] Invalid id");
    } else if (!stream2_.isValid()) {
        throw std::logic_error("[DataPacket_gpu_2_stream::extraAsynchronousQueue] No stream");
    }

    return stream2_.accAsyncQueue;
}
#endif

/**
 *
 */
void  DataPacket_gpu_2_stream::pack(void) {
    std::string   errMsg = isNull();
    if (errMsg != "") {
        throw std::logic_error("[DataPacket_gpu_2_stream::pack] " + errMsg);
    } else if (tiles_.size() == 0) {
        throw std::logic_error("[DataPacket_gpu_2_stream::pack] No tiles added");
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

    // DEV: The DataPacket will free stream_, but not stream2_.  In addition,
    // stream2_ is not included in the nullify/isNull routines.  As part of
    // this, it is assumed that the patch code will free stream2_ immediately
    // after it is finished with it as stream2_ is acquired here on behalf of
    // the patch code for the purpose of computation only.
    stream_  = Backend::instance().requestStream(true);
    stream2_ = Backend::instance().requestStream(true);
    if ((!stream_.isValid()) || (!stream2_.isValid())) {
        throw std::runtime_error("[DataPacket_gpu_2_stream::pack] Unable to acquire streams");
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
        throw std::logic_error("[DataPacket_gpu_2_stream::pack] Pinned pointers already exist");
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
            throw std::runtime_error("[DataPacket_gpu_2_stream::pack] Bad tileDesc");
        }
 
        const RealVect      deltas = tileDesc_h->deltas();
        const IntVect       lo     = tileDesc_h->lo();
        const IntVect       hi     = tileDesc_h->hi();
        const IntVect       loGC   = tileDesc_h->loGC();
        const IntVect       hiGC   = tileDesc_h->hiGC();
        Real*               data_h = tileDesc_h->dataPtr();
        if (data_h == nullptr) {
            throw std::logic_error("[DataPacket_gpu_2_stream::pack] "
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

}
