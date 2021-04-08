#include "DataPacket_gpu_1.h"

#include <cassert>
#include <cstring>

#include "Grid_IntVect.h"
#include "Grid_RealVect.h"
#include "Grid_Axis.h"
#include "Grid_Edge.h"
#include "Grid.h"
#include "FArray1D.h"
#include "FArray4D.h"
#include "Backend.h"

#include "Driver.h"

#include "Flash.h"

namespace orchestration {

/**
 *
 */
std::unique_ptr<DataPacket>   DataPacket_gpu_1::clone(void) const {
    return std::unique_ptr<DataPacket>{new DataPacket_gpu_1{}};
}

/**
 *
 */
void  DataPacket_gpu_1::pack(void) {
    std::string   errMsg = isNull();
    if (errMsg != "") {
        throw std::logic_error("[DataPacket_gpu_1::pack] " + errMsg);
    } else if (tiles_.size() == 0) {
        throw std::logic_error("[DataPacket_gpu_1::pack] No tiles added");
    }

    Grid&   grid = Grid::instance();

    // TODO: Deltas should just be placed into the packet once.
    std::size_t  nTiles = tiles_.size();
    std::size_t  nBytesPerPacket =            sizeof(std::size_t) 
                                   + nTiles * sizeof(PacketContents)
                                   + nTiles * (         1 * DELTA_SIZE_BYTES
                                               +        2 * POINT_SIZE_BYTES
                                               + N_BLOCKS * CC_BLOCK_SIZE_BYTES
                                               + N_BLOCKS * ARRAY4_SIZE_BYTES);

    stream_ = Backend::instance().requestStream(true);
    if (!stream_.isValid()) {
        throw std::runtime_error("[DataPacket_gpu_1::pack] Unable to acquire stream");
    }

    // Allocate memory in pinned and device memory on demand for now
    Backend::instance().requestGpuMemory(nBytesPerPacket, &packet_p_,
                                         nBytesPerPacket, &packet_d_);

    // Store for later unpacking the location in pinned memory of the different
    // blocks.
    if (pinnedPtrs_) {
        throw std::logic_error("[DataPacket_gpu_1::pack] Pinned pointers already exist");
    }
    pinnedPtrs_ = new BlockPointersPinned[nTiles];

    // TODO: The test that uses this packet is not yet used for timing purposes.
    // Therefore, we can get away with a suboptimal packet design.
    //
    // This presently ships over more than what is needed and moves all data
    // both back and forth.  Optimiza when there is time.  Use Sedov packet as
    // an example.
    nCopyToGpuBytes_    = nBytesPerPacket;
    nReturnToHostBytes_ = nBytesPerPacket;
    copyInStart_p_    = static_cast<char*>(packet_p_);
    copyInStart_d_    = static_cast<char*>(packet_d_);
    copyInOutStart_p_ = static_cast<char*>(packet_p_);
    copyInOutStart_d_ = static_cast<char*>(packet_d_);

    // Pointer to the next free byte in the current data packets
    // Should be true by C++ standard
    static_assert(sizeof(char) == 1, "Invalid char size");
    char*   ptr_p = static_cast<char*>(packet_p_);
    char*   ptr_d = static_cast<char*>(packet_d_);

    nTiles_d_ = static_cast<std::size_t*>((void*)ptr_d); 
    std::memcpy((void*)ptr_p, (void*)&nTiles, sizeof(std::size_t));
    ptr_p += sizeof(std::size_t);
    ptr_d += sizeof(std::size_t);

    contents_p_ = static_cast<PacketContents*>((void*)ptr_p);
    contents_d_ = static_cast<PacketContents*>((void*)ptr_d);
    ptr_p += nTiles * sizeof(PacketContents);
    ptr_d += nTiles * sizeof(PacketContents);

    PacketContents*   tilePtrs_p = contents_p_;
    for (std::size_t n=0; n<nTiles; ++n, ++tilePtrs_p) {
        Tile*   tileDesc_h = tiles_[n].get();
        if (tileDesc_h == nullptr) {
            throw std::runtime_error("[DataPacket_gpu_1::pack] Bad tileDesc");
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
        if (data_h == nullptr) {
            throw std::logic_error("[DataPacket_gpu_1::pack] "
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

        location_ = PacketDataLocation::CC1;
        pinnedPtrs_[n].CC1_data = static_cast<Real*>((void*)ptr_p);
        CC1_data_d  = static_cast<Real*>((void*)ptr_d);
        std::memcpy((void*)ptr_p, (void*)data_h, CC_BLOCK_SIZE_BYTES);
        ptr_p += CC_BLOCK_SIZE_BYTES;
        ptr_d += CC_BLOCK_SIZE_BYTES;

        pinnedPtrs_[n].CC2_data = static_cast<Real*>((void*)ptr_p);
        CC2_data_d  = static_cast<Real*>((void*)ptr_d);
        ptr_p += CC_BLOCK_SIZE_BYTES;
        ptr_d += CC_BLOCK_SIZE_BYTES;

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
    }
}

}

