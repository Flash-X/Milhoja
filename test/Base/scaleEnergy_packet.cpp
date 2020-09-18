#include "scaleEnergy_packet.h"

#include "Grid_IntVect.h"
#include "Grid_RealVect.h"
#include "FArray1D.h"
#include "FArray4D.h"
#include "DataItem.h"

#include "constants.h"
#include "scaleEnergy_block.h"

void ThreadRoutines::scaleEnergy_packet(const int tId, void* dataItem) {
    using namespace orchestration;

    constexpr std::size_t    N_CELLS =   (NXB + 2 * NGUARD * K1D)
                                       * (NYB + 2 * NGUARD * K2D)
                                       * (NZB + 2 * NGUARD * K3D)
                                       * NUNKVAR;
    constexpr std::size_t    N_BYTES_PER_BLOCK   = N_CELLS * sizeof(Real);
    constexpr std::size_t    COORDS_X_SIZE_BYTES =     NXB * sizeof(Real);
    constexpr std::size_t    COORDS_Y_SIZE_BYTES =     NYB * sizeof(Real);

    DataItem*  packet = reinterpret_cast<DataItem*>(dataItem);
    const int  streamId = packet->stream().id;

    // Use the host to determine pointer offsets into data packet in device
    // memory.  This should work so long as these pointers aren't used 
    // before the data is actually in the device memory.
    char*  p = static_cast<char*>(packet->gpuPointer());
//    const RealVect*  deltas_d = reinterpret_cast<RealVect*>(p);
    p += sizeof(RealVect);
    const IntVect*   lo_d = reinterpret_cast<IntVect*>(p);
    p += sizeof(IntVect);
    const IntVect*   hi_d = reinterpret_cast<IntVect*>(p);
    p += sizeof(IntVect);
//    const int*   loGC_d = reinterpret_cast<int*>(p);
    p += sizeof(IntVect);
//    const int*   hiGC_d = reinterpret_cast<int*>(p);
    p += sizeof(IntVect);
    // No need to get pointer to data in device since we use the following
    // object to access the data and it is already configured with this
    // pointer.
    p += 2 * N_BYTES_PER_BLOCK;
    FArray4D*   f_d = reinterpret_cast<FArray4D*>(p);
    p += sizeof(FArray4D);
//    FArray4D*   scratch_d = reinterpret_cast<FArray4D*>(p);
    p += sizeof(FArray4D);
    p += COORDS_X_SIZE_BYTES;
    p += COORDS_Y_SIZE_BYTES;
    FArray1D*   xCoords_d = reinterpret_cast<FArray1D*>(p);
    p += sizeof(FArray1D);
    FArray1D*   yCoords_d = reinterpret_cast<FArray1D*>(p);

    scaleEnergy_block(lo_d, hi_d, xCoords_d, yCoords_d, f_d, streamId);
}

