#include "gpuKernel.h"

#include <cstdlib>

#include "Grid_IntVect.h"
#include "Grid_REAL.h"
#include "FArray4D.h"
#include "CudaDataPacket.h"

#include "constants.h"

using namespace orchestration;

void   gpuKernel::kernel(const int tId, void* dataItem) {
    constexpr std::size_t    N_CELLS =   (NXB + 2 * NGUARD * K1D)
                                       * (NYB + 2 * NGUARD * K2D)
                                       * (NZB + 2 * NGUARD * K3D)
                                       * NUNKVAR;
    constexpr std::size_t    N_BYTES_PER_BLOCK = N_CELLS * sizeof(Real);

    CudaDataPacket*  packet = reinterpret_cast<CudaDataPacket*>(dataItem);
    int              streamId = packet->stream().id;

    // Use the host to determine pointer offsets into data packet in device
    // memory.  This should work so long as these pointers aren't used 
    // before the data is actually in the device memory.
    char*  p = static_cast<char*>(packet->gpuPointer());
//    const Real*  deltas_d = reinterpret_cast<Real*>(p);
    p += MDIM * sizeof(Real);
//    const IntVect*  lo_d = reinterpret_cast<IntVect*>(p);
    p += sizeof(IntVect);
//    const IntVect*  hi_d = reinterpret_cast<IntVect*>(p);
    p += sizeof(IntVect);
    const IntVect*  loGC_d = reinterpret_cast<IntVect*>(p);
    p += sizeof(IntVect);
    const IntVect*  hiGC_d = reinterpret_cast<IntVect*>(p);
    p += sizeof(IntVect);
    // No need to get pointer to data in device since we use the following
    // object to access the data and it is already configured with this
    // pointer.
    p += 2 * N_BYTES_PER_BLOCK;
    FArray4D*   f_d = reinterpret_cast<FArray4D*>(p);
    p += sizeof(FArray4D);
    FArray4D*   scratch_d = reinterpret_cast<FArray4D*>(p);

    // This kernel is queued up in the given stream and ostensibly behind the
    // asynchronous transfer of the given data packet from host-to-device.
    // The use of the stream will ensure that this kernel does not start until
    // the data is present in the device.
    //
    // NOTE: This routine is not written to block program execution until the
    // kernel computation finishes.  Rather, it is assumed that the code that
    // called this function and provided the stream will manage the
    // sychronization of the host thread with the work on this stream.
    #pragma acc data deviceptr(loGC_d, hiGC_d, f_d)
    {
        #pragma acc parallel loop default(none) async(streamId)
        for         (int k=loGC_d->K(); k<=hiGC_d->K(); ++k) {
            #pragma acc loop
            for     (int j=loGC_d->J(); j<=hiGC_d->J(); ++j) {
                #pragma acc loop
                for (int i=loGC_d->I(); i<=hiGC_d->I(); ++i) {
                    f_d->at(i, j, k, DENS_VAR_C) += 2.1 * j;
                    f_d->at(i, j, k, ENER_VAR_C) -= i;
                }
            }
        }
        #pragma acc wait(streamId)
    }
}

