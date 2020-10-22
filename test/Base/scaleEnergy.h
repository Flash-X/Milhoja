#ifndef SCALE_ENERGY_H__
#define SCALE_ENERGY_H__

#include "DataItem.h"
#include "Grid_IntVect.h"
#include "FArray1D.h"
#include "FArray4D.h"

namespace StaticPhysicsRoutines {
    // This is the declaration/interface as designed by a PUD.  It should not
    // have unnecessary variables (e.g. stream) and should be written under the
    // assumption that the code will be executed by the host with all given
    // arguments resident in host memory.
    void scaleEnergy(const orchestration::IntVect& lo,
                     const orchestration::IntVect& hi,
                     const orchestration::FArray1D& xCoords,
                     const orchestration::FArray1D& yCoords,
                     orchestration::FArray4D& U,
                     const orchestration::Real scaleFactor);
}

namespace ActionRoutines {
    void scaleEnergy_tile_cpu(const int tId, orchestration::DataItem* dataItem);
}

#ifdef ENABLE_OPENACC_OFFLOAD
namespace StaticPhysicsRoutines {
    // This is the version that would be adapted from the above by the 
    // OFFLINE TOOLCHAIN based on the target platfarm and high-level
    // offloading program model that has been specified for the build.
    // In this case, we imagine that offloading, where specified, should
    // be effected by OpenACC and that the simulation will run on Summit.
    // Since offloading, we pass in all arguments as pointers to data in
    // device memory.  The exceptions to this are
    //  - queue_h as this information is needed on the host for offloading
    //  - scaleFactor as this variable has not yet been included in
    //    the host-to-device data packet (pending).
    #pragma acc routine vector
    void scaleEnergy_oacc_summit(const orchestration::IntVect* lo_d,
                                 const orchestration::IntVect* hi_d,
                                 const orchestration::FArray1D* xCoords_d,
                                 const orchestration::FArray1D* yCoords_d,
                                 orchestration::FArray4D* U_d,
                                 const orchestration::Real scaleFactor);
}

namespace ActionRoutines {
    void scaleEnergy_packet_oacc_summit(const int tId,
                                        orchestration::DataItem* dataItem);
}
#endif

#endif

