#ifndef COMPUTE_LAPLACIAN_FUSED_H__
#define COMPUTE_LAPLACIAN_FUSED_H__

#include "DataItem.h"
#include "Grid_IntVect.h"
#include "Grid_RealVect.h"
#include "FArray4D.h"

namespace StaticPhysicsRoutines{
    void computeLaplacianFusedKernels(const orchestration::IntVect& lo,
                                      const orchestration::IntVect& hi,
                                      orchestration::FArray4D& U,
                                      orchestration::FArray4D& scratch,
                                      const orchestration::RealVect& deltas);
}

namespace ActionRoutines {
    void computeLaplacianFusedKernels_tile_cpu(const int tId,
                                               orchestration::DataItem* dataItem);
}

#ifdef ENABLE_OPENACC_OFFLOAD
namespace StaticPhysicsRoutines{
    #pragma acc routine vector
    void computeLaplacianFusedKernels_oacc_summit(const orchestration::IntVect* lo_d,
                                                  const orchestration::IntVect* hi_d,
                                                  const orchestration::FArray4D* Uin_d,
                                                  orchestration::FArray4D* Uout_d,
                                                  const orchestration::RealVect* deltas_d);
}

namespace ActionRoutines {
    void computeLaplacianFusedKernelsStrong_packet_oacc_summit(const int tId,
                                                               orchestration::DataItem* dataItem);
    void computeLaplacianFusedKernelsWeak_packet_oacc_summit(const int tId,
                                                             orchestration::DataItem* dataItem);
    void computeLaplacianFusedActions_packet_oacc_summit(const int tId,
                                                         orchestration::DataItem* dataItem);
}
#endif

#endif

