#ifndef COMPUTE_LAPLACIAN_FUSED_H__
#define COMPUTE_LAPLACIAN_FUSED_H__

#include <Milhoja.h>
#include <Milhoja_DataItem.h>
#include <Milhoja_IntVect.h>
#include <Milhoja_RealVect.h>
#include <Milhoja_FArray4D.h>

namespace StaticPhysicsRoutines{
    void computeLaplacianFusedKernels(const milhoja::IntVect& lo,
                                      const milhoja::IntVect& hi,
                                      milhoja::FArray4D& U,
                                      milhoja::FArray4D& scratch,
                                      const milhoja::RealVect& deltas);
}

namespace ActionRoutines {
    void computeLaplacianFusedKernels_tile_cpu(const int tId,
                                               milhoja::DataItem* dataItem);
}

#ifdef MILHOJA_OPENACC_OFFLOADING
namespace StaticPhysicsRoutines{
    #pragma acc routine vector
    void computeLaplacianFusedKernels_oacc_summit(const milhoja::IntVect* lo_d,
                                                  const milhoja::IntVect* hi_d,
                                                  const milhoja::FArray4D* Uin_d,
                                                  milhoja::FArray4D* Uout_d,
                                                  const milhoja::RealVect* deltas_d);
}

namespace ActionRoutines {
    void computeLaplacianFusedKernelsStrong_packet_oacc_summit(const int tId,
                                                               milhoja::DataItem* dataItem);
    void computeLaplacianFusedKernelsWeak_packet_oacc_summit(const int tId,
                                                             milhoja::DataItem* dataItem);
    void computeLaplacianFusedActions_packet_oacc_summit(const int tId,
                                                         milhoja::DataItem* dataItem);
}
#endif

#endif

