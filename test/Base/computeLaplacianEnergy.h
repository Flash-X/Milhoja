#ifndef COMPUTE_LAPLACIAN_ENERGY_H__
#define COMPUTE_LAPLACIAN_ENERGY_H__

#include "DataItem.h"
#include "Grid_IntVect.h"
#include "Grid_RealVect.h"
#include "FArray4D.h"

namespace StaticPhysicsRoutines {
    void computeLaplacianEnergy(const orchestration::IntVect& lo,
                                const orchestration::IntVect& hi,
                                orchestration::FArray4D& U,
                                orchestration::FArray4D& scratch,
                                const orchestration::RealVect& deltas);

    void computeLaplacianEnergy_oacc_summit(const orchestration::IntVect* lo_d,
                                            const orchestration::IntVect* hi_d,
                                            orchestration::FArray4D* Uin_d,
                                            orchestration::FArray4D* Uout_d,
                                            const orchestration::RealVect* deltas_d,
                                            const int queue_h);
}

namespace ActionRoutines {
    void computeLaplacianEnergy_tile_cpu(const int tId,
                                         orchestration::DataItem* dataItem);
    void computeLaplacianEnergy_packet_oacc_summit(const int tId,
                                                   orchestration::DataItem* dataItem);
}

#endif

