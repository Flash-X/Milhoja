#ifndef COMPUTE_LAPLACIAN_ENERGY_H__
#define COMPUTE_LAPLACIAN_ENERGY_H__

#include <Milhoja.h>
#include <Milhoja_DataItem.h>
#include <Milhoja_IntVect.h>
#include <Milhoja_RealVect.h>
#include <Milhoja_FArray4D.h>

namespace StaticPhysicsRoutines {
    void computeLaplacianEnergy(const milhoja::IntVect& lo,
                                const milhoja::IntVect& hi,
                                milhoja::FArray4D& U,
                                milhoja::FArray4D& scratch,
                                const milhoja::RealVect& deltas);
}

namespace ActionRoutines {
    void computeLaplacianEnergy_tile_cpu(const int tId,
                                         milhoja::DataItem* dataItem);
}

#ifdef MILHOJA_OPENACC_OFFLOADING
namespace StaticPhysicsRoutines {
    #pragma acc routine vector
    void computeLaplacianEnergy_oacc_summit(const milhoja::IntVect* lo_d,
                                            const milhoja::IntVect* hi_d,
                                            const milhoja::FArray4D* Uin_d,
                                            milhoja::FArray4D* Uout_d,
                                            const milhoja::RealVect* deltas_d);
}

namespace ActionRoutines {
    void computeLaplacianEnergy_packet_oacc_summit(const int tId,
                                                   milhoja::DataItem* dataItem);
}
#endif

#endif

