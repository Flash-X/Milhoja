#ifndef HYDRO_H__
#define HYDRO_H__

#include "Grid_REAL.h"
#include "Grid_IntVect.h"
#include "Grid_RealVect.h"
#include "FArray3D.h"
#include "FArray4D.h"
#include "DataItem.h"

namespace Hydro {
    //----- ORCHESTRATION RUNTIME ACTION ROUTINES
    void advanceSolutionHll_tile_cpu(const int tId,
                                     orchestration::DataItem* dataItem);
    void advanceSolutionHll_packet_oacc_summit_1(const int tId,
                                                 orchestration::DataItem* dataItem_h);
    void advanceSolutionHll_packet_oacc_summit_2(const int tId,
                                                 orchestration::DataItem* dataItem_h);
    void advanceSolutionHll_packet_oacc_summit_3(const int tId,
                                                 orchestration::DataItem* dataItem_h);
};

namespace hy {
    //----- CPU ACTION ROUTINES
    void computeFluxesHll(const orchestration::Real dt,
                          const orchestration::IntVect& lo,
                          const orchestration::IntVect& hi,
                          const orchestration::RealVect& deltas,
                          const orchestration::FArray4D& U,
                          orchestration::FArray4D& flX,
                          orchestration::FArray4D& flY,
                          orchestration::FArray4D& flZ,
                          orchestration::FArray3D& auxC);

    void updateSolutionHll(const orchestration::IntVect& lo,
                           const orchestration::IntVect& hi,
                           orchestration::FArray4D& U,
                           const orchestration::FArray4D& flX,
                           const orchestration::FArray4D& flY,
                           const orchestration::FArray4D& flZ);

    //----- GPU ACTION ROUTINES FOR ADVANCE SOLUTION HYDRO OPERATION
    // Kernels that compose compute flux
    #pragma acc routine vector
    void computeSoundSpeedHll_oacc_summit(const orchestration::IntVect* lo_d,
                                          const orchestration::IntVect* hi_d,
                                          const orchestration::FArray4D* U_d,
                                          orchestration::FArray4D* auxC_d);

    #pragma acc routine vector
    void computeFluxesHll_X_oacc_summit(const orchestration::Real* dt_d,
                                        const orchestration::IntVect* lo_d,
                                        const orchestration::IntVect* hi_d,
                                        const orchestration::RealVect* deltas_d,
                                        const orchestration::FArray4D* U_d,
                                        orchestration::FArray4D* flX_d,
                                        const orchestration::FArray4D* auxC_d);

    #pragma acc routine vector
    void computeFluxesHll_Y_oacc_summit(const orchestration::Real* dt_d,
                                        const orchestration::IntVect* lo_d,
                                        const orchestration::IntVect* hi_d,
                                        const orchestration::RealVect* deltas_d,
                                        const orchestration::FArray4D* U_d,
                                        orchestration::FArray4D* flY_d,
                                        const orchestration::FArray4D* auxC_d);

    #pragma acc routine vector
    void computeFluxesHll_Z_oacc_summit(const orchestration::Real* dt_d,
                                        const orchestration::IntVect* lo_d,
                                        const orchestration::IntVect* hi_d,
                                        const orchestration::RealVect* deltas_d,
                                        const orchestration::FArray4D* U_d,
                                        orchestration::FArray4D* flZ_d,
                                        const orchestration::FArray4D* auxC_d);

    // Kernels that compose update solution
    // Conceptually, this should be a single kernel.  However, it could be that
    // such an implementation would require many registers and lead to
    // counterproductive register pressure.  Therefore, we offer two
    // decompositions of this action to test out lower register counts/kernel
    // and the possibility of executing small kernels concurrently in the GPU.
    //
    // Kernel-decomposition I -
    // apply one operation at a time across all variables
    #pragma acc routine vector
    void scaleSolutionHll_oacc_summit(const orchestration::IntVect* lo_d,
                                      const orchestration::IntVect* hi_d,
                                      const orchestration::FArray4D* Uin_d,
                                      orchestration::FArray4D* Uout_d);

    #pragma acc routine vector
    void updateSolutionHll_FlX_oacc_summit(const orchestration::IntVect* lo_d,
                                           const orchestration::IntVect* hi_d,
                                           orchestration::FArray4D* U_d,
                                           const orchestration::FArray4D* flX_d);

    #pragma acc routine vector
    void updateSolutionHll_FlY_oacc_summit(const orchestration::IntVect* lo_d,
                                           const orchestration::IntVect* hi_d,
                                           orchestration::FArray4D* U_d,
                                           const orchestration::FArray4D* flY_d);

    #pragma acc routine vector
    void updateSolutionHll_FlZ_oacc_summit(const orchestration::IntVect* lo_d,
                                           const orchestration::IntVect* hi_d,
                                           orchestration::FArray4D* U_d,
                                           const orchestration::FArray4D* flZ_d);

    #pragma acc routine vector
    void rescaleSolutionHll_oacc_summit(const orchestration::IntVect* lo_d,
                                        const orchestration::IntVect* hi_d,
                                        orchestration::FArray4D* U_d);

    #pragma acc routine vector
    void computeEintHll_oacc_summit(const orchestration::IntVect* lo_d,
                                    const orchestration::IntVect* hi_d,
                                    orchestration::FArray4D* U_d);

    // Kernel-decomposition II -
    // Updated each variable individually
    #pragma acc routine vector
    void updateDensityHll_oacc_summit(const orchestration::IntVect* lo_d,
                                      const orchestration::IntVect* hi_d,
                                      const orchestration::FArray4D* Uin_d,
                                      orchestration::FArray4D* Uout_d,
                                      const orchestration::FArray4D* flX_d,
                                      const orchestration::FArray4D* flY_d,
                                      const orchestration::FArray4D* flZ_d);

    #pragma acc routine vector
    void updateVelxHll_oacc_summit(const orchestration::IntVect* lo_d,
                                   const orchestration::IntVect* hi_d,
                                   const orchestration::FArray4D* Uin_d,
                                   orchestration::FArray4D* Uout_d,
                                   const orchestration::FArray4D* flX_d,
                                   const orchestration::FArray4D* flY_d,
                                   const orchestration::FArray4D* flZ_d);

    #pragma acc routine vector
    void updateVelyHll_oacc_summit(const orchestration::IntVect* lo_d,
                                   const orchestration::IntVect* hi_d,
                                   const orchestration::FArray4D* Uin_d,
                                   orchestration::FArray4D* Uout_d,
                                   const orchestration::FArray4D* flX_d,
                                   const orchestration::FArray4D* flY_d,
                                   const orchestration::FArray4D* flZ_d);

    #pragma acc routine vector
    void updateVelzHll_oacc_summit(const orchestration::IntVect* lo_d,
                                   const orchestration::IntVect* hi_d,
                                   const orchestration::FArray4D* Uin_d,
                                   orchestration::FArray4D* Uout_d,
                                   const orchestration::FArray4D* flX_d,
                                   const orchestration::FArray4D* flY_d,
                                   const orchestration::FArray4D* flZ_d);

    #pragma acc routine vector
    void updateEnergyHll_oacc_summit(const orchestration::IntVect* lo_d,
                                     const orchestration::IntVect* hi_d,
                                     const orchestration::FArray4D* Uin_d,
                                     orchestration::FArray4D* Uout_d,
                                     const orchestration::FArray4D* flX_d,
                                     const orchestration::FArray4D* flY_d,
                                     const orchestration::FArray4D* flZ_d);

    // Kernel-decomposition III -
    // One giant kernel!
    #pragma acc routine vector
    void updateSolutionHll_oacc_summit(const orchestration::IntVect* lo_d,
                                       const orchestration::IntVect* hi_d,
                                       const orchestration::FArray4D* Uin_d,
                                       orchestration::FArray4D* Uout_d,
                                       const orchestration::FArray4D* flX_d,
                                       const orchestration::FArray4D* flY_d,
                                       const orchestration::FArray4D* flZ_d);
};

#endif

