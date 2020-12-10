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
                                                 orchestration::DataItem* dataItem);
    void advanceSolutionHll_packet_oacc_summit_2(const int tId,
                                                 orchestration::DataItem* dataItem);
    void advanceSolutionHll_packet_oacc_summit_3(const int tId,
                                                 orchestration::DataItem* dataItem);
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
    void computeSoundSpeedHll_oacc_summit(const orchestration::IntVect& lo,
                                          const orchestration::IntVect& hi,
                                          const orchestration::FArray4D& U,
                                          orchestration::FArray3D& auxC);

    void computeFluxesHll_X_oacc_summit(const orchestration::Real dt,
                                        const orchestration::IntVect& lo,
                                        const orchestration::IntVect& hi,
                                        const orchestration::RealVect& deltas,
                                        const orchestration::FArray4D& U,
                                        orchestration::FArray4D& flX,
                                        const orchestration::FArray3D& auxC);

    void computeFluxesHll_Y_oacc_summit(const orchestration::Real dt,
                                        const orchestration::IntVect& lo,
                                        const orchestration::IntVect& hi,
                                        const orchestration::RealVect& deltas,
                                        const orchestration::FArray4D& U,
                                        orchestration::FArray4D& flY,
                                        const orchestration::FArray3D& auxC);

    void computeFluxesHll_Z_oacc_summit(const orchestration::Real dt,
                                        const orchestration::IntVect& lo,
                                        const orchestration::IntVect& hi,
                                        const orchestration::RealVect& deltas,
                                        const orchestration::FArray4D& U,
                                        orchestration::FArray4D& flZ,
                                        const orchestration::FArray3D& auxC);

    // Kernels that compose update solution
    // TODO: Conceptually, these should be a single kernel.  Remains to be seen
    // if they should be broken up to limit register pressure.  I think that
    // each variable should have its own kernel as this might keep register
    // pressure low and make better use of data locality.
    //
    // Kernel-decomposition I -
    // apply one operation at a time across all variables
    void scaleSolutionHll_oacc_summit(const orchestration::IntVect& lo,
                                      const orchestration::IntVect& hi,
                                      const orchestration::FArray4D& Uin,
                                      orchestration::FArray4D& Uout);

    void updateSolutionHll_FlX_oacc_summit(const orchestration::IntVect& lo,
                                           const orchestration::IntVect& hi,
                                           orchestration::FArray4D& U,
                                           const orchestration::FArray4D& flX);

    void updateSolutionHll_FlY_oacc_summit(const orchestration::IntVect& lo,
                                           const orchestration::IntVect& hi,
                                           orchestration::FArray4D& U,
                                           const orchestration::FArray4D& flY);

    void updateSolutionHll_FlZ_oacc_summit(const orchestration::IntVect& lo,
                                           const orchestration::IntVect& hi,
                                           orchestration::FArray4D& U,
                                           const orchestration::FArray4D& flZ);

    void rescaleSolutionHll_oacc_summit(const orchestration::IntVect& lo,
                                        const orchestration::IntVect& hi,
                                        orchestration::FArray4D& U);

    void computeEintHll_oacc_summit(const orchestration::IntVect& lo,
                                    const orchestration::IntVect& hi,
                                    orchestration::FArray4D& U);

    // Kernel-decomposition II -
    // Updated each variable individually
    void updateDensityHll_oacc_summit(const orchestration::IntVect& lo,
                                      const orchestration::IntVect& hi,
                                      const orchestration::FArray4D& Uin,
                                      orchestration::FArray4D& Uout,
                                      const orchestration::FArray4D& flX,
                                      const orchestration::FArray4D& flY,
                                      const orchestration::FArray4D& flZ);

    void updateVelxHll_oacc_summit(const orchestration::IntVect& lo,
                                   const orchestration::IntVect& hi,
                                   const orchestration::FArray4D& Uin,
                                   orchestration::FArray4D& Uout,
                                   const orchestration::FArray4D& flX,
                                   const orchestration::FArray4D& flY,
                                   const orchestration::FArray4D& flZ);

    void updateVelyHll_oacc_summit(const orchestration::IntVect& lo,
                                   const orchestration::IntVect& hi,
                                   const orchestration::FArray4D& Uin,
                                   orchestration::FArray4D& Uout,
                                   const orchestration::FArray4D& flX,
                                   const orchestration::FArray4D& flY,
                                   const orchestration::FArray4D& flZ);

    void updateVelzHll_oacc_summit(const orchestration::IntVect& lo,
                                   const orchestration::IntVect& hi,
                                   const orchestration::FArray4D& Uin,
                                   orchestration::FArray4D& Uout,
                                   const orchestration::FArray4D& flX,
                                   const orchestration::FArray4D& flY,
                                   const orchestration::FArray4D& flZ);
    void updateEnergyHll_oacc_summit(const orchestration::IntVect& lo,
                                     const orchestration::IntVect& hi,
                                     const orchestration::FArray4D& Uin,
                                     orchestration::FArray4D& Uout,
                                     const orchestration::FArray4D& flX,
                                     const orchestration::FArray4D& flY,
                                     const orchestration::FArray4D& flZ);

    // Kernel-decomposition III -
    // One giant kernel!
    void updateSolutionHll_oacc_summit(const orchestration::IntVect& lo,
                                       const orchestration::IntVect& hi,
                                       orchestration::FArray4D& U,
                                       const orchestration::FArray4D& flX,
                                       const orchestration::FArray4D& flY,
                                       const orchestration::FArray4D& flZ);
};

#endif

