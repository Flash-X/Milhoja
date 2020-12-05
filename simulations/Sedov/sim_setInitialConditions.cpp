//  Initializes fluid data (density, pressure, velocity, etc.) for
//  a specified block.  This version sets up the Sedov spherical
//  explosion problem.
//
//  References:  Sedov, L. I., 1959, Similarity and Dimensional Methods
//                 in Mechanics (New York:  Academic)
//               Landau, L. D., & Lifshitz, E. M., 1987, Fluid Mechanics,
//                 2d ed. (Oxford:  Pergamon)
//
// PARAMETERS
//  sim_pAmbient       Initial ambient pressure
//  sim_rhoAmbient     Initial ambient density
//  sim_expEnergy      Explosion energy (distributed over 2^dimen central zones)
//  sim_minRhoInit     Density floor for initial condition
//  sim_rInit          Radial position of inner edge of grid (for 1D )
//  sim_xctr           Explosion center coordinates
//  sim_yctr           Explosion center coordinates
//  sim_zctr           Explosion center coordinates
//  sim_nsubzones      Number of `sub-zones' in cells for applying 1d profile

#include <cmath>
#include <algorithm>
#include <iostream>

#include "Simulation.h"

#include "Grid_REAL.h"
#include "Grid.h"
#include "FArray2D.h"

#include "constants.h"
#include "Flash.h"

// Hardcoded
constexpr orchestration::Real   PI = 4.0_wp * atan(1.0_wp);
const     orchestration::Real   MIN_DIST = 1.0e-10_wp;

// Should be runtime parameters
// TODO: Should these be moved to the sim namespace in Simulation.h?
constexpr orchestration::Real    sim_xMin = X_MIN;
constexpr orchestration::Real    sim_xMax = X_MAX;
constexpr orchestration::Real    sim_yMin = Y_MIN;
constexpr orchestration::Real    sim_yMax = Y_MAX;
constexpr orchestration::Real    sim_zMin = Z_MIN;
constexpr orchestration::Real    sim_zMax = Z_MAX;
constexpr unsigned int           sim_nProfile   = 10000;                          // Simulation_data.F90
constexpr orchestration::Real    sim_gamma      = GAMMA;
constexpr orchestration::Real    sim_pAmbient   = 1.0e-5_wp;                      // test_pseudoug_2d.par
constexpr orchestration::Real    sim_rhoAmbient = 1.0_wp;                         // test_pseudoug_2d.par 
constexpr orchestration::Real    sim_expEnergy  = 1.0_wp;                         // test_pseudoug_2d.par 
constexpr orchestration::Real    sim_minRhoInit = 1.0e-20_wp;                     // Config default
constexpr orchestration::Real    sim_rInit      = 0.013671875_wp;                 // test_pseudoug_2d.par 
constexpr orchestration::Real    sim_smallRho   = 1.0e-10_wp;                     // setup_params
constexpr orchestration::Real    sim_smallP     = 1.0e-10_wp;                     // setup_params 
constexpr orchestration::Real    sim_smallT     = 1.0e-10_wp;                     // setup_params   
constexpr orchestration::Real    sim_smallE     = 1.0e-10_wp;                     // setup_params  
constexpr unsigned int           sim_nSubZones  = 7;                              // setup_params  
constexpr orchestration::Real    sim_xCenter    = 0.5_wp*(sim_xMax - sim_xMin);   // test_pseudoug_2d.par 
constexpr orchestration::Real    sim_yCenter    = 0.5_wp*(sim_yMax - sim_yMin);   // test_pseudoug_2d.par  
constexpr orchestration::Real    sim_zCenter    = 0.5_wp*(sim_zMax - sim_zMin);   // test_pseudoug_2d.par  

#if NDIM == 1
constexpr orchestration::Real   vctr = 2.0_wp * sim_rInit;
#elif NDIM == 2
constexpr orchestration::Real   vctr = PI * sim_rInit*sim_rInit;
#else
constexpr orchestration::Real   vctr = 4.0_wp / 3.0_wp * PI * sim_rInit*sim_rInit*sim_rInit;
#endif
constexpr orchestration::Real   sim_pExp = (sim_gamma - 1.0_wp) * sim_expEnergy / vctr;
constexpr orchestration::Real   sim_inSubzones = 1.0 / orchestration::Real(sim_nSubZones);

/**
  *
  */
void  sim::setInitialConditions(const orchestration::IntVect& lo,
                                const orchestration::IntVect& hi,
                                const unsigned int level,
                                const orchestration::FArray1D& xCoords,
                                const orchestration::FArray1D& yCoords,
                                const orchestration::FArray1D& zCoords,
                                const orchestration::RealVect& deltas,
                                orchestration::FArray4D& solnData) {
    // TODO: Add in ability to analytically compute initial conditions
    //       for t_0 > 0.
    setInitialConditions_topHat(lo, hi, level,
                                xCoords, yCoords, zCoords,
                                deltas, solnData);
}

/**
  *
  */
void  sim::setInitialConditions_topHat(const orchestration::IntVect& lo,
                                       const orchestration::IntVect& hi,
                                       const unsigned int level,
                                       const orchestration::FArray1D& xCoords,
                                       const orchestration::FArray1D& yCoords,
                                       const orchestration::FArray1D& zCoords,
                                       const orchestration::RealVect& deltas,
                                       orchestration::FArray4D& solnData) {
    using namespace orchestration;

    //Construct the radial samples needed for the initialization.
    Real    diagonal =        (sim_xMax - sim_xMin)
                            * (sim_xMax - sim_xMin);
    diagonal        += (K2D * (sim_yMax - sim_yMin)
                            * (sim_yMax - sim_yMin));
    diagonal        += (K3D * (sim_zMax - sim_zMin)
                            * (sim_zMax - sim_zMin));
    diagonal = sqrt(diagonal);

    Real    drProf = diagonal / (Real(sim_nProfile - 1));

    //  just use a top-hat.
    Real   rProf[sim_nProfile];
    Real   rhoProf[sim_nProfile];
    Real   pProf[sim_nProfile];
    Real   vProf[sim_nProfile];
    for (int i=0; i<sim_nProfile; ++i) {
        rProf[i]   = i * drProf;
        rhoProf[i] = sim_rhoAmbient;
        pProf[i]   = sim_pAmbient;
        vProf[i]   = 0.0_wp;
        if (rProf[i] <= sim_rInit) {
            pProf[i] = sim_pExp;
        }
    }

    Grid&   grid = Grid::instance();

    Real      dxx = deltas.I();
    Real      dyy = deltas.J() * K2D;
    Real      dzz = deltas.K() * K3D;

    Real      dvSub_buffer[sim_nSubZones * sim_nSubZones];
    FArray2D  dvSub{dvSub_buffer,
                    IntVect{LIST_NDIM(0, 0, 0)},
                    IntVect{LIST_NDIM(sim_nSubZones-1, sim_nSubZones-1, sim_nSubZones-1)}};

    unsigned int    jLo = 0;
    unsigned int    jHi = 0;

    Real            xx = 0.0;
    Real            yy = 0.0;
    Real            zz = 0.0;

    Real            xDist = 0.0;
    Real            yDist = 0.0;
    Real            zDist = 0.0;
    Real            dist = 0.0;
    Real            distInv = 0.0;
    Real            frac = 0.0;

    Real            dvc = 0.0;
    Real            quotinv = 0.0;

    Real            sumRho = 0.0;
    Real            sumP = 0.0;
    Real            sumVX = 0.0;
    Real            sumVY = 0.0;
    Real            sumVZ = 0.0;

    Real            vSub = 0.0;
    Real            rhoSub = 0.0;
    Real            pSub = 0.0;
    Real            vel = 0.0;

    Real            vx = 0.0;
    Real            vy = 0.0;
    Real            vz = 0.0;
    Real            p = 0.0;
    Real            rho = 0.0;
    Real            e = 0.0;
    Real            ek = 0.0;
    Real            eint = 0.0;

    // Assuming that:
    // - NSPECIES = 0
    // - EINT_VAR in use?
    // - BDRY_VAR in use?
    for         (int k=lo.K(); k<=hi.K(); ++k) {
        for     (int j=lo.J(); j<=hi.J(); ++j) {
            for (int i=lo.I(); i<=hi.I(); ++i) {
                // TODO: Get the volumes in a single array outside loop nest
                dvc = grid.getCellVolume(level, IntVect{LIST_NDIM(i, j, k)});

                grid.subcellGeometry(     sim_nSubZones,
                                     1 + (sim_nSubZones - 1)*K2D,
                                     1 + (sim_nSubZones - 1)*K3D,
                                     dvc, dvSub_buffer,
                                     xCoords(i) - 0.5_wp*dxx,
                                     xCoords(i) + 0.5_wp*dxx);

                // Break the cell into sim_nSubZones^NDIM sub-zones, and look up the
                // appropriate quantities along the 1d profile for that subzone.  
                // 
                // Have the final values for the zone be equal to the average of
                // the subzone values.
                sumRho = 0.0;
                sumP   = 0.0;
                sumVX  = 0.0;
                sumVY  = 0.0;
                sumVZ  = 0.0;
                for (unsigned int kk=0; kk<=(sim_nSubZones - 1)*K3D; ++kk) {
                    zz    = zCoords(k) + ((kk + 0.5_wp) * sim_inSubzones - 0.5_wp) * dzz;
                    zDist = (zz - sim_zCenter) * K3D;

                    for (unsigned int jj=0; jj<=(sim_nSubZones - 1)*K2D; ++jj) {
                        yy    = yCoords(j) + ((jj + 0.5_wp) * sim_inSubzones - 0.5_wp) * dyy;
                        yDist = (yy - sim_yCenter) * K2D;

                        for (unsigned int ii=0; ii<sim_nSubZones; ++ii) {
                            xx    = xCoords(i) + ((ii + 0.5_wp) * sim_inSubzones - 0.5_wp) * dxx;
                            xDist = xx - sim_xCenter;

                            dist    = sqrt(xDist*xDist + yDist*yDist + zDist*zDist);
                            distInv = 1.0_wp / std::max(dist, MIN_DIST);
 
                            // jLo is index into rProf of largest r value in
                            // profile less than or equal to dist so that
                            //        rProf[jLo] <= dist < rProf[jHi].
                            //
                            // rProf starts at zero and ends at diagonal.
                            // It is corresponds to sim_nProfile equally-spaced
                            // mesh points.  Note that
                            //       dist < diagonal
                            // since xx,yy,zz are coordinates to the center of
                            // subzones and diagonal is based on the extreme
                            // corners of the domain.  Therefore,
                            //       0   <= jLo <  sim_nProfile - 1
                            // and
                            //       jLo <  jHi <= sim_nProfile - 1.
                            jLo = floor( dist / drProf );
                            jHi = jLo + 1;
                            frac = (dist - rProf[jLo]) / drProf;
                            if (jHi >= sim_nProfile) {
                                throw std::runtime_error("jHi search failed!");
                            } else if ((rProf[jLo] > dist) || (rProf[jHi] <= dist)) {
                                throw std::runtime_error("What the blurg?!");
                            }

                            // a point at `dist' is frac-way between jLo and jHi.   We do a
                            // linear interpolation of the quantities at jLo and jHi and sum those.
                            pSub   =   pProf[jLo] + frac * (  pProf[jHi] -   pProf[jLo]);
                            rhoSub = rhoProf[jLo] + frac * (rhoProf[jHi] - rhoProf[jLo]);
                            vSub   =   vProf[jLo] + frac * (  vProf[jHi] -   vProf[jLo]);
                            rhoSub = std::max(rhoSub, sim_minRhoInit);

                            //   Now total these quantities.
                            sumP   += (  pSub * dvSub(ii, jj));
                            sumRho += (rhoSub * dvSub(ii, jj));

                            // Note that the velocities are radial based on the
                            // assumed symmetries of the initial conditions.
                            // Therefore, the velocity vector at a point must be
                            // parallel to the position vector of the point.
                            // Similar triangles gives the components:
                            vel = vSub * dvSub(ii, jj);
                            sumVX += (vel * xDist * distInv);
                            sumVY += (vel * yDist * distInv);
                            sumVZ += (vel * zDist * distInv);
                        }
                    }
                }

                quotinv = 1.0_wp / dvc;
                rho = std::max(sumRho * quotinv, sim_smallRho);
                p   = std::max(sumP   * quotinv, sim_smallP);
                vx  = sumVX * quotinv;
                vy  = sumVY * quotinv;
                vz  = sumVZ * quotinv;
                ek  = 0.5_wp * (vx*vx + vy*vy + vz*vz);

                // Assume a gamma-law equation of state
                e    = p / (sim_gamma - 1.0_wp);
                eint = e / rho;
                e    = e / rho + ek;
                e    = std::max(e, sim_smallE);

//                solnData(i, j, k, DENS_VAR_C) = rho;
                solnData(i, j, k, DENS_VAR_C) = 1.0_wp;
                solnData(i, j, k, PRES_VAR_C) = p;
                solnData(i, j, k, ENER_VAR_C) = e;
#ifdef EINT_VAR_C
                solnData(i, j, k, EINT_VAR_C) = eint;
#endif
                solnData(i, j, k, GAME_VAR_C) = sim_gamma;
                solnData(i, j, k, GAMC_VAR_C) = sim_gamma;
                solnData(i, j, k, VELX_VAR_C) = vx;
                solnData(i, j, k, VELY_VAR_C) = vy;
                solnData(i, j, k, VELZ_VAR_C) = vz;
                solnData(i, j, k, TEMP_VAR_C) = sim_smallT;
#ifdef BDRY_VAR_C
                solnData(i, j, k, BDRY_VAR_C) = -1.0_wp;
#endif 
            }
        }
    }
}

