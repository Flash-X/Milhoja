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

#include "Simulation.h"

#include <cmath>
#include <algorithm>
#include <iostream>

#include <milhoja.h>
#include <Milhoja_real.h>
#include <Milhoja_Grid.h>
#include <Milhoja_FArray2D.h>

#include "Sedov.h"

#include "Flash_par.h"

// Hardcoded
const     milhoja::Real   MIN_DIST = 1.0e-10_wp;

/**
  *
  */
void  sim::setInitialConditions(const milhoja::IntVect& lo,
                                const milhoja::IntVect& hi,
                                const unsigned int level,
                                const milhoja::FArray1D& xCoords,
                                const milhoja::FArray1D& yCoords,
                                const milhoja::FArray1D& zCoords,
                                const milhoja::RealVect& deltas,
                                milhoja::FArray4D& solnData) {
    // TODO: Add in ability to analytically compute initial conditions
    //       for t_0 > 0.
    setInitialConditions_topHat(lo, hi, level,
                                xCoords, yCoords, zCoords,
                                deltas, solnData);
}

/**
  *
  */
void  sim::setInitialConditions_topHat(const milhoja::IntVect& lo,
                                       const milhoja::IntVect& hi,
                                       const unsigned int level,
                                       const milhoja::FArray1D& xCoords,
                                       const milhoja::FArray1D& yCoords,
                                       const milhoja::FArray1D& zCoords,
                                       const milhoja::RealVect& deltas,
                                       milhoja::FArray4D& solnData) {
    using namespace milhoja;

    //Construct the radial samples needed for the initialization.
    Real    diagonal =        (rp_Grid::X_MAX - rp_Grid::X_MIN)
                            * (rp_Grid::X_MAX - rp_Grid::X_MIN);
    diagonal        += (K2D * (rp_Grid::Y_MAX - rp_Grid::Y_MIN)
                            * (rp_Grid::Y_MAX - rp_Grid::Y_MIN));
    diagonal        += (K3D * (rp_Grid::Z_MAX - rp_Grid::Z_MIN)
                            * (rp_Grid::Z_MAX - rp_Grid::Z_MIN));
    diagonal = sqrt(diagonal);

    Real    drProf = diagonal / (Real(rp_Simulation::N_PROFILE - 1));

    //  just use a top-hat.
    Real   rProf[rp_Simulation::N_PROFILE];
    Real   rhoProf[rp_Simulation::N_PROFILE];
    Real   pProf[rp_Simulation::N_PROFILE];
    Real   vProf[rp_Simulation::N_PROFILE];
    for (int i=0; i<rp_Simulation::N_PROFILE; ++i) {
        rProf[i]   = i * drProf;
        rhoProf[i] = rp_Simulation::RHO_AMBIENT;
        pProf[i]   = rp_Simulation::P_AMBIENT;
        vProf[i]   = 0.0_wp;
        if (rProf[i] <= rp_Simulation::R_INIT) {
            pProf[i] = rp_Simulation::P_EXP;
        }
    }

    Grid&   grid = Grid::instance();

    Real      dxx = deltas.I();
    Real      dyy = deltas.J() * K2D;
    Real      dzz = deltas.K() * K3D;

    Real      dvSub_buffer[rp_Simulation::N_SUB_ZONES * rp_Simulation::N_SUB_ZONES];
    FArray2D  dvSub{dvSub_buffer,
                    IntVect{LIST_NDIM(0, 0, 0)},
                    IntVect{LIST_NDIM(rp_Simulation::N_SUB_ZONES-1,
                                      rp_Simulation::N_SUB_ZONES-1,
                                      rp_Simulation::N_SUB_ZONES-1)}};

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

                grid.subcellGeometry(     rp_Simulation::N_SUB_ZONES,
                                     1 + (rp_Simulation::N_SUB_ZONES - 1)*K2D,
                                     1 + (rp_Simulation::N_SUB_ZONES - 1)*K3D,
                                     dvc, dvSub_buffer,
                                     xCoords(i) - 0.5_wp*dxx,
                                     xCoords(i) + 0.5_wp*dxx);

                // Break the cell into rp_Simulation::N_SUB_ZONES^NDIM sub-zones, and look up the
                // appropriate quantities along the 1d profile for that subzone.  
                // 
                // Have the final values for the zone be equal to the average of
                // the subzone values.
                sumRho = 0.0;
                sumP   = 0.0;
                sumVX  = 0.0;
                sumVY  = 0.0;
                sumVZ  = 0.0;
                for (unsigned int kk=0; kk<=(rp_Simulation::N_SUB_ZONES - 1)*K3D; ++kk) {
                    zz    = zCoords(k) + ((kk + 0.5_wp) * rp_Simulation::IN_SUBZONES - 0.5_wp) * dzz;
                    zDist = (zz - rp_Simulation::Z_CENTER) * K3D;

                    for (unsigned int jj=0; jj<=(rp_Simulation::N_SUB_ZONES - 1)*K2D; ++jj) {
                        yy    = yCoords(j) + ((jj + 0.5_wp) * rp_Simulation::IN_SUBZONES - 0.5_wp) * dyy;
                        yDist = (yy - rp_Simulation::Y_CENTER) * K2D;

                        for (unsigned int ii=0; ii<rp_Simulation::N_SUB_ZONES; ++ii) {
                            xx    = xCoords(i) + ((ii + 0.5_wp) * rp_Simulation::IN_SUBZONES - 0.5_wp) * dxx;
                            xDist = xx - rp_Simulation::X_CENTER;

                            dist    = sqrt(xDist*xDist + yDist*yDist + zDist*zDist);
                            distInv = 1.0_wp / std::max(dist, MIN_DIST);
 
                            // jLo is index into rProf of largest r value in
                            // profile less than or equal to dist so that
                            //        rProf[jLo] <= dist < rProf[jHi].
                            //
                            // rProf starts at zero and ends at diagonal.
                            // It is corresponds to N_PROFILE equally-spaced
                            // mesh points.  Note that
                            //       dist < diagonal
                            // since xx,yy,zz are coordinates to the center of
                            // subzones and diagonal is based on the extreme
                            // corners of the domain.  Therefore,
                            //       0   <= jLo <  N_PROFILE - 1
                            // and
                            //       jLo <  jHi <= N_PROFILE - 1.
                            jLo = floor( dist / drProf );
                            jHi = jLo + 1;
                            frac = (dist - rProf[jLo]) / drProf;
                            if (jHi >= rp_Simulation::N_PROFILE) {
                                throw std::runtime_error("jHi search failed!");
                            } else if ((rProf[jLo] > dist) || (rProf[jHi] <= dist)) {
                                throw std::runtime_error("What the blurg?!");
                            }

                            // a point at `dist' is frac-way between jLo and jHi.   We do a
                            // linear interpolation of the quantities at jLo and jHi and sum those.
                            pSub   =   pProf[jLo] + frac * (  pProf[jHi] -   pProf[jLo]);
                            rhoSub = rhoProf[jLo] + frac * (rhoProf[jHi] - rhoProf[jLo]);
                            vSub   =   vProf[jLo] + frac * (  vProf[jHi] -   vProf[jLo]);
                            rhoSub = std::max(rhoSub, rp_Simulation::MIN_RHO_INIT);

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
                rho = std::max(sumRho * quotinv, rp_Simulation::SMALL_RHO);
                p   = std::max(sumP   * quotinv, rp_Simulation::SMALL_P);
                vx  = sumVX * quotinv;
                vy  = sumVY * quotinv;
                vz  = sumVZ * quotinv;
                ek  = 0.5_wp * (vx*vx + vy*vy + vz*vz);

                // Assume a gamma-law equation of state
                e    = p / (rp_Eos::GAMMA - 1.0_wp);
                eint = e / rho;
                e    = e / rho + ek;
                e    = std::max(e, rp_Simulation::SMALL_E);

                solnData(i, j, k, DENS_VAR) = rho;
                solnData(i, j, k, PRES_VAR) = p;
                solnData(i, j, k, ENER_VAR) = e;
#ifdef EINT_VAR
                solnData(i, j, k, EINT_VAR) = eint;
#endif
                solnData(i, j, k, GAME_VAR) = rp_Eos::GAMMA;
                solnData(i, j, k, GAMC_VAR) = rp_Eos::GAMMA;
                solnData(i, j, k, VELX_VAR) = vx;
                solnData(i, j, k, VELY_VAR) = vy;
                solnData(i, j, k, VELZ_VAR) = vz;
                solnData(i, j, k, TEMP_VAR) = rp_Simulation::SMALL_T;
#ifdef BDRY_VAR
                solnData(i, j, k, BDRY_VAR) = -1.0_wp;
#endif 
            }
        }
    }
}

