//  Initializes fluid data (density, pressure, velocity, etc.) for
//  a specified block.  This version sets up the Sedov spherical
//  explosion problem.
//
//  References:  Sedov, L. I., 1959, Similarity and Dimensional Methods
//                 in Mechanics (New York:  Academic)
//               Landau, L. D., & Lifshitz, E. M., 1987, Fluid Mechanics,
//                 2d ed. (Oxford:  Pergamon)

#include "Simulation.h"

#include <cmath>
#include <algorithm>
#include <iostream>

#include <Milhoja.h>
#include <Milhoja_real.h>
#include <Milhoja_Grid.h>
#include <Milhoja_FArray2D.h>

#include "Sedov.h"
#include "RuntimeParameters.h"
#include "Eos.h"

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
    using namespace milhoja;

    constexpr Real             PI           = 3.1415926535897932384_wp;
    constexpr Real             MIN_DIST     = 1.0e-10_wp;
    constexpr unsigned int     N_PROFILE    = 10000;
    constexpr Real             P_AMBIENT    = 1.0e-5_wp;
    constexpr Real             RHO_AMBIENT  = 1.0_wp;
    constexpr Real             MIN_RHO_INIT = 1.0e-20_wp;
    constexpr Real             SMALL_RHO    = 1.0e-10_wp;
    constexpr Real             SMALL_P      = 1.0e-10_wp;
    constexpr Real             SMALL_E      = 1.0e-10_wp;
    constexpr Real             SMALL_T      = 1.0e-10_wp;
    constexpr Real             EXP_ENERGY   = 1.0_wp;
    constexpr unsigned int     N_SUB_ZONES  = 7;
    constexpr Real             IN_SUBZONES  = 1.0 / Real(N_SUB_ZONES);
#if   MILHOJA_NDIM == 1
#error "Missing 1D R_INIT and vctr constexpr"
#elif MILHOJA_NDIM == 2
    constexpr Real             R_INIT       = 0.013671875_wp;
    constexpr Real             vctr         = PI * R_INIT*R_INIT;
#elif MILHOJA_NDIM == 3
    constexpr Real             R_INIT       = 0.109375_wp;
    constexpr Real             vctr         = 4.0_wp / 3.0_wp * PI * R_INIT*R_INIT*R_INIT;
#endif
    constexpr Real             P_EXP        = (Eos::GAMMA - 1.0_wp) * EXP_ENERGY / vctr;

    RuntimeParameters&   RPs = RuntimeParameters::instance();

    Real    xMin{RPs.getReal("Grid", "xMin")};
    Real    xMax{RPs.getReal("Grid", "xMax")};
    Real    yMin{RPs.getReal("Grid", "yMin")};
    Real    yMax{RPs.getReal("Grid", "yMax")};
    Real    zMin{RPs.getReal("Grid", "zMin")};
    Real    zMax{RPs.getReal("Grid", "zMax")};

    Real    xCenter = 0.5_wp*(xMax - xMin);
    Real    yCenter = 0.5_wp*(yMax - yMin);
    Real    zCenter = 0.5_wp*(zMax - zMin);

    //Construct the radial samples needed for the initialization.
    Real    diagonal =                (xMax - xMin)
                                    * (xMax - xMin);
    diagonal        += (MILHOJA_K2D * (yMax - yMin)
                                    * (yMax - yMin));
    diagonal        += (MILHOJA_K3D * (zMax - zMin)
                                    * (zMax - zMin));
    diagonal = sqrt(diagonal);

    Real    drProf = diagonal / (Real(N_PROFILE - 1));

    //  just use a top-hat.
    Real   rProf[N_PROFILE];
    Real   rhoProf[N_PROFILE];
    Real   pProf[N_PROFILE];
    Real   vProf[N_PROFILE];
    for (int i=0; i<N_PROFILE; ++i) {
        rProf[i]   = i * drProf;
        rhoProf[i] = RHO_AMBIENT;
        pProf[i]   = P_AMBIENT;
        vProf[i]   = 0.0_wp;
        if (rProf[i] <= R_INIT) {
            pProf[i] = P_EXP;
        }
    }

    Grid&   grid = Grid::instance();

    Real      dxx = deltas.I();
    Real      dyy = deltas.J() * MILHOJA_K2D;
    Real      dzz = deltas.K() * MILHOJA_K3D;

    Real      dvSub_buffer[N_SUB_ZONES * N_SUB_ZONES];
    FArray2D  dvSub{dvSub_buffer,
                    IntVect{LIST_NDIM(0, 0, 0)},
                    IntVect{LIST_NDIM(N_SUB_ZONES-1,
                                      N_SUB_ZONES-1,
                                      N_SUB_ZONES-1)}};

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

    for         (int k=lo.K(); k<=hi.K(); ++k) {
        for     (int j=lo.J(); j<=hi.J(); ++j) {
            for (int i=lo.I(); i<=hi.I(); ++i) {
                // TODO: Get the volumes in a single array outside loop nest
                dvc = grid.getCellVolume(level, IntVect{LIST_NDIM(i, j, k)});

                grid.subcellGeometry(     N_SUB_ZONES,
                                     1 + (N_SUB_ZONES - 1)*MILHOJA_K2D,
                                     1 + (N_SUB_ZONES - 1)*MILHOJA_K3D,
                                     dvc, dvSub_buffer,
                                     xCoords(i) - 0.5_wp*dxx,
                                     xCoords(i) + 0.5_wp*dxx);

                // Break the cell into N_SUB_ZONES^MILHOJA_NDIM sub-zones, and look up the
                // appropriate quantities along the 1d profile for that subzone.  
                // 
                // Have the final values for the zone be equal to the average of
                // the subzone values.
                sumRho = 0.0;
                sumP   = 0.0;
                sumVX  = 0.0;
                sumVY  = 0.0;
                sumVZ  = 0.0;
                for (unsigned int kk=0; kk<=(N_SUB_ZONES - 1)*MILHOJA_K3D; ++kk) {
                    zz    = zCoords(k) + ((kk + 0.5_wp) * IN_SUBZONES - 0.5_wp) * dzz;
                    zDist = (zz - zCenter) * MILHOJA_K3D;

                    for (unsigned int jj=0; jj<=(N_SUB_ZONES - 1)*MILHOJA_K2D; ++jj) {
                        yy    = yCoords(j) + ((jj + 0.5_wp) * IN_SUBZONES - 0.5_wp) * dyy;
                        yDist = (yy - yCenter) * MILHOJA_K2D;

                        for (unsigned int ii=0; ii<N_SUB_ZONES; ++ii) {
                            xx    = xCoords(i) + ((ii + 0.5_wp) * IN_SUBZONES - 0.5_wp) * dxx;
                            xDist = xx - xCenter;

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
                            if (jHi >= N_PROFILE) {
                                throw std::runtime_error("jHi search failed!");
                            } else if ((rProf[jLo] > dist) || (rProf[jHi] <= dist)) {
                                throw std::runtime_error("What the what?!");
                            }

                            // a point at `dist' is frac-way between jLo and jHi.   We do a
                            // linear interpolation of the quantities at jLo and jHi and sum those.
                            pSub   =   pProf[jLo] + frac * (  pProf[jHi] -   pProf[jLo]);
                            rhoSub = rhoProf[jLo] + frac * (rhoProf[jHi] - rhoProf[jLo]);
                            vSub   =   vProf[jLo] + frac * (  vProf[jHi] -   vProf[jLo]);
                            rhoSub = std::max(rhoSub, MIN_RHO_INIT);

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
                rho = std::max(sumRho * quotinv, SMALL_RHO);
                p   = std::max(sumP   * quotinv, SMALL_P);
                vx  = sumVX * quotinv;
                vy  = sumVY * quotinv;
                vz  = sumVZ * quotinv;
                ek  = 0.5_wp * (vx*vx + vy*vy + vz*vz);

                // Assume a gamma-law equation of state
                e    = p / (Eos::GAMMA - 1.0_wp);
                eint = e / rho;
                e    = e / rho + ek;
                e    = std::max(e, SMALL_E);

                solnData(i, j, k, DENS_VAR) = rho;
                solnData(i, j, k, PRES_VAR) = p;
                solnData(i, j, k, ENER_VAR) = e;
                solnData(i, j, k, EINT_VAR) = eint;
                solnData(i, j, k, GAME_VAR) = Eos::GAMMA;
                solnData(i, j, k, GAMC_VAR) = Eos::GAMMA;
                solnData(i, j, k, VELX_VAR) = vx;
                solnData(i, j, k, VELY_VAR) = vy;
                solnData(i, j, k, VELZ_VAR) = vz;
                solnData(i, j, k, TEMP_VAR) = SMALL_T;
            }
        }
    }
}

