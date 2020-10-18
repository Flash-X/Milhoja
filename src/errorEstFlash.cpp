#include "errorEstFlash.h"

#include "Grid.h"
#include "Grid_RealVect.h"
#include "Grid_Axis.h"
#include "Flash.h"
#include "constants.h"

#include <cmath>
#include <limits>

using namespace orchestration;

/**
  * \todo use I,J,K methods for RealVect
  */
Real Simulation::errorEstFlash(std::shared_ptr<Tile> tileDesc, const int iref, const Real ref_filter) {

    Real error = 0.0_wp;

    const int      lev    = tileDesc->level();
    const IntVect  lo     = tileDesc->lo();
    const IntVect  hi     = tileDesc->hi();
    const IntVect  loGC   = tileDesc->loGC();
    const IntVect  hiGC   = tileDesc->hiGC();
    const RealVect deltas = tileDesc->deltas();
    auto           f      = tileDesc->data();

    RealVect del = RealVect{LIST_NDIM(0.5_wp,0.5_wp,0.5_wp)} / deltas;
    RealVect del_f;
    for(int n=Axis::I; n<NDIM; ++n) del_f[n] = del[n];

    // Allocate for first derivatives
    auto delu  = FArray4D::buildScratchArray4D(loGC,hiGC,NDIM);
    auto delua = FArray4D::buildScratchArray4D(loGC,hiGC,NDIM);

    // some non-cartesian stuff

    // Compute first derivatives
    constexpr IntVect knd{ LIST_NDIM(1,1,1) };
    for(         int k= loGC.K()+knd.K(); k<= hiGC.K()-knd.K(); ++k ) {
        for(     int j= loGC.J()+knd.J(); j<= hiGC.J()-knd.J(); ++j ) {
            for( int i= loGC.I()+knd.I(); i<= hiGC.I()-knd.I(); ++i ) {
                // TODO spherical: adjust del

                // d/dx
                delu(i,j,k,Axis::I)  = f(i+1,j,k,iref) - f(i-1,j,k,iref);
                delu(i,j,k,Axis::I)  = delu(i,j,k,Axis::I)*del[0];
                delua(i,j,k,Axis::I) = std::abs(f(i+1,j,k,iref)) +
                                       std::abs(f(i-1,j,k,iref));
                delua(i,j,k,Axis::I) = delua(i,j,k,Axis::I)*del[0];

#if NDIM>=2
                // d/dy
                // TODO non-cartesian: adjust del_f
                delu(i,j,k,Axis::J)  = f(i,j+1,k,iref) - f(i,j-1,k,iref);
                delu(i,j,k,Axis::J)  = delu(i,j,k,Axis::J)*del_f[1];
                delua(i,j,k,Axis::J) = std::abs(f(i,j+1,k,iref)) +
                                       std::abs(f(i,j-1,k,iref));
                delua(i,j,k,Axis::J) = delua(i,j,k,Axis::J)*del_f[1];
#endif
#if NDIM==3
                // d/dz
                // TODO non-cartesian: adjust del_f
                delu(i,j,k,Axis::K)  = f(i,j,k+1,iref) - f(i,j,k-1,iref);
                delu(i,j,k,Axis::K)  = delu(i,j,k,Axis::K)*del_f[2];
                delua(i,j,k,Axis::K) = std::abs(f(i,j,k+1,iref)) +
                                       std::abs(f(i,j,k-1,iref));
                delua(i,j,k,Axis::K) = delua(i,j,k,Axis::K)*del_f[2];
#endif
            }
        }
    }

    // Allocate for second derivatives
    constexpr int sqndim = NDIM*NDIM;
    Real delu2[sqndim], delu3[sqndim], delu4[sqndim];

    // Compute second derivatives
    int grd = NGUARD - std::min(2,NGUARD-2); // two guardcells
    IntVect bstart = loGC + grd;
    IntVect bend   = hiGC - grd;
    // TODO adjust bstart and bend for non-periodic BCs

    for(         int k= bstart.K(); k<= bend.K(); ++k ) {
        for(     int j= bstart.J(); j<= bend.J(); ++j ) {
            for( int i= bstart.I(); i<= bend.I(); ++i ) {
                // TODO spherical: adjust del

                // d/dxdx
                delu2[0] = delu(i+1,j,k,Axis::I) - delu(i-1,j,k,Axis::I);
                delu2[0] = delu2[0]*del[0];

                delu3[0] = std::abs(delu(i+1,j,k,Axis::I)) +
                           std::abs(delu(i-1,j,k,Axis::I));
                delu3[0] = delu3[0]*del[0];

                delu4[0] = delua(i+1,j,k,Axis::I) + delua(i-1,j,k,Axis::I);
                delu4[0] = delu4[0]*del[0];

#if NDIM>=2
                // TODO non-cartesian: adjust del_f

                // d/dydx
                delu2[1] = delu(i,j+1,k,Axis::I) - delu(i,j-1,k,Axis::I);
                delu2[1] = delu2[1]*del_f[1];

                delu3[1] = std::abs(delu(i,j+1,k,Axis::I)) +
                           std::abs(delu(i,j-1,k,Axis::I));
                delu3[1] = delu3[1]*del_f[1];

                delu4[1] = delua(i,j+1,k,Axis::I) + delua(i,j-1,k,Axis::I);
                delu4[1] = delu4[1]*del_f[1];

                // d/dxdy
                delu2[2] = delu(i+1,j,k,Axis::J) - delu(i-1,j,k,Axis::J);
                delu2[2] = delu2[2]*del[0];

                delu3[2] = std::abs(delu(i+1,j,k,Axis::J)) +
                           std::abs(delu(i-1,j,k,Axis::J));
                delu3[2] = delu3[2]*del[0];

                delu4[2] = delua(i+1,j,k,Axis::J) + delua(i-1,j,k,Axis::J);
                delu4[2] = delu4[2]*del[0];

                // d/dydy
                delu2[3] = delu(i,j+1,k,Axis::J) - delu(i,j-1,k,Axis::J);
                delu2[3] = delu2[3]*del_f[1];

                delu3[3] = std::abs(delu(i,j+1,k,Axis::J)) +
                           std::abs(delu(i,j-1,k,Axis::J));
                delu3[3] = delu3[3]*del_f[1];

                delu4[3] = delua(i,j+1,k,Axis::J) + delua(i,j-1,k,Axis::J);
                delu4[3] = delu4[3]*del_f[1];
#endif

#if NDIM==3
                //TODO non-cartesian: adjust del_f

                // d/dzdx
                delu2[4] = delu(i,j,k+1,Axis::I) - delu(i,j,k-1,Axis::I);
                delu2[4] = delu2[4]*del_f[2];

                delu3[4] = std::abs(delu(i,j,k+1,Axis::I)) +
                           std::abs(delu(i,j,k-1,Axis::I));
                delu3[4] = delu3[4]*del_f[2];

                delu4[4] = delua(i,j,k+1,Axis::I) + delua(i,j,k-1,Axis::I);
                delu4[4] = delu4[4]*del_f[2];

                // d/dzdy
                delu2[5] = delu(i,j,k+1,Axis::J) - delu(i,j,k-1,Axis::J);
                delu2[5] = delu2[5]*del_f[2];

                delu3[5] = std::abs(delu(i,j,k+1,Axis::J)) +
                           std::abs(delu(i,j,k-1,Axis::J));
                delu3[5] = delu3[5]*del_f[2];

                delu4[5] = delua(i,j,k+1,Axis::J) + delua(i,j,k-1,Axis::J);
                delu4[5] = delu4[5]*del_f[2];

                // d/dxdz
                delu2[6] = delu(i+1,j,k,Axis::K) - delu(i-1,j,k,Axis::K);
                delu2[6] = delu2[6]*del[0];

                delu3[6] = std::abs(delu(i+1,j,k,Axis::K)) +
                           std::abs(delu(i-1,j,k,Axis::K));
                delu3[6] = delu3[6]*del[0];

                delu4[6] = delua(i+1,j,k,Axis::K) + delua(i-1,j,k,Axis::K);
                delu4[6] = delu4[6]*del[0];

                // d/dydz
                delu2[7] = delu(i,j+1,k,Axis::K) - delu(i,j-1,k,Axis::K);
                delu2[7] = delu2[7]*del_f[1];

                delu3[7] = std::abs(delu(i,j+1,k,Axis::K)) +
                           std::abs(delu(i,j-1,k,Axis::K));
                delu3[7] = delu3[7]*del_f[1];

                delu4[7] = delua(i,j+1,k,Axis::K) + delua(i,j-1,k,Axis::K);
                delu4[7] = delu4[7]*del_f[1];

                // d/dzdz
                delu2[8] = delu(i,j,k+1,Axis::K) - delu(i,j,k-1,Axis::K);
                delu2[8] = delu2[8]*del_f[2];

                delu3[8] = std::abs(delu(i,j,k+1,Axis::K)) +
                           std::abs(delu(i,j,k-1,Axis::K));
                delu3[8] = delu3[8]*del_f[2];

                delu4[8] = delua(i,j,k+1,Axis::K) + delua(i,j,k-1,Axis::K);
                delu4[8] = delu4[8]*del_f[2];
#endif

                // Compute the error
                Real num{0.0_wp}, denom{0.0_wp};
                for( int kk=0; kk<sqndim; ++kk) {
                    num = num + delu2[kk]*delu2[kk];
                    denom = denom + (delu3[kk] + (ref_filter*delu4[kk]))*
                                    (delu3[kk] + (ref_filter*delu4[kk]));
                }

                if (denom==0.0_wp && num!=0.0_wp) {
                    error = std::numeric_limits<Real>::max();
                }
                else if (denom!=0.0_wp) {
                    error = std::max(error, num/denom);
                }
            }
        }
    }

    // return max of the square root of the errors
    error = std::sqrt(error);

    return error;

}

