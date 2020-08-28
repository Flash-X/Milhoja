#include "Driver.h"

#include "Grid.h"
#include "GridAmrex.h"
#include "OrchestrationLogger.h"

#include <AMReX_MultiFabUtil.H>
#include <AMReX_Box.H>
#include <AMReX_FArrayBox.H>

#include <AmrCoreAdv_F.H>

using namespace orchestration;

Real dt[3], t_new[3];
int step[3];
int regrid_int = 2;
Real stop_time = 2.0_wp;
int max_steps = 125;


void doTimestep(const int lev, Real time) {
    Grid& grid = Grid::instance();
    Logger::instance().log("[Driver] Level " + std::to_string(lev)
                           + ", step " + std::to_string(step[lev]+1)
                           + "; Advancing with dt = "+std::to_string(dt[lev]));

    // do advance
    {
        // update time
        Real t_old = t_new[lev];
        t_new[lev] = t_new[lev] + dt[lev];
        const Real ctr_time = 0.5*(t_new[lev]+t_old) ;

        // get some pointers
        const RealVect dxVect = grid.getDeltas(lev);
        const Real* dx = dxVect.dataPtr();
        const RealVect probLoVect = grid.getProbLo();
        const Real* prob_lo = probLoVect.dataPtr();

        amrex::FArrayBox flux[NDIM], uface[NDIM];
        // no tiling for now
        for(auto ti=grid.buildTileIter(lev); ti->isValid(); ti->next()) {
            auto tileDesc = ti->buildCurrentTile();

            const amrex::Box& bx{amrex::IntVect(tileDesc->lo()),
                                 amrex::IntVect(tileDesc->hi())}; //tilebox

            // statein (t=t_0), stateout (t=t_0)
            Real* stateinPtr  = tileDesc->dataPtr();
            Real* stateoutPtr = tileDesc->dataPtr();
            const IntVect   loGC = tileDesc->loGC();
            const IntVect   hiGC = tileDesc->hiGC();

            for(int i=0; i<NDIM; ++i) {
                const amrex::Box& bxtmp = amrex::surroundingNodes(bx,i);
                flux[i].resize(bxtmp, NUNKVAR);
                uface[i].resize(amrex::grow(bxtmp,1),1);
            }

            get_face_velocity(&lev, &ctr_time,
                              AMREX_D_DECL(BL_TO_FORTRAN(uface[0]),
                                           BL_TO_FORTRAN(uface[1]),
                                           BL_TO_FORTRAN(uface[2])),
                              dx, prob_lo);

            advect(&time, bx.loVect(), bx.hiVect(),
                   stateinPtr,
                   loGC.dataPtr(),
                   hiGC.dataPtr(),
                   stateoutPtr,
                   loGC.dataPtr(),
                   hiGC.dataPtr(),
                   LIST_NDIM(BL_TO_FORTRAN_3D(uface[0]),
                             BL_TO_FORTRAN_3D(uface[1]),
                             BL_TO_FORTRAN_3D(uface[2])),
                   LIST_NDIM(BL_TO_FORTRAN_3D(flux[0]),
                             BL_TO_FORTRAN_3D(flux[1]),
                             BL_TO_FORTRAN_3D(flux[2])),
                   dx, &dt[lev] );
        } // TileIter

    }// advance
}

void ComputeDt(const Real time) {
    Grid& grid = Grid::instance();
    std::vector<Real> dt_tmp(grid.getMaxLevel()+1);

    //estimate time step for each level
    for(int lev=0; lev<=grid.getMaxLevel(); ++lev) {
        Real dt_est = std::numeric_limits<Real>::max();

        const RealVect dxVect = grid.getDeltas(lev);
        const Real* dx = dxVect.dataPtr();
        const RealVect probLoVect = grid.getProbLo();
        const Real* prob_lo = probLoVect.dataPtr();

        amrex::FArrayBox uface[NDIM];
        for(auto ti=grid.buildTileIter(lev); ti->isValid(); ti->next()) {
            auto tileDesc = ti->buildCurrentTile();
            const amrex::Box& bx{amrex::IntVect(tileDesc->lo()),
                                 amrex::IntVect(tileDesc->hi())}; //tilebox

            for(int i=0; i<NDIM; ++i) {
                const amrex::Box& bxtmp = amrex::surroundingNodes(bx,i);
                uface[i].resize(amrex::grow(bxtmp,i,1),1);
            }
            get_face_velocity(&lev, &time,
                              AMREX_D_DECL(BL_TO_FORTRAN(uface[0]),
                                           BL_TO_FORTRAN(uface[1]),
                                           BL_TO_FORTRAN(uface[2])),
                              dx, prob_lo);
            for(int i=0; i<NDIM; ++i) {
                Real umax = uface[i].norm<amrex::RunOn::Host>(0);
                if(umax> 1.e-100) {
                    dt_est = std::min(dt_est, dx[i]/umax);
                }
            }

        }
        dt_est = dt_est*0.7_wp; // cfl = 0.7
        dt_tmp[lev] = dt_est;

    }
    amrex::ParallelDescriptor::ReduceRealMin(&dt_tmp[0], dt_tmp.size());

    // get minimum time step, and enforce max change ratio of 1.1
    Real dt_0 = dt_tmp[0];
    const Real change_max = 1.1;
    for (int lev=0; lev<=grid.getMaxLevel(); ++lev) {
        dt_tmp[lev] = std::min(dt_tmp[lev], change_max*dt[lev]);
        dt_0 = std::min(dt_0, dt_tmp[lev]);
    }

    // limit time step by stop time
    const Real eps = 1.e-3*dt_0;
    if (t_new[0]+dt_0 > stop_time-eps) {
        dt_0 = stop_time - t_new[0];
    }

    // set dt for all levels
    for(int lev=0; lev<=grid.getMaxLevel(); ++lev) {
        dt[lev] = dt_0;
    }
}

namespace Driver {

void EvolveAdvection() {
    Grid& grid = Grid::instance();

    // initialize driver variables
    Real time = 0.0_wp;
    for(int i=0;i<=grid.getMaxRefinement();++i) {
        t_new[i] = 0.0_wp;
        dt[i] = 1e100;
        step[i] = 0;
    }

    for(int nstep=0; nstep<max_steps; ++nstep) {

        Logger::instance().log("[Driver] Starting Step "
                               + std::to_string(nstep+1)
                               + "...");

        // Update dt array
        ComputeDt(time);

        // Regrid
        if(nstep>0 && (nstep%regrid_int==0) ) {
            grid.regrid();
        }

        // All level GC fill
        for(int lev=0; lev<=grid.getMaxLevel(); ++lev) {
            grid.fillGC(lev);
        }

        // advance all levels
        for(int lev=0; lev<=grid.getMaxLevel(); ++lev) {
            doTimestep(lev, time);
            ++step[lev];
        }

        dynamic_cast<GridAmrex&>(grid).averageDownAll();

        time = time + dt[0];

        Logger::instance().log("[Driver] Done with Step "
                               + std::to_string(nstep+1)
                               + ", with dt = " + std::to_string(dt[0]) + ".");

        if (time >= 2.0 - 1.e-6*dt[0]) break;
    }

    std::string pltName = amrex::Concatenate("adv_plt_", step[0], 4);
    grid.writePlotfile(pltName);

}

} //Driver
