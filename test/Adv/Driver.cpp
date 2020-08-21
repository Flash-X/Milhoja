#include "Driver.h"

#include "Grid.h"
#include "GridAmrex.h"
#include "OrchestrationLogger.h"

#include <AMReX_MultiFabUtil.H>
#include <AMReX_FillPatchUtil.H>
#include <AMReX_Box.H>
#include <AMReX_FArrayBox.H>

using namespace orchestration;

Real dt[3], t_old[3], t_new[3];
int step[3];
int nsubsteps[3] = {2,2,2};
int regrid_int = 2;
std::vector<int> last_regrid_step(max_level,0);

amrex::MultiFab getSBorder(Real time, int lev, int nComp, int nGrow) {
    GridAmrex& grid = dynamic_cast<GridAmrex&>( Grid::instance() );
    amrex::MultiFab Sborder(grid.getBoxArray(lev),
                            grid.getDMap(lev),
                            nComp, nGrow);
    // fill patch
    // FillPatch(lev, time, Sborder, 0, Sborder.nComp());
    if (lev==0) {
        amrex::Vector<MultiFab*> smf;
        amrex::Vector<amrex::Real> stime;
        //GetData(0,time,smf,stime);
        //BndryFuncArray bfunc(phifill);
        //PhysBCFunct<BndryFuncArray> physbc(grid.getGeom(lev), bcs, bfunc);
        amrex::FillPatchSingleLevel(mf, time, smf,
                                    stime, 0, icomp, nComp,
                                    grid.getGeom(lev), physbc, 0);
    }
    else {
        //Vector<MultiFab*> cmf, fmf;
        //Vector<Real> ctime, ftime;
        //GetData(lev-1, time, cmf, ctime);
        //GetData(lev  , time, fmf, ftime);

        //BndryFuncArray bfunc(phifill);
        //PhysBCFunct<BndryFuncArray> cphysbc(geom[lev-1],bcs,bfunc);
        //PhysBCFunct<BndryFuncArray> fphysbc(geom[lev  ],bcs,bfunc);

        //Interpolater* mapper = &cell_cons_interp;

        //amrex::FillPatchTwoLevels(mf, time, cmf, ctime, fmf, ftime,
        //                          0, icomp, ncomp, geom[lev-1], geom[lev],
        //                          cphysbc, 0, fphysbc, 0,
        //                          refRatio(lev-1), mapper, bcs, 0);
    }
    return Sborder;
}

void doTimestep(unsigned int lev, Real time, int iteration) {
    Grid& grid = Grid::instance();
    int max_level = grid.getMaxRefinement();
    static std::vector<int> last_regrid_step(max_level, 0);

    // possible regrid
    if(lev < max_level && step[lev] > last_regrid_step[lev] ) {
        //do regrid
        throw std::logic_error("Implement regridding");
    }

    Logger::instance().log("[Driver] Level " + std::to_string(lev)
                           + ", step " + std::to_string(step[lev]+1)
                           + "; Advancing with dt = " + std::to_string(dt[lev]));

    // do advance
    {
        // update time
        t_old[lev] = t_new[lev];
        t_new[lev] = t_old[lev] + dt[lev];

        // get some pointers
        const Real old_time = t_old[lev];
        const Real new_time = t_new[lev];
        const Real ctr_time = 0.5*(old_time+new_time);
        const RealVect dxVect = grid.getDeltas();
        const Real* dx = dxVect.dataPtr();
        const RealVect probLoVect = grid.getProbLo();
        const Real* prob_lo = probLoVect.dataPtr();

        // make Sborder (backup/sratch)
        const int nComp = NUNKVAR;
        const int nGrow = 3;
        amrex::MultiFab Sborder = getSborder(time, lev, nComp, nGrow);

        FArrayBox flux[NDIM], uface[NDIM];
        for(auto ti=grid.buildTileIter(lev); ti.isValid(); ti->next()) {
            auto tileDesc = ti->buildCurrentTile();

            const amrex::Box& bx{amrex::IntVect(tileDesc->lo()),
                                 amrex::IntVect(tileDesc->hi())}; //tilebox

            // statein (t=t_0), stateout (t=t_0)
            const amrex::FArrayBox& statein = Sborder[tileDesc->gridIndex()];
            Real* stateoutPtr = tileDesc->dataPtr();
            const IntVect   loGC = tileDesc->loGC();
            const IntVect   hiGC = tileDesc->hiGC();

            for(int i=0; i<NDIM; ++i) {
                const amrex::Box& bxtmp = amrex::surroundingNodes(bx,i);
                flux[i].resize(bxtmp, nComp);
                uface[i].resize(amrex::grow(bxtmp,1),1);
            }

            get_face_velocity(&lev, &ctr_time,
                              AMREX_D_DECL(BL_TO_FORTRAN(uface[0]),
                                           BL_TO_FORTRAN(uface[1]),
                                           BL_TO_FORTRAN(uface[2])),
                              dx, prob_lo);

            advect(&time, bx.loVect(), bx.hiVect(),
                   BL_TO_FORTRAN_3D(statein),
                   stateoutPtr,
                   loGC.dataPtr(),
                   hiGC.dataPtr(),
                   AMREX_D_DECL(BL_TO_FORTRAN(uface[0]),
                                BL_TO_FORTRAN(uface[1]),
                                BL_TO_FORTRAN(uface[2])),
                   AMREX_D_DECL(BL_TO_FORTRAN(flux[0]),
                                BL_TO_FORTRAN(flux[1]),
                                BL_TO_FORTRAN(flux[2])),
                   dx, &dt[lev] );

            // now statein (t=t_0), stateout (t=t_0+dt)
        }
    }

    ++step[lev];

    if(lev< grid.getMaxLevel()) {
        // recursive call to do next-finer level
        for(int i=0;i<nsubsteps[lev+1];++i) {
            doTimestep(lev+1,time+i*dt[lev+1], i+1);
        }

        // average down to??
    }
}

void ComputeDt() {
    dt[0] = 0.01097491125;
    //dt[0] = 0.01093983378;
    //dt[1] = 0.005469916891;
    //dt[2] = 0.002734958446;
}

namespace Driver {

void EvolveAdvection() {
    Real time = 0.0_wp;
    for(int i=0;i<NDIM;++i) {
        t_old[i] = 0.0_wp;
        t_new[i] = 0.0_wp;
    }

    // TODO loop over steps
    for(int i=0;i<NDIM;++i) step[i] = 0;

    Logger::instance().log("Starting coarse Step " + std::to_string(step[0])
                           + "...");


    ComputeDt();

    unsigned int lev = 0;
    int iteration = 1;
    doTimestep(lev, time, iteration);

    time = time + dt[0];

    Logger::instance().log("Done with coarse Step " + std::to_string(step[0])
                           + ", with dt = " + std::to_string(dt[0]) + ".");

    //if (time >= 2.0 - 1.e-6*dt[0]) break;
}

} //Driver
