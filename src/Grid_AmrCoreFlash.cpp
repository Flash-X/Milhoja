#include "Grid_AmrCoreFlash.h"

#include "Grid.h"
#include "OrchestrationLogger.h"
#include "RuntimeAction.h"
#include "ThreadTeamDataType.h"
#include "ThreadTeam.h"

#include <AMReX_PlotFileUtil.H>
#include <AMReX_Interpolater.H>
#include <AMReX_FillPatchUtil.H>

#include "Flash.h"

namespace orchestration {

/** \brief Constructor for AmrCoreFlash
  *
  * Creates blank multifabs on each level.
  */
AmrCoreFlash::AmrCoreFlash(ACTION_ROUTINE initBlock,
                           ERROR_ROUTINE errorEst)
    : initBlock_{initBlock},
      errorEst_{errorEst} {

    // Allocate and resize unk_ (vector of Multifabs).
    unk_.resize(max_level+1);

    // Set periodic Boundary conditions
    bcs_.resize(1);
    for(int i=0; i<NDIM; ++i) {
        bcs_[0].setLo(i, amrex::BCType::int_dir);
        bcs_[0].setHi(i, amrex::BCType::int_dir);
    }
}

//! Default constructor
AmrCoreFlash::~AmrCoreFlash() {
}

//! Write all levels of unk_ to plotfile.
void AmrCoreFlash::writeMultiPlotfile(const std::string& filename) const {
#ifdef GRID_LOG
    std::string msg = "[GridAmrex] Writing to plotfile: "+filename+"...";
    Logger::instance().log(msg);
#endif
    amrex::Vector<std::string>    names(unk_[0].nComp());
    if (names.size()==1) {
        names[0] = "phi";
    } else {
        names[0] = "Density";
        names[1] = "Energy";
    }
    amrex::Vector<const amrex::MultiFab*> mfs;
    for(int i=0; i<=finest_level; ++i) {
        mfs.push_back( &unk_[i] );
    }
    amrex::Vector<int> lsteps( max_level+1 , 0);

    amrex::WriteMultiLevelPlotfile(filename,
                                   finest_level+1,
                                   mfs,
                                   names,
                                   Geom(),
                                   0.0,
                                   lsteps,
                                   refRatio());
}

/**
  * \brief Make New Level from Coarse
  *
  * \param lev Level being made
  * \param time Simulation time
  * \param ba BoxArray of level being made
  * \param dm DistributionMapping of leving being made
  */
void AmrCoreFlash::MakeNewLevelFromCoarse (int lev, amrex::Real time,
            const amrex::BoxArray& ba, const amrex::DistributionMapping& dm) {
#ifdef GRID_LOG
    std::string msg = "[AmrCoreFlash::MakeNewLevelFromCoarse] Making level " +
                      std::to_string(lev) + " from coarse...";
    Logger::instance().log(msg);
#endif

    Grid& grid = Grid::instance();

    // Build multifab unk_[lev].
    unk_[lev].define(ba, dm, NUNKVAR, NGUARD);

    fillFromCoarse(unk_[lev], lev);
}

/**
  * \brief Remake Level
  *
  * \param lev Level being made
  * \param time Simulation time
  * \param ba BoxArray of level being made
  * \param dm DistributionMapping of leving being made
  */
void AmrCoreFlash::RemakeLevel (int lev, amrex::Real time,
            const amrex::BoxArray& ba, const amrex::DistributionMapping& dm) {
#ifdef GRID_LOG
    std::string msg = "[AmrCoreFlash::RemakeLevel] Remaking level " +
                      std::to_string(lev) + "...";
    Logger::instance().log(msg);
#endif

    amrex::MultiFab unkTmp{ba, dm, NUNKVAR, NGUARD};
    fillPatch(unkTmp, lev);

    std::swap(unkTmp, unk_[lev] );
#ifdef GRID_LOG
    std::string msg2 = "[AmrCoreFlash::RemakeLevel] Remade level " +
                      std::to_string(lev) + " with " +
                      std::to_string(ba.size()) + " blocks.";
    Logger::instance().log(msg2);
#endif
}

/**
  * \brief Clear level
  *
  * \param lev Level being cleared
  */
void AmrCoreFlash::ClearLevel (int lev) {
#ifdef GRID_LOG
    std::string msg = "[AmrCoreFlash::ClearLevel] Clearing level " +
                      std::to_string(lev) + "...";
    Logger::instance().log(msg);
#endif
    unk_[lev].clear();
}

/**
  * \brief Make new level from scratch
  *
  * \param lev Level being made
  * \param time Simulation time
  * \param ba BoxArray of level being made
  * \param dm DistributionMapping of leving being made
  */
void AmrCoreFlash::MakeNewLevelFromScratch (int lev, amrex::Real time,
            const amrex::BoxArray& ba, const amrex::DistributionMapping& dm) {
#ifdef GRID_LOG
    std::string msg = "[AmrCoreFlash::MakeNewLevelFromScratch] Creating level " +
                      std::to_string(lev) + "...";
    Logger::instance().log(msg);
#endif

    Grid& grid = Grid::instance();

    // Build multifab unk_[lev].
    unk_[lev].define(ba, dm, NUNKVAR, NGUARD);

    // Initialize data in unk_[lev] to 0.0.
    unk_[lev].setVal(0.0_wp);

    // Initalize simulation block data in unk_[lev].
    // Must fill interiors, GC optional.
    // TODO: Thread count should be a runtime variable
    RuntimeAction    action;
    action.name = "initBlock";
    action.nInitialThreads = 4;
    action.teamType = ThreadTeamDataType::BLOCK;
    action.routine = initBlock_;
    ThreadTeam  team(4, 1);
    team.startCycle(action, "Cpu");
    for (auto ti = grid.buildTileIter(lev); ti->isValid(); ti->next()) {
        team.enqueue( ti->buildCurrentTile() );
    }
    team.closeQueue();
    team.wait();

    // DO A GC FILL HERE?

#ifdef GRID_LOG
    std::string msg2 = "[AmrCoreFlash::MakeNewLevelFromScratch] Created level " +
                      std::to_string(lev) + " with " +
                      std::to_string(ba.size()) + " blocks.";
    Logger::instance().log(msg2);
#endif
}

/**
  * \brief Tag boxes for refinement
  *
  * \param lev Level being checked
  * \param tags Tags for Box array
  * \param time Simulation time
  * \param ngrow ngrow
  */
void AmrCoreFlash::ErrorEst (int lev, amrex::TagBoxArray& tags,
                             amrex::Real time, int ngrow) {
#ifdef GRID_LOG
    std::string msg = "[AmrCoreFlash::ErrorEst] Doing ErrorEst for level " +
                      std::to_string(lev) + "...";
    Logger::instance().log(msg);
#endif
    static const int clearval = amrex::TagBox::CLEAR;
    static const int tagval = amrex::TagBox::SET;

    Grid& grid = Grid::instance();

    //TODO use tiling for this loop?
    for (auto ti = grid.buildTileIter(lev); ti->isValid(); ti->next()) {
        std::shared_ptr<Tile> tileDesc = ti->buildCurrentTile();
        const IntVect lo = tileDesc->lo();
        const IntVect hi = tileDesc->hi();

        amrex::TagBox& tagfab = tags[tileDesc->gridIndex()];
        auto tagdata = tagfab.array();
        IntVect tag_lo{ tagfab.smallEnd() };
        IntVect tag_hi{ tagfab.bigEnd() };
#ifndef GRID_ERRCHECK_OFF
        if( !(tag_lo.allLE(lo) && tag_hi.allGE(hi)) ) {
            throw std::logic_error ("Tagbox is smaller than tile box.");
        }
#endif
        for(     int k=lo.K(); k<=hi.K(); ++k) {
          for(   int j=lo.J(); j<=hi.J(); ++j) {
            for( int i=lo.I(); i<=hi.I(); ++i) {
              tagdata(i,j,k,0) = clearval;
        }}}

        // TODO make these RPs
        int numRefineVars = 1;
        std::vector<int> refineVars;
        std::vector<Real> refineCutoff, refineFilter;
        refineVars.push_back(0);
        refineFilter.push_back(1.0_wp);
        refineCutoff.push_back(5.0_wp);
        // Loop over variables
        for (int n=0; n<numRefineVars; ++n) {
            int iref = refineVars[n];
            if(iref<0) continue;

            Real error = errorEst_(tileDesc, iref, refineFilter[n]);

            if (error > refineCutoff[n]) {
#ifndef ADVECTION_TUTORIAL
                IntVect mid = (0.5_wp * RealVect(tag_lo + tag_hi)).round();
                tagdata(mid.I(),mid.J(),mid.K(), 0) = tagval;
#else
                static bool first = true;
                static amrex::Vector<amrex::Real> phierr;
                if(first) {
                    first = false;
                    //get phierr from parm parse or runtime parameters?
                    phierr.push_back(1.01);
                    phierr.push_back(1.1);
                    phierr.push_back(1.5);
                }
                auto f = tileDesc->data();
                for(     int k=lo.K(); k<=hi.K(); ++k) {
                  for(   int j=lo.J(); j<=hi.J(); ++j) {
                    for( int i=lo.I(); i<=hi.I(); ++i) {
                        if(f(i,j,k,0)>phierr[lev] ) {
                            tagdata(i,j,k,0) = tagval;
                        }
                }}}
#endif
            }
        }

    } //tile iterator

#ifdef GRID_LOG
    std::string msg2 = "[AmrCoreFlash::ErrorEst] Did ErrorEst for level " +
                      std::to_string(lev) + ".";
    Logger::instance().log(msg2);
#endif
}

void AmrCoreFlash::fillPatch(amrex::MultiFab& mf, const unsigned int lev) {
    if (lev==0) {
        amrex::Vector<amrex::MultiFab*> smf;
        amrex::Vector<amrex::Real> stime;
        smf.push_back(&unk_[0]);
        stime.push_back(0.0_wp);

        amrex::CpuBndryFuncFab bndry_func(nullptr);
        amrex::PhysBCFunct<amrex::CpuBndryFuncFab>
            physbc(geom[lev],bcs_,bndry_func);

        amrex::FillPatchSingleLevel(mf, 0.0_wp, smf, stime,
                                    0, 0, mf.nComp(),
                                    geom[lev], physbc, 0);
    }
    else {
        amrex::Vector<amrex::MultiFab*> cmf, fmf;
        amrex::Vector<amrex::Real> ctime, ftime;
        cmf.push_back(&unk_[lev-1]);
        ctime.push_back(0.0_wp);
        fmf.push_back(&unk_[lev]);
        ftime.push_back(0.0_wp);

        amrex::CpuBndryFuncFab bndry_func(nullptr);
        amrex::PhysBCFunct<amrex::CpuBndryFuncFab>
            cphysbc(geom[lev-1],bcs_,bndry_func);
        amrex::PhysBCFunct<amrex::CpuBndryFuncFab>
            fphysbc(geom[lev],bcs_,bndry_func);

        // CellConservativeLinear interpolator from AMReX_Interpolator.H
        amrex::Interpolater* mapper = &amrex::cell_cons_interp;

        amrex::FillPatchTwoLevels(mf, 0.0_wp,
                                  cmf, ctime, fmf, ftime,
                                  0, 0, mf.nComp(),
                                  geom[lev-1], geom[lev],
                                  cphysbc, 0, fphysbc, 0,
                                  refRatio(lev-1), mapper, bcs_, 0);
    }

}

void AmrCoreFlash::fillFromCoarse(amrex::MultiFab& mf, const unsigned int lev) {

    amrex::CpuBndryFuncFab bndry_func(nullptr);
    amrex::PhysBCFunct<amrex::CpuBndryFuncFab>
        cphysbc(geom[lev-1],bcs_,bndry_func);
    amrex::PhysBCFunct<amrex::CpuBndryFuncFab>
        fphysbc(geom[lev  ],bcs_,bndry_func);

    // CellConservativeLinear interpolator from AMReX_Interpolator.H
    amrex::Interpolater* mapper = &amrex::cell_cons_interp;

    amrex::InterpFromCoarseLevel(mf, 0.0_wp, unk_[lev-1],
                                 0, 0, mf.nComp(),
                                 geom[lev-1], geom[lev],
                                 cphysbc, 0, fphysbc, 0,
                                 refRatio(lev-1), mapper, bcs_, 0);
}


}
