#include "Milhoja_AmrCoreAmrex.h"

#include <stdexcept>

#include <AMReX_PlotFileUtil.H>
#include <AMReX_Interpolater.H>
#include <AMReX_FillPatchUtil.H>

#include "Milhoja_Grid.h"
#include "Milhoja_Logger.h"
#include "Milhoja_RuntimeAction.h"
#include "Milhoja_ThreadTeamDataType.h"
#include "Milhoja_ThreadTeam.h"

namespace milhoja {

/** \brief Constructor for AmrCoreAmrex
  *
  * Creates blank multifabs on each level.
  *
  * \todo Check for overflow on nGuard/nCcVars casts
  */
AmrCoreAmrex::AmrCoreAmrex(const unsigned int nGuard,
                           const unsigned int nCcVars,
                           ACTION_ROUTINE initBlock,
                           const unsigned int nDistributorThreads,
                           const unsigned int nRuntimeThreads,
                           ERROR_ROUTINE errorEst)
    : nGuard_{static_cast<int>(nGuard)},
      nCcVars_{static_cast<int>(nCcVars)},
      initBlock_{initBlock},
      nThreads_initBlock_{nRuntimeThreads},
      nDistributorThreads_initBlock_{nDistributorThreads},
      errorEst_{errorEst} {

#ifndef USE_THREADED_DISTRIBUTOR
      // Override if multithreading is disabled
      nDistributorThreads_initBlock_ = 1;
#endif

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
AmrCoreAmrex::~AmrCoreAmrex() {
}

//! Write all levels of unk_ to plotfile.
void AmrCoreAmrex::writeMultiPlotfile(const std::string& filename,
                                      const amrex::Vector<std::string>& names) const {
#ifdef GRID_LOG
    std::string msg = "[GridAmrex] Writing to plotfile: "+filename+"...";
    Logger::instance().log(msg);
#endif
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
void AmrCoreAmrex::MakeNewLevelFromCoarse (int lev, amrex::Real time,
            const amrex::BoxArray& ba, const amrex::DistributionMapping& dm) {
#ifdef GRID_LOG
    std::string msg = "[AmrCoreAmrex::MakeNewLevelFromCoarse] Making level " +
                      std::to_string(lev) + " from coarse...";
    Logger::instance().log(msg);
#endif

    Grid& grid = Grid::instance();

    // Build multifab unk_[lev].
    unk_[lev].define(ba, dm, nCcVars_, nGuard_);

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
void AmrCoreAmrex::RemakeLevel (int lev, amrex::Real time,
            const amrex::BoxArray& ba, const amrex::DistributionMapping& dm) {
#ifdef GRID_LOG
    std::string msg = "[AmrCoreAmrex::RemakeLevel] Remaking level " +
                      std::to_string(lev) + "...";
    Logger::instance().log(msg);
#endif

    amrex::MultiFab unkTmp{ba, dm, nCcVars_, nGuard_};
    fillPatch(unkTmp, lev);

    std::swap(unkTmp, unk_[lev] );
}

/**
  * \brief Clear level
  *
  * \param lev Level being cleared
  */
void AmrCoreAmrex::ClearLevel (int lev) {
#ifdef GRID_LOG
    std::string msg = "[AmrCoreAmrex::ClearLevel] Clearing level " +
                      std::to_string(lev) + "...";
    Logger::instance().log(msg);
#endif
    unk_[lev].clear();
}

/**
  * \brief Make new level from scratch
  *
  * \todo Simulations should be allowed to use the GPU for setting the ICs.
  * Therefore, we need a means for expressing if the CPU-only or GPU-only thread
  * team configuration should be used.  If the GPU-only configuration is
  * allowed, then we should allow for more than one distributor thread.
  *
  * \param lev Level being made
  * \param time Simulation time
  * \param ba BoxArray of level being made
  * \param dm DistributionMapping of leving being made
  */
void AmrCoreAmrex::MakeNewLevelFromScratch (int lev, amrex::Real time,
            const amrex::BoxArray& ba, const amrex::DistributionMapping& dm) {
#ifdef GRID_LOG
    std::string msg = "[AmrCoreAmrex::MakeNewLevelFromScratch] Creating level " +
                      std::to_string(lev) + "...";
    Logger::instance().log(msg);
#endif

    Grid& grid = Grid::instance();

    // Build multifab unk_[lev].
    unk_[lev].define(ba, dm, nCcVars_, nGuard_);

    // Initialize data in unk_[lev] to 0.0.
    unk_[lev].setVal(0.0_wp);

    // Initalize simulation block data in unk_[lev].
    // Must fill interiors, GC optional.
    if (nThreads_initBlock_ <= 0) {
        throw std::invalid_argument("[AmrCoreAmrex::AmrCoreAmrex] "
                                    "N computation threads must be positive");
    } else if (nDistributorThreads_initBlock_ != 1) {
        throw std::invalid_argument("[AmrCoreAmrex::AmrCoreAmrex] "
                                    "Only one distributor thread presently allowed");
    } else if (nDistributorThreads_initBlock_ > nThreads_initBlock_) {
        throw std::invalid_argument("[AmrCoreAmrex::AmrCoreAmrex] "
                                    "More distributor threads than computation threads");
    }

    RuntimeAction    action;
    action.name = "initBlock";
    action.nInitialThreads = nThreads_initBlock_ - nDistributorThreads_initBlock_;
    action.teamType = ThreadTeamDataType::BLOCK;
    action.routine = initBlock_;
    ThreadTeam  team(nThreads_initBlock_, 1);
    team.startCycle(action, "Cpu");

    for (auto ti = grid.buildTileIter(lev); ti->isValid(); ti->next()) {
        team.enqueue( ti->buildCurrentTile() );
    }
    team.closeQueue(nullptr);
    team.increaseThreadCount(nDistributorThreads_initBlock_);
    team.wait();

    // DO A GC FILL HERE?

#ifdef GRID_LOG
    std::string msg2 = "[AmrCoreAmrex::MakeNewLevelFromScratch] Created level " +
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
void AmrCoreAmrex::ErrorEst (int lev, amrex::TagBoxArray& tags,
                             amrex::Real time, int ngrow) {
#ifdef GRID_LOG
    std::string msg = "[AmrCoreAmrex::ErrorEst] Doing ErrorEst for level " +
                      std::to_string(lev) + "...";
    Logger::instance().log(msg);
#endif
    Grid& grid = Grid::instance();

    amrex::Vector<int> itags;

    //TODO use tiling for this loop?
    for (auto ti = grid.buildTileIter(lev); ti->isValid(); ti->next()) {
        std::shared_ptr<Tile> tileDesc = ti->buildCurrentTile();
        amrex::Box validbox{ amrex::IntVect(tileDesc->lo()),
                             amrex::IntVect(tileDesc->hi()) };
        amrex::TagBox& tagfab = tags[tileDesc->gridIndex()];
        tagfab.get_itags(itags,validbox);

        //errorEst_(lev, tags, time, ngrow, tileDesc);
        int* tptr = itags.dataPtr();
        errorEst_(tileDesc, tptr);

        tagfab.tags_and_untags(itags,validbox);
    }

#ifdef GRID_LOG
    std::string msg2 = "[AmrCoreAmrex::ErrorEst] Did ErrorEst for level " +
                      std::to_string(lev) + ".";
    Logger::instance().log(msg2);
#endif
}

void AmrCoreAmrex::fillPatch(amrex::MultiFab& mf, const unsigned int lev) {
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

void AmrCoreAmrex::fillFromCoarse(amrex::MultiFab& mf, const unsigned int lev) {

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
