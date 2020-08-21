#include "Grid_AmrCoreFlash.h"

#include "Grid.h"
#include "OrchestrationLogger.h"
#include "RuntimeAction.h"
#include "ThreadTeamDataType.h"
#include "ThreadTeam.h"
#include <AMReX_PlotFileUtil.H>

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
                      std::to_string(lev) + "from coarse...";
    Logger::instance().log(msg);
#endif

    throw std::logic_error("need MakeNewLevelFromCoarse callback");
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
    throw std::logic_error("need RemakeLevel callback");
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
    throw std::logic_error("need ClearLevel callback");
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

    // Initialize data in unk_ to 0.0.
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

    // DO A GC FILL HERE

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
    std::string msg2 = "[AmrCoreFlash::ErrorEst] Did ErrorEst for level " +
                      std::to_string(lev) + ".";
    Logger::instance().log(msg2);
#endif
}


}
