#include "Grid_AmrCoreFlash.h"

#include "Grid.h"
#include "OrchestrationLogger.h"
#include "RuntimeAction.h"
#include "ThreadTeamDataType.h"
#include "ThreadTeam.h"

#include "Flash.h"

namespace orchestration {

/** \brief Constructor for AmrCoreFlash
  *
  * Creates blank multifabs on each level.
  */
AmrCoreFlash::AmrCoreFlash(ACTION_ROUTINE initBlock)
    : initBlock_{initBlock} {

    // Allocate and resize unk_ (vector of Multifabs).
    unk_.resize(max_level);
}

//! Default constructor
AmrCoreFlash::~AmrCoreFlash() {
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
    std::string msg = "[AmrCoreFlash] MakeNewLevelFromCoarse for level " +
                      std::to_string(lev) + "...";
    Logger::instance().log(msg);
#endif
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
    std::string msg = "[AmrCoreFlash] Remaking level " +
                      std::to_string(lev) + "...";
    Logger::instance().log(msg);
#endif
}

/**
  * \brief Clear level
  *
  * \param lev Level being cleared
  */
void AmrCoreFlash::ClearLevel (int lev) {
#ifdef GRID_LOG
    std::string msg = "[AmrCoreFlash] Clearing level " +
                      std::to_string(lev) + "...";
    Logger::instance().log(msg);
#endif
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
    std::string msg = "[AmrCoreFlash] Creating level " +
                      std::to_string(lev) + "...";
    Logger::instance().log(msg);
#endif

    Grid& grid = Grid::instance();

    // Build multifab unk_[lev].
    unk_[lev].define(ba, dm, NUNKVAR, NGUARD);

    // Initialize data in unk_[lev].
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

#ifdef GRID_LOG
    std::string msg2 = "[AmrCoreFlash] Created level " +
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
    std::string msg = "[AmrCoreFlash] Doing ErrorEst for level " +
                      std::to_string(lev) + "...";
    Logger::instance().log(msg);
#endif
}


}
