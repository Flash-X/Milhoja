#include "Grid_AmrCoreFlash.h"

#include "Grid.h"
#include "Flash.h"

#include "RuntimeAction.h"
#include "ThreadTeamDataType.h"
#include "ThreadTeam.h"

#include <iostream>

namespace orchestration {

AmrCoreFlash::AmrCoreFlash(ACTION_ROUTINE initBlock)
    : initBlock_{initBlock} {

    // Allocate and resize unk_ (vector of Multifabs).
    unk_.resize(max_level);
}

AmrCoreFlash::~AmrCoreFlash() {
}

void AmrCoreFlash::MakeNewLevelFromCoarse (int lev, amrex::Real time,
            const amrex::BoxArray& ba, const amrex::DistributionMapping& dm) {
    std::cout << "Doing MakeNewLevelFromCoarse Callback" << std::endl;
}

void AmrCoreFlash::RemakeLevel (int lev, amrex::Real time,
            const amrex::BoxArray& ba, const amrex::DistributionMapping& dm) {
    std::cout << "Doing RemakeLevel Callback" << std::endl;
}

void AmrCoreFlash::ClearLevel (int lev) {
    std::cout << "Doing ClearLevel Callback" << std::endl;
}

void AmrCoreFlash::MakeNewLevelFromScratch (int lev, amrex::Real time,
            const amrex::BoxArray& ba, const amrex::DistributionMapping& dm) {
    std::cout << "Doing MakeNewLevelFromScratch Callback" << std::endl;
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
}

void AmrCoreFlash::ErrorEst (int lev, amrex::TagBoxArray& tags,
                             amrex::Real time, int ngrow) {
    std::cout << "Doing ErrorEst Callback" << std::endl;
}


}
