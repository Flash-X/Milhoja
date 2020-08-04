#include "Grid_AmrCoreFlash.h"

#include "Grid.h"
#include "Flash.h"

#include "RuntimeAction.h"
#include "ThreadTeamDataType.h"
#include "ThreadTeam.h"

#include <iostream>

namespace orchestration {

AmrCoreFlash::AmrCoreFlash(ACTION_ROUTINE initBlock)
    : unk_{nullptr},
      initBlock_{initBlock} {}

AmrCoreFlash::~AmrCoreFlash() {
    if (unk_) {
        delete unk_;
        unk_ = nullptr;
    }
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
    Grid& grid = Grid::instance();

    if(lev==0) {
        delete unk_;
        unk_ =  new amrex::MultiFab(ba, dm, NUNKVAR, NGUARD);

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
    std::cout << "Doing MakeNewLevelFromScratch Callback" << std::endl;
}

void AmrCoreFlash::ErrorEst (int lev, amrex::TagBoxArray& tags,
                             amrex::Real time, int ngrow) {
    std::cout << "Doing ErrorEst Callback" << std::endl;
}


}
