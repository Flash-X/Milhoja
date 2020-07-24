#include "Grid_AmrCoreFlash.h"
#include "Grid.h"
#include "Flash.h"

#include <iostream>

namespace orchestration {

AmrCoreFlash::AmrCoreFlash() {
    std::cout << "Constructor" << std::endl;
}

AmrCoreFlash::~AmrCoreFlash() {
    std::cout << "Denstructor" << std::endl;
}

void AmrCoreFlash::MakeNewLevelFromCoarse (int lev, amrex::Real time, const amrex::BoxArray& ba, const amrex::DistributionMapping& dm)
{
    std::cout << "Doing MakeNewLevelFromCoarse Callback" << std::endl;
}

void AmrCoreFlash::RemakeLevel (int lev, amrex::Real time, const amrex::BoxArray& ba, const amrex::DistributionMapping& dm) {
    std::cout << "Doing RemakeLevel Callback" << std::endl;
}

void AmrCoreFlash::ClearLevel (int lev) {
    std::cout << "Doing ClearLevel Callback" << std::endl;
}

void AmrCoreFlash::MakeNewLevelFromScratch (int lev, amrex::Real time, const amrex::BoxArray& ba, const amrex::DistributionMapping& dm){
    Grid::instance().unk_ =  new amrex::MultiFab(ba, dm, NUNKVAR, NGUARD);
    std::cout << "Doing MakeNewLevelFromScratch Callback" << std::endl;
}

void AmrCoreFlash::ErrorEst (int lev, amrex::TagBoxArray& tags, amrex::Real time, int ngrow) {
    std::cout << "Doing ErrorEst Callback" << std::endl;
}


}
