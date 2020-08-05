#ifndef AMRCOREFLASH_H__
#define AMRCOREFLASH_H__

#include <AMReX_AmrCore.H>

#include "actionRoutine.h"
#include <AMReX_MultiFab.H>

namespace orchestration {

class AmrCoreFlash
    : public amrex::AmrCore
{
public:
    AmrCoreFlash(ACTION_ROUTINE initBlock);
    ~AmrCoreFlash();

    // Overrides from AmrCore
    void MakeNewLevelFromCoarse (int lev, amrex::Real time,
                                 const amrex::BoxArray& ba,
                                 const amrex::DistributionMapping& dm) override;

    void RemakeLevel (int lev,
                      amrex::Real time,
                      const amrex::BoxArray& ba,
                      const amrex::DistributionMapping& dm) override;

    void ClearLevel (int lev) override;

    void MakeNewLevelFromScratch(int lev,
                                 amrex::Real time,
                                 const amrex::BoxArray& ba,
                                 const amrex::DistributionMapping& dm) override;

    void ErrorEst (int lev,
                   amrex::TagBoxArray& tags,
                   amrex::Real time,
                   int ngrow) override;

    // Allow Grid access to unk
    amrex::MultiFab& unk(const unsigned int lev) {
#ifndef GRID_ERRCHECK_OFF
        if(lev>finest_level) {
            throw std::logic_error("[AmrCoreFlash]: tried to get unk "
                                   "for lev>finest_level");
        }
#endif
        return unk_[lev];
    }

private:
    std::vector<amrex::MultiFab>   unk_;
    ACTION_ROUTINE initBlock_;

};


}
#endif
