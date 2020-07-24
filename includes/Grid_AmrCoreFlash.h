#ifndef AMRCOREFLASH_H__
#define AMRCOREFLASH_H__

#include <AMReX_AmrCore.H>

namespace orchestration {

class AmrCoreFlash
    : public amrex::AmrCore
{
    public:
    AmrCoreFlash();
    virtual ~AmrCoreFlash();

    virtual void MakeNewLevelFromCoarse (int lev, amrex::Real time, const amrex::BoxArray& ba, const amrex::DistributionMapping& dm) override;

    virtual void RemakeLevel (int lev, amrex::Real time, const amrex::BoxArray& ba, const amrex::DistributionMapping& dm) override;

    virtual void ClearLevel (int lev) override;

    virtual void MakeNewLevelFromScratch (int lev, amrex::Real time, const amrex::BoxArray& ba, const amrex::DistributionMapping& dm) override;

    virtual void ErrorEst (int lev, amrex::TagBoxArray& tags, amrex::Real time, int ngrow) override;

    private:

};

}

#endif
