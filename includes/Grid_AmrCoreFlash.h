#ifndef AMRCOREFLASH_H__
#define AMRCOREFLASH_H__

#include <AMReX_AmrCore.H>

#include "actionRoutine.h"
#include <AMReX_MultiFab.H>

namespace orchestration {

/** \brief Manages AMR functionality through AMReX.
  *
  * AmrCoreFlash is built upon amrex::AmrCore. It uses inherited methods
  * from amrex::AmrCore to  manage the mesh. AmrCoreFlash also owns the
  * physical data (stored in unk_, an array of AMReX MultiFabs), and manages
  * it through several overrides to virtual amrex::AmrCore methods.
  */
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

    //! Get reference to multifab with physical data for that level.
    amrex::MultiFab& unk(const unsigned int lev) {
#ifndef GRID_ERRCHECK_OFF
        if(lev>finest_level) {
            throw std::logic_error("[AmrCoreFlash::unk]: tried to get unk "
                                   "for lev>finest_level");
        }
#endif
        return unk_[lev];
    }

    //! Get number of total global blocks.
    int globalNumBlocks() {
        int sum = 0;
        for(int i=0; i<=finest_level; ++i) {
            sum += unk_[i].size();
        }
        return sum;
    }

    void writeMultiPlotfile(const std::string& filename) const;

private:
    std::vector<amrex::MultiFab> unk_; //!< Physical data, one MF per level

    // Pointers to physics routines are cached here so they can be specified
    // only once. More thought should be given to this design.
    ACTION_ROUTINE initBlock_; //!< Routine for initialializing data per block

};


}
#endif
