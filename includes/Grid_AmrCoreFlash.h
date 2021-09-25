#ifndef AMRCOREFLASH_H__
#define AMRCOREFLASH_H__

#include <AMReX_AmrCore.H>

#include "actionRoutine.h"
#include <AMReX_MultiFab.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_PhysBCFunct.H>

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
    AmrCoreFlash(const unsigned int nGuard,
                 const unsigned int nCcVars,
                 ACTION_ROUTINE initBlock,
                 ERROR_ROUTINE errorEst);
    ~AmrCoreFlash();

    void setInitDomainConfiguration(const unsigned int nDistributorThreads,
                                    const unsigned int nRuntimeThreads);

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

    void averageDownAll() {
        for (int lev = finest_level-1; lev >= 0; --lev)
        {
            amrex::average_down(unk_[lev+1],
                                unk_[lev],
                                geom[lev+1],
                                geom[lev],
                                0,
                                unk_[lev].nComp(),
                                refRatio(lev));
        }
    }

    void fillPatch(amrex::MultiFab& mf, const unsigned int lev);
    void fillFromCoarse(amrex::MultiFab& mf, const unsigned int lev);


    void writeMultiPlotfile(const std::string& filename) const;

private:
    std::vector<amrex::MultiFab> unk_; //!< Physical data, one MF per level

    amrex::Vector<amrex::BCRec> bcs_; //!< Boundary conditions

    unsigned int   nGuard_;
    unsigned int   nCcVars_;

    // Pointers to physics routines are cached here so they can be specified
    // only once. More thought should be given to this design.
    ACTION_ROUTINE initBlock_; //!< Routine for initialializing data per block
    unsigned int   nThreads_initBlock_;  //!< Number of runtime threads to use for computing the ICs
    unsigned int   nDistributorThreads_initBlock_;  //!< Number of host threads to use for distributing data items for computing the ICs
    ERROR_ROUTINE errorEst_; //!< Routine for marking blocks for refinement

};


}
#endif
