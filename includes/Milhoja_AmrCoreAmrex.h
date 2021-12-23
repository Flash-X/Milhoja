#ifndef MILHOJA_AMR_CORE_AMREX_H__
#define MILHOJA_AMR_CORE_AMREX_H__

#include <AMReX_AmrCore.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_PhysBCFunct.H>

#include "Milhoja.h"
#include "Milhoja_actionRoutine.h"

#ifndef MILHOJA_GRID_AMREX
#error "This file need not be compiled if the AMReX backend isn't used"
#endif

namespace milhoja {

/** \brief Manages AMR functionality through AMReX.
  *
  * AmrCoreAmrex is built upon amrex::AmrCore. It uses inherited methods
  * from amrex::AmrCore to  manage the mesh. AmrCoreAmrex also owns the
  * physical data (stored in unk_, an array of AMReX MultiFabs), and manages
  * it through several overrides to virtual amrex::AmrCore methods.
  */
class AmrCoreAmrex
    : public amrex::AmrCore
{
public:
    AmrCoreAmrex(const unsigned int nGuard,
                 const unsigned int nCcVars,
                 ACTION_ROUTINE initBlock,
                 const unsigned int nDistributorThreads,
                 const unsigned int nRuntimeThreads,
                 ERROR_ROUTINE errorEst);
    ~AmrCoreAmrex();

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
            throw std::logic_error("[AmrCoreAmrex::unk]: tried to get unk "
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


    void writeMultiPlotfile(const std::string& filename,
                            const amrex::Vector<std::string>& names) const;

private:
    std::vector<amrex::MultiFab> unk_; //!< Physical data, one MF per level

    amrex::Vector<amrex::BCRec> bcs_; //!< Boundary conditions

    //----- GRID CONFIGURATION VALUES OWNED BY AmrCore
    // These cannot be acquired from AMReX, are not needed in GridAmrex, and
    // play an important role here in terms of constructing MultiFabs.
    //
    // NOTE: nCcVars_ could be retrieved with nComp() from a MultiFab that
    // already exists, but this class has to establish the first MultiFab at
    // some time after construction.
    //
    // We would prefer to store these as unsigned int, but AmrCore works with
    // them as ints.  Therefore, we will eagerly cast and store these results.
    const int   nGuard_;
    const int   nCcVars_;

    // Pointers to physics routines are cached here so they can be specified
    // only once. More thought should be given to this design.
    ACTION_ROUTINE initBlock_; //!< Routine for initialializing data per block
    unsigned int   nThreads_initBlock_;  //!< Number of runtime threads to use for computing the ICs
    unsigned int   nDistributorThreads_initBlock_;  //!< Number of host threads to use for distributing data items for computing the ICs
    ERROR_ROUTINE errorEst_; //!< Routine for marking blocks for refinement

};


}
#endif
