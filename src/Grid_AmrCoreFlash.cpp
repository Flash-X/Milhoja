#include "Grid_AmrCoreFlash.h"

#include "Grid.h"
#include "OrchestrationLogger.h"
#include "RuntimeAction.h"
#include "ThreadTeamDataType.h"
#include "ThreadTeam.h"

#include <AMReX_PlotFileUtil.H>
#include <AMReX_Interpolater.H>
#include <AMReX_FillPatchUtil.H>

#include "Flash.h"
#include "Flash_par.h"

namespace orchestration {

/** \brief Constructor for AmrCoreFlash
  *
  * Creates blank multifabs on each level.
  */
AmrCoreFlash::AmrCoreFlash(ACTION_ROUTINE initBlock,
                           const unsigned int nRuntimeThreads,
                           ERROR_ROUTINE errorEst)
    : initBlock_{initBlock},
      nThreads_initBlock_{nRuntimeThreads},
      errorEst_{errorEst} {

    // Allocate and resize unk_ (vector of Multifabs).
    unk_.resize(max_level+1);

    // Set periodic Boundary conditions
    bcs_.resize(1);
    for(int i=0; i<NDIM; ++i) {
        bcs_[0].setLo(i, amrex::BCType::int_dir);
        bcs_[0].setHi(i, amrex::BCType::int_dir);
    }
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
#ifdef DENS_VAR_C
    // FIXME: Temporarily use the same generic names that are used in FLASH-X
    // AMREX-based plot files so that we can compare runtime vs. FLASH-X Sedov
    // results for testing.
//        names[DENS_VAR_C] = "dens";
        names[DENS_VAR_C] = "var0001";
#endif
#ifdef VELX_VAR_C
//        names[VELX_VAR_C] = "velx";
        names[VELX_VAR_C] = "var0008";
#endif
#ifdef VELY_VAR_C
//        names[VELY_VAR_C] = "vely";
        names[VELY_VAR_C] = "var0009";
#endif
#ifdef VELZ_VAR_C
//        names[VELZ_VAR_C] = "velz";
        names[VELZ_VAR_C] = "var0010";
#endif
#ifdef PRES_VAR_C
//        names[PRES_VAR_C] = "pres";
        names[PRES_VAR_C] = "var0006";
#endif
#ifdef ENER_VAR_C
//        names[ENER_VAR_C] = "ener";
        names[ENER_VAR_C] = "var0003";
#endif
#ifdef GAMC_VAR_C
//        names[GAMC_VAR_C] = "gamc";
        names[GAMC_VAR_C] = "var0004";
#endif
#ifdef GAME_VAR_C
//        names[GAME_VAR_C] = "game";
        names[GAME_VAR_C] = "var0005";
#endif
#ifdef TEMP_VAR_C
//        names[TEMP_VAR_C] = "temp";
        names[TEMP_VAR_C] = "var0007";
#endif
#ifdef EINT_VAR_C
//        names[EINT_VAR_C] = "eint";
        names[EINT_VAR_C] = "var0002";
#endif
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
                      std::to_string(lev) + " from coarse...";
    Logger::instance().log(msg);
#endif

    Grid& grid = Grid::instance();

    // Build multifab unk_[lev].
    unk_[lev].define(ba, dm, NUNKVAR, NGUARD);

    fillFromCoarse(unk_[lev], lev);
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

    amrex::MultiFab unkTmp{ba, dm, NUNKVAR, NGUARD};
    fillPatch(unkTmp, lev);

    std::swap(unkTmp, unk_[lev] );
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
    unk_[lev].clear();
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

    // Initialize data in unk_[lev] to 0.0.
    unk_[lev].setVal(0.0_wp);

    // Initalize simulation block data in unk_[lev].
    // Must fill interiors, GC optional.
    RuntimeAction    action;
    action.name = "initBlock";
    action.nInitialThreads = nThreads_initBlock_ - 1;
    action.teamType = ThreadTeamDataType::BLOCK;
    action.routine = initBlock_;
    ThreadTeam  team(nThreads_initBlock_, 1);
    team.startCycle(action, "Cpu");
    for (auto ti = grid.buildTileIter(lev); ti->isValid(); ti->next()) {
        team.enqueue( ti->buildCurrentTile() );
    }
    team.closeQueue();
    team.increaseThreadCount(1);
    team.wait();

    // DO A GC FILL HERE?

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

void AmrCoreFlash::fillPatch(amrex::MultiFab& mf, const unsigned int lev) {
    if (lev==0) {
        amrex::Vector<amrex::MultiFab*> smf;
        amrex::Vector<amrex::Real> stime;
        smf.push_back(&unk_[0]);
        stime.push_back(0.0_wp);

        amrex::CpuBndryFuncFab bndry_func(nullptr);
        amrex::PhysBCFunct<amrex::CpuBndryFuncFab>
            physbc(geom[lev],bcs_,bndry_func);

        amrex::FillPatchSingleLevel(mf, 0.0_wp, smf, stime,
                                    0, 0, mf.nComp(),
                                    geom[lev], physbc, 0);
    }
    else {
        amrex::Vector<amrex::MultiFab*> cmf, fmf;
        amrex::Vector<amrex::Real> ctime, ftime;
        cmf.push_back(&unk_[lev-1]);
        ctime.push_back(0.0_wp);
        fmf.push_back(&unk_[lev]);
        ftime.push_back(0.0_wp);

        amrex::CpuBndryFuncFab bndry_func(nullptr);
        amrex::PhysBCFunct<amrex::CpuBndryFuncFab>
            cphysbc(geom[lev-1],bcs_,bndry_func);
        amrex::PhysBCFunct<amrex::CpuBndryFuncFab>
            fphysbc(geom[lev],bcs_,bndry_func);

        // CellConservativeLinear interpolator from AMReX_Interpolator.H
        amrex::Interpolater* mapper = &amrex::cell_cons_interp;

        amrex::FillPatchTwoLevels(mf, 0.0_wp,
                                  cmf, ctime, fmf, ftime,
                                  0, 0, mf.nComp(),
                                  geom[lev-1], geom[lev],
                                  cphysbc, 0, fphysbc, 0,
                                  refRatio(lev-1), mapper, bcs_, 0);
    }

}

void AmrCoreFlash::fillFromCoarse(amrex::MultiFab& mf, const unsigned int lev) {

    amrex::CpuBndryFuncFab bndry_func(nullptr);
    amrex::PhysBCFunct<amrex::CpuBndryFuncFab>
        cphysbc(geom[lev-1],bcs_,bndry_func);
    amrex::PhysBCFunct<amrex::CpuBndryFuncFab>
        fphysbc(geom[lev  ],bcs_,bndry_func);

    // CellConservativeLinear interpolator from AMReX_Interpolator.H
    amrex::Interpolater* mapper = &amrex::cell_cons_interp;

    amrex::InterpFromCoarseLevel(mf, 0.0_wp, unk_[lev-1],
                                 0, 0, mf.nComp(),
                                 geom[lev-1], geom[lev],
                                 cphysbc, 0, fphysbc, 0,
                                 refRatio(lev-1), mapper, bcs_, 0);
}


}
