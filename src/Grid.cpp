#include "Grid.h"

#include <cassert>
#include <stdexcept>

#include <AMReX_Vector.H>
#include <AMReX_IntVect.H>
#include <AMReX_IndexType.H>
#include <AMReX_Box.H>
#include <AMReX_RealBox.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>

#include "Flash.h"
#include "constants.h"
#include "Tile.h"
#include "RuntimeAction.h"
#include "ThreadTeamDataType.h"
#include "ThreadTeam.h"

/**
 * 
 *
 * \return 
 */
Grid&   Grid::instance(void) {
    static Grid     gridSingleton;
    return gridSingleton;
}

/**
 * 
 *
 * \return 
 */
Grid::Grid(void) 
    : unk_(nullptr)
{
    amrex::Initialize(MPI_COMM_WORLD);
    destroyDomain();
}

/**
 * 
 */
Grid::~Grid(void) {
    // All Grid finalization is carried out here
    destroyDomain();
    amrex::Finalize();
}

/**
 *
 */
void    Grid::initDomain(const amrex::Real xMin, const amrex::Real xMax,
                         const amrex::Real yMin, const amrex::Real yMax,
                         const amrex::Real zMin, const amrex::Real zMax,
                         const unsigned int nBlocksX,
                         const unsigned int nBlocksY,
                         const unsigned int nBlocksZ,
                         const unsigned int nVars,
                         TASK_FCN initBlock) {
    // TODO: Error check all given parameters
    if (unk_) {
        throw std::logic_error("[Grid::initDomain] Grid unit's initDomain already called");
    } else if (!initBlock) {
        throw std::logic_error("[Grid::initDomain] Null initBlock function pointer given");
    }

    //***** SETUP DOMAIN, PROBLEM, and MESH
    amrex::IndexType    ccIndexSpace(amrex::IntVect(AMREX_D_DECL(0, 0, 0)));
    amrex::IntVect      domainLo(AMREX_D_DECL(0, 0, 0));
    amrex::IntVect      domainHi(AMREX_D_DECL(nBlocksX * NXB - 1,
                                              nBlocksY * NYB - 1,
                                              nBlocksZ * NZB - 1));
    amrex::Box          domain = amrex::Box(domainLo, domainHi, ccIndexSpace);
    amrex::BoxArray     ba(domain);
    ba.maxSize(amrex::IntVect(AMREX_D_DECL(NXB, NYB, NZB)));
    amrex::DistributionMapping  dm(ba);

    // Setup with Cartesian coordinate and non-periodic BC so that we can set
    // the BC ourselves
    int coordSystem = 0;  // Cartesian
    amrex::RealBox   physicalDomain = amrex::RealBox({AMREX_D_DECL(xMin, yMin, zMin)},
                                                     {AMREX_D_DECL(xMax, yMax, zMax)});
    geometry_ = amrex::Geometry(domain, physicalDomain,
                                coordSystem, {AMREX_D_DECL(0, 0, 0)});

    assert(nBlocksX * nBlocksY * nBlocksZ == ba.size());
    assert(NXB*nBlocksX * NYB*nBlocksY * NZB*nBlocksZ == ba.numPts());
    for (unsigned int i=0; i<ba.size(); ++i) {
        assert(ba[i].size() == amrex::IntVect(AMREX_D_DECL(NXB, NYB, NZB)));
    }

    unsigned int   level = 0;
    unk_ = new amrex::MultiFab(ba, dm, nVars, NGUARD);

    // TODO: Thread count should be a runtime variable
    RuntimeAction    action;
    action.name = "initBlock";
    action.nInitialThreads = 4;
    action.teamType = ThreadTeamDataType::BLOCK;
    action.routine = initBlock;

    ThreadTeam<Tile>  team(4, 1, "no.log");
    team.startTask(action, "Cpu");
    for (amrex::MFIter  itor(*unk_); itor.isValid(); ++itor) {
        Tile   tileDesc(itor, level);
        team.enqueue(tileDesc, true);
    }
    team.closeTask();
    team.wait();
}

/**
 *
 */
void    Grid::destroyDomain(void) {
    if (unk_) {
        delete unk_;
        unk_ = nullptr;
    }
    geometry_ = amrex::Geometry();
}

/**
 *
 */
void    Grid::writeToFile(const std::string& filename) const {
    amrex::Vector<std::string>    names(unk_->nComp());
    names[0] = "Density";
    names[1] = "Energy";

    amrex::WriteSingleLevelPlotfile(filename, *unk_, names, geometry_, 0.0, 0);
}

