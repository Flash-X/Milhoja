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
#include <AMReX_Geometry.H>

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
void    Grid::initDomain(const grid::Real xMin, const grid::Real xMax,
                         const grid::Real yMin, const grid::Real yMax,
                         const grid::Real zMin, const grid::Real zMax,
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
  * getDomainLo returns a vector with 
  * the lower boundary of the domain.
  * Note: returns 0.0 for any dimension
  * higher than NDIM.
  */
grid::Vector<grid::Real>    Grid::getDomainLo() {
    grid::Vector<grid::Real> domainLo{0_rt,0_rt,0_rt};
    amrex::Geometry* geom = amrex::AMReX::top()->getDefaultGeometry();
    for(unsigned int i=0;i<NDIM;i++){
      domainLo[i] = geom->ProbLo(i);
    }
    return domainLo;
}

/**
  * getDomainHi returns a vector with 
  * the upper boundary of the domain.
  * Note: returns 0.0 for any dimension
  * higher than NDIM.
  */
grid::Vector<grid::Real>    Grid::getDomainHi() {
    grid::Vector<grid::Real> domainHi{0_rt,0_rt,0_rt};
    amrex::Geometry* geom = amrex::AMReX::top()->getDefaultGeometry();
    for(unsigned int i=0;i<NDIM;i++){
      domainHi[i] = geom->ProbHi(i);
    }
    return domainHi;
}

/**
  * getDeltas returns the vector {dx,dy,dz} for a given level.
  * Note: returns 0.0 for any dimension higher than NDIM.
  *
  * @param level The level of refinement (0-based).
  */
grid::Vector<grid::Real>    Grid::getDeltas(unsigned int level) {
    grid::Vector<grid::Real> deltas{0_rt,0_rt,0_rt};
    //DEV NOTE: Why does top()->GetDefaultGeometry() not get the right cell sizes? 
    //amrex::Geometry* geom = amrex::AMReX::top()->getDefaultGeometry();
    Grid&   grid = Grid::instance();
    amrex::Geometry&  geom = grid.geometry();
    for(unsigned int i=0;i<NDIM;i++){
      deltas[i] = geom.CellSize(i);
    }
    return deltas;
}

/**
  * getBlkCenterCoords returns the (real) coordinates of the
  * center of the given block/tile.
  * Note: returns 0.0 for any dimension higher than NDIM.
  *
  * @param tileDesc A Tile object.
  */
grid::Vector<grid::Real>    Grid::getBlkCenterCoords(const Tile& tileDesc) {
    grid::Vector<grid::Real> coords{0_rt,0_rt,0_rt};
    Grid&   grid = Grid::instance();
    grid::Vector<grid::Real> dx = grid.getDeltas(tileDesc.level());
    grid::Vector<grid::Real> x0 = grid.getDomainLo();
    grid::Vector<int> lo = tileDesc.loVect();
    grid::Vector<int> hi = tileDesc.hiVect();
    for(unsigned int i=0;i<NDIM;i++){
      coords[i] = x0[i] + dx[i] * static_cast<grid::Real>(lo[i]+hi[i]) / 2.0;
    }
    return coords;
}

/**
  * getMaxRefinement returns lrefine_max (the maximum possible refinement level).
  * Note: 0-based.
  */
unsigned int Grid::getMaxRefinement() {
    //TODO obviously has to change when AMR is implemented
    return 0;
}

/**
  * getMaxRefinement returns the highest level of blocks actually in existence. 
  * Note: 0-based.
  */
unsigned int Grid::getMaxLevel() {
    //TODO obviously has to change when AMR is implemented
    return 0;
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

