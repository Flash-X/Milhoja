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

namespace orchestration {

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
    if(!std::is_same<amrex::Real,Real>::value) {
      throw std::logic_error("amrex::Real does not match orchestration::Real");
    }
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
void    Grid::initDomain(const Real xMin, const Real xMax,
                         const Real yMin, const Real yMax,
                         const Real zMin, const Real zMax,
                         const unsigned int nBlocksX,
                         const unsigned int nBlocksY,
                         const unsigned int nBlocksZ,
                         const unsigned int nVars,
                         ACTION_ROUTINE initBlock) {
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
    team.startCycle(action, "Cpu");
    for (amrex::MFIter  itor(*unk_); itor.isValid(); ++itor) {
        team.enqueue( std::make_shared<Tile>(itor, level) );
    }
    team.closeQueue();
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
  * getDomainLo gets the lower boundary of the domain.
  * Note: returns 0.0 for any dimension
  * higher than NDIM.
  *
  * @return A real vector: <xlo, ylo, zlo>
  */
RealVect    Grid::getDomainLo() const {
    RealVect domainLo{0.0_wp,0.0_wp,0.0_wp};
    amrex::Geometry* geom = amrex::AMReX::top()->getDefaultGeometry();
    for(unsigned int i=0;i<NDIM;i++){
      domainLo[i] = geom->ProbLo(i);
    }
    return domainLo;
}

/**
  * getDomainHi gets the upper boundary of the domain.
  * Note: returns 0.0 for any dimension
  * higher than NDIM.
  *
  * @return A real vector: <xhi, yhi, zhi>
  */
RealVect    Grid::getDomainHi() const {
    RealVect domainHi{0.0_wp,0.0_wp,0.0_wp};
    amrex::Geometry* geom = amrex::AMReX::top()->getDefaultGeometry();
    for(unsigned int i=0;i<NDIM;i++){
      domainHi[i] = geom->ProbHi(i);
    }
    return domainHi;
}

/**
  * getDeltas gets the cell size for a given level.
  * Note: returns 0.0 for any dimension higher than NDIM.
  *
  * @param level The level of refinement (0 is coarsest).
  * @return The vector <dx,dy,dz> for a given level.
  */
RealVect    Grid::getDeltas(const unsigned int level) const {
    RealVect deltas{0.0_wp,0.0_wp,0.0_wp};
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
  * getBlkCenterCoords gets the physical coordinates of the
  * center of the given tile.
  * Note: returns 0.0 for any dimension higher than NDIM.
  *
  * @param tileDesc A Tile object.
  * @return A real vector with the physical center coordinates of the tile.
  */
RealVect    Grid::getBlkCenterCoords(const Tile& tileDesc) const {
    Grid&   grid = Grid::instance();
    RealVect dx = grid.getDeltas(tileDesc.level());
    RealVect x0 = grid.getDomainLo();
    IntVect lo = tileDesc.loVect();
    IntVect hi = tileDesc.hiVect();
    RealVect coords = x0 + dx*RealVect(lo+hi)*0.5_wp;
    return coords;
}

/**
  * getMaxRefinement returns the maximum possible refinement level. (Specified by user).
  *
  * @return Maximum refinement level of simulation.
  */
unsigned int Grid::getMaxRefinement() const {
    //TODO obviously has to change when AMR is implemented
    return 0;
}

/**
  * getMaxRefinement returns the highest level of blocks actually in existence. 
  *
  * @return The max level of existing blocks (0 is coarsest).
  */
unsigned int Grid::getMaxLevel() const {
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


}
