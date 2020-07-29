#include "Grid.h"

#include <cassert>
#include <stdexcept>

#include <AMReX_Vector.H>
#include <AMReX_IntVect.H>
#include <AMReX_IndexType.H>
#include <AMReX_Box.H>
#include <AMReX_RealBox.H>
#include <AMReX_BoxArray.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_CoordSys.H>
#include <AMReX_Geometry.H>

#include "Flash.h"
#include "constants.h"
#include "Tile.h"
#include "RuntimeAction.h"
#include "ThreadTeamDataType.h"
#include "ThreadTeam.h"
#include "Grid_Axis.h"
#include "Grid_Edge.h"

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
 * initDomain creates the domain in AMReX.
 *
 * @param probMin The physical lower boundary of the domain.
 * @param probMax The physical upper boundary of the domain.
 * @param nBlocks The number of root blocks in each direction.
 * @param nVars Number of physical variables.
 * @param initBlock Function pointer to the simulation's initBlock routine.
 */
void    Grid::initDomain(const RealVect& probMin,
                         const RealVect& probMax,
                         const IntVect& nBlocks,
                         const unsigned int nVars,
                         ACTION_ROUTINE initBlock) {
    // TODO: Error check all given parameters
    if (unk_) {
        throw std::logic_error("[Grid::initDomain] Grid unit's initDomain already called");
    } else if (!initBlock) {
        throw std::logic_error("[Grid::initDomain] Null initBlock function pointer given");
    }

    amrex::IntVect nCells_am{LIST_NDIM(NXB,NYB,NZB)};
    amrex::RealVect probMin_am = amrex::RealVect(probMin);
    amrex::RealVect probMax_am = amrex::RealVect(probMax);
    amrex::IntVect nBlocks_am = amrex::IntVect(nBlocks);

    //***** SETUP DOMAIN, PROBLEM, and MESH
    amrex::IndexType    ccIndexSpace(amrex::IntVect(0));
    amrex::IntVect      domainLo{0};
    amrex::IntVect      domainHi = nBlocks_am * nCells_am - 1;
    amrex::Box          domain{domainLo, domainHi, ccIndexSpace};

    amrex::BoxArray     ba{domain};
    ba.maxSize(nCells_am);

    amrex::DistributionMapping  dm{ba};

    // Setup with Cartesian coordinate and non-periodic BC so that we can set
    // the BC ourselves
    int coordSystem = 0;  // Cartesian
    amrex::RealBox      probDomain{probMin_am.dataPtr(), probMax_am.dataPtr()};
    geometry_ = amrex::Geometry(domain, probDomain,
                                coordSystem, {LIST_NDIM(0,0,0)} );

    unsigned int   level = 0;
    unk_ = new amrex::MultiFab(ba, dm, nVars, NGUARD);

    // TODO: Thread count should be a runtime variable
    RuntimeAction    action;
    action.name = "initBlock";
    action.nInitialThreads = 4;
    action.teamType = ThreadTeamDataType::BLOCK;
    action.routine = initBlock;

    ThreadTeam  team(4, 1);
    team.startCycle(action, "Cpu");
    for (amrex::MFIter  itor(*unk_); itor.isValid(); ++itor) {
        team.enqueue( std::shared_ptr<DataItem>{ new Tile{itor, level} } );
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
  * getProbLo gets the physical lower boundary of the domain.
  *
  * @return A real vector: <xlo, ylo, zlo>
  */
RealVect    Grid::getProbLo() const {
    return RealVect{geometry_.ProbLo()};
}

/**
  * getProbHi gets the physical upper boundary of the domain.
  *
  * @return A real vector: <xhi, yhi, zhi>
  */
RealVect    Grid::getProbHi() const {
    return RealVect{geometry_.ProbHi()};
}

/**
  * getDeltas gets the cell size for a given level.
  *
  * @param level The level of refinement (0 is coarsest).
  * @return The vector <dx,dy,dz> for a given level.
  */
RealVect    Grid::getDeltas(const unsigned int level) const {
    return RealVect{geometry_.CellSize()};
}

/**
  * getBlkCenterCoords gets the physical coordinates of the
  * center of the given tile.
  *
  * @param tileDesc A Tile object.
  * @return A real vector with the physical center coordinates of the tile.
  */
RealVect    Grid::getBlkCenterCoords(const Tile& tileDesc) const {
    RealVect dx = getDeltas(tileDesc.level());
    RealVect x0 = getProbLo();
    IntVect lo = tileDesc.lo();
    IntVect hi = tileDesc.hi();
    RealVect coords = x0 + dx*RealVect(lo+hi+1)*0.5_wp;
    return coords;
}

/** getCellFaceAreaLo gets lo face area of a cell with given (integer) coordinates
  *
  * @param axis Axis of desired face, returns the area of the lo side.
  * @param lev Level (0-based)
  * @param coord Cell-centered coordinates (integer, 0-based)
  * @return area of face (Real)
  */
Real  Grid::getCellFaceAreaLo(const unsigned int axis, const unsigned int lev, const IntVect& coord) const {
    return geometry_.AreaLo( amrex::IntVect(coord) , axis);
}

/** getCellVolume gets the volume of a cell with given (integer) coordinates
  *
  * @param lev Level (0-based)
  * @param coord Cell-centered coordinates (integer, 0-based)
  * @return Volume of cell (Real)
  */
Real  Grid::getCellVolume(const unsigned int lev, const IntVect& coord) const {
    return geometry_.Volume( amrex::IntVect(coord) );
}

/** fillCellCoords fills a Real array (passed by pointer) with the
  * cell coordinates in a given range
  *
  * @param axis Axis of desired coord (allowed: Axis::{I,J,K})
  * @param edge Edge of desired coord (allowed: Edge::{Left,Right,Center})
  * @param lev Level (0-based)
  * @param lo Lower bound of range (cell-centered 0-based integer coordinates)
  * @param hi Upper bound of range (cell-centered 0-based integer coordinates)
  * @param coordPtr Real Ptr to array of length hi[axis]-lo[axis]+1.
  */
void    Grid::fillCellCoords(const unsigned int axis, const unsigned int edge, const unsigned int lev, const IntVect& lo, const IntVect& hi, Real* coordPtr) const {
#ifndef GRID_ERRCHECK_OFF
    if(axis!=Axis::I && axis!=Axis::J && axis!=Axis::K ){
        throw std::logic_error("Grid::fillCellCoords: Invalid axis.");
    }
    if(edge!=Edge::Left && edge!=Edge::Right && edge!=Edge::Center){
        throw std::logic_error("Grid::fillCellCoords: Invalid edge.");
    }
#endif
    amrex::Box range{ amrex::IntVect(lo), amrex::IntVect(hi) };
    int nElements = hi[axis] - lo[axis] + 1;
    int offset = 0; //accounts for indexing of left/right cases

    //coordvec is length nElements + 1 if edge is Left or Right
    amrex::Vector<amrex::Real> coordvec;
    switch (edge) {
        case Edge::Left:
            geometry_.GetEdgeLoc(coordvec,range,axis);
            break;
        case Edge::Right:
            offset = 1;
            geometry_.GetEdgeLoc(coordvec,range,axis);
            break;
        case Edge::Center:
            geometry_.GetCellLoc(coordvec,range,axis);
            break;
    }

    //copy results to output
    for(int i=0; i<nElements; ++i) {
        coordPtr[i] = coordvec[i+offset];
    }
}

/** fillCellFaceAreasLo fills a Real array (passed by pointer) with the
  * cell face areas in a given range.
  * DEV NOTE: I assumed CoordSys::SetFaceArea corresponds to AreaLo (not AreaHi)
  *
  * @param axis Axis of desired coord (allowed: Axis::{I,J,K})
  * @param lev Level (0-based)
  * @param lo Lower bound of range (cell-centered 0-based integer coordinates)
  * @param hi Upper bound of range (cell-centered 0-based integer coordinates)
  * @param areaPtr Real Ptr to some fortran-style data structure. Will be filled with areas.
  *             Should be of shape (lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2], 1).
  */
void    Grid::fillCellFaceAreasLo(const unsigned int axis, const unsigned int lev, const IntVect& lo, const IntVect& hi, Real* areaPtr) const {
#ifndef GRID_ERRCHECK_OFF
    if(axis!=Axis::I && axis!=Axis::J && axis!=Axis::K ){
        throw std::logic_error("Grid::fillCellFaceAreasLo: Invalid axis.");
    }
#endif
    amrex::Box range{ amrex::IntVect(lo), amrex::IntVect(hi) };
    amrex::FArrayBox area_fab{range,1,areaPtr};
    geometry_.CoordSys::SetFaceArea(area_fab,range,axis);
}


/** fillCellVolumes fills a Real array (passed by pointer) with the
  * volumes of cells in a given range
  *
  * @param lev Level (0-based)
  * @param lo Lower bound of range (cell-centered 0-based integer coordinates)
  * @param hi Upper bound of range (cell-centered 0-based integer coordinates)
  * @param vols Real Ptr to some fortran-style data structure. Will be filled with volumes.
  *             Should be of shape (lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2], 1).
  */
void    Grid::fillCellVolumes(const unsigned int lev, const IntVect& lo, const IntVect& hi, Real* volPtr) const {
    amrex::Box range{ amrex::IntVect(lo), amrex::IntVect(hi) };
    amrex::FArrayBox vol_fab{range,1,volPtr};
    geometry_.CoordSys::SetVolume(vol_fab,range);
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
