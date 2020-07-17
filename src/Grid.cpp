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
                         TASK_FCN initBlock) {
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
    amrex::RealBox      physicalDomain{probMin_am.dataPtr(), probMax_am.dataPtr()};
    geometry_ = amrex::Geometry(domain, physicalDomain,
                                coordSystem, {LIST_NDIM(0,0,0)} );

    assert(nBlocks.product() == ba.size());
    assert((nBlocks*IntVect(nCells_am)).product() == ba.numPts());
    for (unsigned int i=0; i<ba.size(); ++i) {
        assert(ba[i].size() == nCells_am);
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
  * getDomainLo gets the lower boundary of the domain.
  *
  * @return A real vector: <xlo, ylo, zlo>
  */
RealVect    Grid::getDomainLo() const {
    return RealVect{geometry_.ProbLo()};
}

/**
  * getDomainHi gets the upper boundary of the domain.
  *
  * @return A real vector: <xhi, yhi, zhi>
  */
RealVect    Grid::getDomainHi() const {
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
    RealVect x0 = getDomainLo();
    IntVect lo = tileDesc.lo();
    IntVect hi = tileDesc.hi();
    RealVect coords = x0 + dx*RealVect(lo+hi+1)*0.5_wp;
    return coords;
}

/** getCellFaceArea gets face area of a cell with given (integer) coordinates
  *
  * @param lev level (0-based)
  * @param coord coordinates (integer, 0-based)
  * @return Area of face (Real)
  */
Real  Grid::getCellFaceArea(const unsigned int axis, const unsigned int lev, const IntVect& coord) const {
    Real area{0.0_wp};
    area = geometry_.AreaLo( amrex::IntVect(coord) , axis);
    if (area != geometry_.AreaHi( amrex::IntVect(coord) ,axis)) {
        throw std::logic_error("Something going on in getCellFaceArea");
    }

    return area;
}

/** getCellVolume gets the volume of a cell with given (integer) coordinates
  *
  * @param lev Level (0-based)
  * @param coord Coordinates (integer, 0-based)
  * @return Volume of cell (Real)
  */
Real  Grid::getCellVolume(const unsigned int lev, const IntVect& coord) const {
    return geometry_.Volume( amrex::IntVect(coord) );
}


/** fillCellVolumes fills a Real array (passed by pointer) with the
  * volumes of a cell in a given range
  *
  * @param lev Level (0-based)
  * @param lo lower bound of coordinates (integer, 0-based)
  * @param hi upper bound of coordinates (integer, 0-based)
  * @param vols Real Ptr to some fortran-style data structure. Will be filled with volumes.
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
