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

    IntVect nCells{LIST_NDIM(NXB,NYB,NZB)};

    //***** SETUP DOMAIN, PROBLEM, and MESH
    amrex::IndexType    ccIndexSpace(amrex::IntVect(LIST_NDIM(0,0,0)));
    amrex::IntVect      domainLo{LIST_NDIM(0,0,0)};
    IntVect hi = nBlocks * nCells - 1;
    amrex::IntVect      domainHi{LIST_NDIM(hi[0],hi[1],hi[2])};
    amrex::Box          domain = amrex::Box(domainLo, domainHi, ccIndexSpace);

    amrex::BoxArray     ba(domain);
    ba.maxSize(amrex::IntVect(LIST_NDIM(nCells[0],nCells[1],nCells[2])));

    amrex::DistributionMapping  dm(ba);

    // Setup with Cartesian coordinate and non-periodic BC so that we can set
    // the BC ourselves
    int coordSystem = 0;  // Cartesian
    amrex::RealBox   physicalDomain = amrex::RealBox(probMin.dataPtr(),
                                                     probMax.dataPtr());
    geometry_ = amrex::Geometry(domain, physicalDomain,
                                coordSystem, {LIST_NDIM(0, 0, 0)});

    assert(nBlocks.product() == ba.size());
    assert((nBlocks*nCells).product() == ba.numPts());
    for (unsigned int i=0; i<ba.size(); ++i) {
        assert(ba[i].size() == amrex::IntVect(LIST_NDIM(nCells[0],nCells[1],nCells[2])));
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
  * Note: returns 0.0 for any dimension
  * higher than NDIM.
  *
  * @return A real vector: <xlo, ylo, zlo>
  */
RealVect    Grid::getDomainLo() const {
    Grid&   grid = Grid::instance();
    amrex::Geometry&  geom = grid.geometry();

    const Real* probLo = geom.ProbLo();
    RealVect domainLo{LIST_NDIM(probLo[0],probLo[1],probLo[2])};

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
    Grid&   grid = Grid::instance();
    amrex::Geometry&  geom = grid.geometry();

    const Real* probHi = geom.ProbHi();
    RealVect domainHi{LIST_NDIM(probHi[0],probHi[1],probHi[2])};

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
    Grid&   grid = Grid::instance();
    amrex::Geometry&  geom = grid.geometry();

    RealVect deltas;
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
    Grid&   grid = Grid::instance();
    amrex::Geometry&  geom = grid.geometry();

    Real area{0.0_wp};
    amrex::IntVect coord_am = amrex::IntVect(LIST_NDIM(coord[0],coord[1],coord[2]));
    area = geom.AreaLo(coord_am,axis);
    if (area != geom.AreaHi(coord_am,axis)) throw std::logic_error("Something going on in getCellFaceArea");

    return area;
}

/** getCellVolume gets the volume of a cell with given (integer) coordinates
  *
  * @param lev Level (0-based)
  * @param coord Coordinates (integer, 0-based)
  * @return Volume of cell (Real)
  */
Real  Grid::getCellVolume(const unsigned int lev, const IntVect& coord) const {
    Grid&   grid = Grid::instance();
    amrex::Geometry&  geom = grid.geometry();
    RealVect deltas = grid.getDeltas(lev);

    Real vol{0.0_wp};
    amrex::IntVect coord_am = amrex::IntVect(LIST_NDIM(coord[0],coord[1],coord[2]));
    vol = geom.Volume(coord_am);

    return vol;
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
