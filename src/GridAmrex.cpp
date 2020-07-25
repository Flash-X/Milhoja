#include "GridAmrex.h"

#include <stdexcept>
#include <string>

#include <AMReX.H>
#include <AMReX_PlotFileUtil.H>

#include "Grid_Axis.h"
#include "Grid_Edge.h"

#include "Tile.h"
#include "RuntimeAction.h"
#include "ThreadTeamDataType.h"
#include "ThreadTeam.h"

namespace orchestration {

GridAmrex::GridAmrex(void) {
    if(!std::is_same<amrex::Real,Real>::value) {
      throw std::logic_error("amrex::Real does not match orchestration::Real");
    }

    // TODO : do parm parse manually instead of pretending to pass from command line
    std::string parfile = "/Users/tklosterman/Documents/orchestrationruntime/amrex_inputs";
    std::vector<char*> argvec;
    argvec.push_back( (char*)parfile.data() ); //technically should be binary name
    argvec.push_back( (char*)parfile.data() );
    argvec.push_back( nullptr );
    int argc = 2;
    char** argv = argvec.data();

    amrex::Initialize( argc , argv ,true,MPI_COMM_WORLD);
    destroyDomain();
}

GridAmrex::~GridAmrex(void) {
    // All Grid finalization is carried out here
    destroyDomain();
    amrex::Finalize();
}

void  GridAmrex::destroyDomain(void) {
    if (unk_) {
        delete unk_;
        unk_ = nullptr;
    }
    if (amrcore_) {
        delete amrcore_;
        amrcore_ = nullptr;
    }
    // TODO why is amrex not finalized here?
}

/**
 * initDomain creates the domain in AMReX.
 *
 * @param initBlock Function pointer to the simulation's initBlock routine.
 */
void GridAmrex::initDomain(TASK_FCN initBlock) {
    if (unk_) {
        throw std::logic_error("[GridAmrex::initDomain] Grid unit's initDomain already called");
    } else if (!initBlock) {
        throw std::logic_error("[GridAmrex::initDomain] Null initBlock function pointer given");
    }

    unsigned int   level = 0;
    amrcore_ = new AmrCoreFlash;
    amrcore_->InitFromScratch(0.0_wp);

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
  * getProbLo gets the physical lower boundary of the domain.
  *
  * @return A real vector: <xlo, ylo, zlo>
  */
RealVect    GridAmrex::getProbLo() const {
    return RealVect{amrcore_->Geom(0).ProbLo()};
}

/**
  * getProbHi gets the physical upper boundary of the domain.
  *
  * @return A real vector: <xhi, yhi, zhi>
  */
RealVect    GridAmrex::getProbHi() const {
    return RealVect{amrcore_->Geom(0).ProbHi()};
}

/**
  * getMaxRefinement returns the maximum possible refinement level. (Specified by user).
  *
  * @return Maximum refinement level of simulation.
  */
unsigned int GridAmrex::getMaxRefinement() const {
    //TODO obviously has to change when AMR is implemented
    return 0;
}

/**
  * getMaxRefinement returns the highest level of blocks actually in existence. 
  *
  * @return The max level of existing blocks (0 is coarsest).
  */
unsigned int GridAmrex::getMaxLevel() const {
    //TODO obviously has to change when AMR is implemented
    return 0;
}

/**
 *
 */
void    GridAmrex::writeToFile(const std::string& filename) const {
    amrex::Vector<std::string>    names(unk_->nComp());
    names[0] = "Density";
    names[1] = "Energy";

    amrex::WriteSingleLevelPlotfile(filename, *unk_, names, amrcore_->Geom(0), 0.0, 0);
}

/**
  * getDeltas gets the cell size for a given level.
  *
  * @param level The level of refinement (0 is coarsest).
  * @return The vector <dx,dy,dz> for a given level.
  */
RealVect    GridAmrex::getDeltas(const unsigned int level) const {
    return RealVect{amrcore_->Geom(0).CellSize()};
}


/** getCellFaceAreaLo gets lo face area of a cell with given (integer) coordinates
  *
  * @param axis Axis of desired face, returns the area of the lo side.
  * @param lev Level (0-based)
  * @param coord Cell-centered coordinates (integer, 0-based)
  * @return area of face (Real)
  */
Real  GridAmrex::getCellFaceAreaLo(const unsigned int axis, const unsigned int lev, const IntVect& coord) const {
    return amrcore_->Geom(0).AreaLo( amrex::IntVect(coord) , axis);
}

/** getCellVolume gets the volume of a cell with given (integer) coordinates
  *
  * @param lev Level (0-based)
  * @param coord Cell-centered coordinates (integer, 0-based)
  * @return Volume of cell (Real)
  */
Real  GridAmrex::getCellVolume(const unsigned int lev, const IntVect& coord) const {
    return amrcore_->Geom(0).Volume( amrex::IntVect(coord) );
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
void    GridAmrex::fillCellCoords(const unsigned int axis, const unsigned int edge, const unsigned int lev, const IntVect& lo, const IntVect& hi, Real* coordPtr) const {
#ifndef GRID_ERRCHECK_OFF
    if(axis!=Axis::I && axis!=Axis::J && axis!=Axis::K ){
        throw std::logic_error("GridAmrex::fillCellCoords: Invalid axis.");
    }
    if(edge!=Edge::Left && edge!=Edge::Right && edge!=Edge::Center){
        throw std::logic_error("GridAmrex::fillCellCoords: Invalid edge.");
    }
#endif
    amrex::Box range{ amrex::IntVect(lo), amrex::IntVect(hi) };
    int nElements = hi[axis] - lo[axis] + 1;
    int offset = 0; //accounts for indexing of left/right cases

    //coordvec is length nElements + 1 if edge is Left or Right
    amrex::Vector<amrex::Real> coordvec;
    switch (edge) {
        case Edge::Left:
            amrcore_->Geom(0).GetEdgeLoc(coordvec,range,axis);
            break;
        case Edge::Right:
            offset = 1;
            amrcore_->Geom(0).GetEdgeLoc(coordvec,range,axis);
            break;
        case Edge::Center:
            amrcore_->Geom(0).GetCellLoc(coordvec,range,axis);
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
void    GridAmrex::fillCellFaceAreasLo(const unsigned int axis, const unsigned int lev, const IntVect& lo, const IntVect& hi, Real* areaPtr) const {
#ifndef GRID_ERRCHECK_OFF
    if(axis!=Axis::I && axis!=Axis::J && axis!=Axis::K ){
        throw std::logic_error("GridAmrex::fillCellFaceAreasLo: Invalid axis.");
    }
#endif
    amrex::Box range{ amrex::IntVect(lo), amrex::IntVect(hi) };
    amrex::FArrayBox area_fab{range,1,areaPtr};
    amrcore_->Geom(0).CoordSys::SetFaceArea(area_fab,range,axis);
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
void    GridAmrex::fillCellVolumes(const unsigned int lev, const IntVect& lo, const IntVect& hi, Real* volPtr) const {
    amrex::Box range{ amrex::IntVect(lo), amrex::IntVect(hi) };
    amrex::FArrayBox vol_fab{range,1,volPtr};
    amrcore_->Geom(0).CoordSys::SetVolume(vol_fab,range);
}


}
