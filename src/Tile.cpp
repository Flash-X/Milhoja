#include "Tile.h"

#include "Grid.h"
#include "constants.h"

/**
 *
 */
Tile::Tile(amrex::MFIter& itor)
    : interior_(itor.validbox()),
      GC_(itor.fabbox()),
      data_(Grid<NXB,NYB,NZB,NGUARD>::instance()->unk().array(itor)) {  }

/**
 *
 */
Tile::~Tile(void) {  }

/**
 *
 */
amrex::Dim3  Tile::lo(void) const {
    return amrex::lbound(interior_);
}

/**
 *
 */
amrex::Dim3  Tile::hi(void) const {
    return amrex::ubound(interior_);
}

/**
 *
 */
const int*  Tile::loVect(void) const {
    return interior_.loVect();
}

/**
 *
 */
const int*  Tile::hiVect(void) const {
    return interior_.hiVect();
}

/**
 *
 */
const amrex::Box&  Tile::interior(void) const {
    return interior_;
}

/**
 *
 */
amrex::Dim3  Tile::loGC(void) const {
    return amrex::lbound(GC_);
}

/**
 *
 */
amrex::Dim3  Tile::hiGC(void) const {
    return amrex::ubound(GC_);
}

/**
 *
 */
const amrex::Box&  Tile::interiorAndGC(void) const {
    return GC_;
}

/**
 *
 */
amrex::Array4<amrex::Real>& Tile::data(void) {
    return data_;
}

/**
 *
 */
amrex::XDim3 Tile::deltas(void) const {
    amrex::Geometry& geometry = Grid<NXB,NYB,NZB,NGUARD>::instance()->geometry();

    amrex::XDim3   deltas;

    deltas.x = geometry.CellSize(0);
    deltas.y = geometry.CellSize(1);
    deltas.z = geometry.CellSize(2);

    return deltas;
}

