#include "Tile.h"

/**
 *
 */
Tile::Tile(amrex::MFIter& itor,
           amrex::MultiFab& mfab,
           const amrex::Geometry& geometry)
    : interior_(itor.validbox()),
      GC_(itor.fabbox()),
      data_(mfab.array(itor)),
      geometry_(geometry) {  }

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
    amrex::XDim3   deltas;

    deltas.x = geometry_.CellSize(0);
    deltas.y = geometry_.CellSize(1);
    deltas.z = geometry_.CellSize(2);

    return deltas;
}

