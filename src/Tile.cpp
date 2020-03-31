#include "Tile.h"

#include "Grid.h"
#include "Flash.h"
#include "constants.h"

/**
 *
 */
Tile::Tile(amrex::MFIter& itor, const unsigned int level)
    : gridIdx_(itor.index()),
      level_(level),
      interior_(nullptr),
      GC_(nullptr)
{
    interior_ = new amrex::Box(itor.validbox());
    GC_       = new amrex::Box(itor.fabbox());
}

Tile::Tile(const Tile& other)
    : gridIdx_(other.gridIdx_),
      level_(other.level_),
      interior_(nullptr),
      GC_(nullptr)
{
    interior_ = new amrex::Box(*(other.interior_));
    GC_       = new amrex::Box(*(other.GC_));
}

Tile::Tile(Tile&& other)
    : gridIdx_(other.gridIdx_),
      level_(other.level_),
      interior_(other.interior_),
      GC_(other.GC_)
{
    other.gridIdx_ = -1;
    other.level_ = 0;
    other.interior_ = nullptr;
    other.GC_ = nullptr;
}

Tile& Tile::operator=(Tile&& rhs) {
    gridIdx_ = rhs.gridIdx_;
    level_ = rhs.level_;
    interior_ = rhs.interior_;
    GC_ = rhs.GC_;

    rhs.gridIdx_ = -1;
    rhs.level_ = 0;
    rhs.interior_ = nullptr;
    rhs.GC_ = nullptr;

    return *this;
}

/**
 *
 */
Tile::~Tile(void) {
    if (interior_) {
        delete interior_;
        interior_ = nullptr;
    }
    if (GC_) {
        delete GC_;
        GC_ = nullptr;
    }
}

/**
 *
 */
amrex::Dim3  Tile::lo(void) const {
    return amrex::lbound(*interior_);
}

/**
 *
 */
amrex::Dim3  Tile::hi(void) const {
    return amrex::ubound(*interior_);
}

/**
 *
 */
const int*  Tile::loVect(void) const {
    return interior_->loVect();
}

/**
 *
 */
const int*  Tile::hiVect(void) const {
    return interior_->hiVect();
}

/**
 *
 */
const amrex::Box&  Tile::interior(void) const {
    return (*interior_);
}

/**
 *
 */
amrex::Dim3  Tile::loGC(void) const {
    return amrex::lbound(*GC_);
}

/**
 *
 */
amrex::Dim3  Tile::hiGC(void) const {
    return amrex::ubound(*GC_);
}

/**
 *
 */
const int*  Tile::loGCVect(void) const {
    return GC_->loVect();
}

/**
 *
 */
const int*  Tile::hiGCVect(void) const {
    return GC_->hiVect();
}

/**
 *
 */
const amrex::Box&  Tile::interiorAndGC(void) const {
    return (*GC_);
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

