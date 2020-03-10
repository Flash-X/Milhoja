#ifndef TILE_H__
#define TILE_H__

#include <AMReX.H>
#include <AMReX_Box.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_MFIter.H>

/**
 * TODO: We use the MFIter to gain access to information about each tile in an
 * AMReX data structure.  For the moment, I am just copying those results into
 * objects instantiated from this class.  If we can assume that the AMReX data
 * structures do not change between acquiring the AMReX information with the
 * MFIter and when the tile information consumed, then we could potentiallystore
 * references here.
 */
class Tile {
public:
    Tile(amrex::MFIter& itor,
         amrex::MultiFab& mfab,
         const amrex::Geometry& geometry);
    ~Tile(void);

    amrex::Dim3                  lo(void) const;
    amrex::Dim3                  hi(void) const;
    const amrex::Box&            interior(void) const;

    amrex::Dim3                  loGC(void) const;
    amrex::Dim3                  hiGC(void) const;
    const amrex::Box&            interiorAndGC(void) const;

    amrex::XDim3                 deltas(void) const;
    amrex::Array4<amrex::Real>&  data(void);

private:
    const amrex::Box             interior_;
    const amrex::Box             GC_;
    amrex::Array4<amrex::Real>   data_;
    const amrex::Geometry&       geometry_;
};

#endif

