#ifndef TILE_H__
#define TILE_H__

#include <AMReX.H>
#include <AMReX_Box.H>
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
    Tile(amrex::MFIter& itor);
    ~Tile(void);

    Tile(const Tile&);
    Tile(Tile&&);
    Tile& operator=(Tile&&);

    int                          gridIndex(void) const  { return gridIdx_; }

    amrex::Dim3                  lo(void) const;
    amrex::Dim3                  hi(void) const;
    const int*                   loVect(void) const;
    const int*                   hiVect(void) const;
    const amrex::Box&            interior(void) const;

    amrex::Dim3                  loGC(void) const;
    amrex::Dim3                  hiGC(void) const;
    const int*                   loGCVect(void) const;
    const int*                   hiGCVect(void) const;
    const amrex::Box&            interiorAndGC(void) const;

    amrex::XDim3                 deltas(void) const;
//    amrex::Array4<amrex::Real>&  data(void)     { return data_; }


private:
    Tile(Tile&) = delete;
    Tile& operator=(Tile&) = delete;
    Tile& operator=(const Tile&) = delete;
    Tile& operator=(const Tile&&) = delete;

    int           gridIdx_;
    amrex::Box*   interior_;
    amrex::Box*   GC_;
//    amrex::Array4<amrex::Real>   data_;
};

#endif

