#ifndef TILE_H__
#define TILE_H__

#include <AMReX.H>
#include <AMReX_Box.H>
#include <AMReX_MFIter.H>

/**
 * TODO: The construction of this class should be well done.  In particular, we
 * want to make sure that we limit creating copies as much as possible in terms
 * of data members here and it terms of granting client code to Grid data -
 * prefer references as much as possible.  Also, the tile objects are added to
 * tile lists (in data packets) and passed through queues.  Therefore, it would
 * be good to maximize use of move semantics where possible.
 */
class Tile {
public:
    Tile(amrex::MFIter& itor, const unsigned int level);
    ~Tile(void);

    Tile(const Tile&);
    Tile(Tile&&);
    Tile& operator=(Tile&&);

    int                  gridIndex(void) const  { return gridIdx_; }
    unsigned int         level(void) const      { return level_; }

    amrex::Dim3          lo(void) const;
    amrex::Dim3          hi(void) const;
    const int*           loVect(void) const;
    const int*           hiVect(void) const;
    const amrex::Box&    interior(void) const;

    amrex::Dim3          loGC(void) const;
    amrex::Dim3          hiGC(void) const;
    const int*           loGCVect(void) const;
    const int*           hiGCVect(void) const;
    const amrex::Box&    interiorAndGC(void) const;

    amrex::XDim3         deltas(void) const;

private:
    Tile(Tile&) = delete;
    Tile& operator=(Tile&) = delete;
    Tile& operator=(const Tile&) = delete;
    Tile& operator=(const Tile&&) = delete;

    int           gridIdx_;
    unsigned int  level_;
    amrex::Box*   interior_;
    amrex::Box*   GC_;
};

#endif

