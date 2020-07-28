#ifndef TILEAMREX_H__
#define TILEAMREX_H__

#include <AMReX.H>
#include <AMReX_Box.H>
#include <AMReX_MFIter.H>
#include "Grid_IntVect.h"

#include "Tile.h"

namespace orchestration {

/**
  * Derived class from Tile.
  */
class TileAmrex : public Tile {
public:
    TileAmrex(void);
    TileAmrex(amrex::MFIter& itor, const unsigned int level);
    ~TileAmrex(void);

    TileAmrex(const TileAmrex&);
    TileAmrex(TileAmrex&&);
    TileAmrex& operator=(TileAmrex&&);

    bool             isNull(void) const override;

    IntVect          lo(void) const override;
    IntVect          hi(void) const override;

    IntVect          loGC(void) const override;
    IntVect          hiGC(void) const override;

    // Pointers to source data in the original data structures in the host
    // memory
    //amrex::Real*    CC_h_;

    // TODO: Replace amrex::Real and double with Real ??

    // If a Tile object has been added to a DataPacket, then these pointers will
    // point to the location of useful data in the DataPacket's pinned memory
    // buffer in the host's memory.
    //double*         CC1_p_; 
    //double*         CC2_p_; 
    //amrex::Dim3*    loGC_p_;
    //amrex::Dim3*    hiGC_p_;

    // If a Tile object has been added to a DataPacket, then these pointers will
    // point to the location of useful data in the DataPacket's device memory
    // buffer in the GPU's memory.
    //double*         CC1_d_; 
    //double*         CC2_d_; 
    //amrex::Dim3*    loGC_d_;
    //amrex::Dim3*    hiGC_d_;
    // FIXME: I would like to specify the type of the following pointer as
    // CudaGpuArray.  However, this would then mean the this class would be
    // CudaTile. It also means that every code that includes Tile.h needs to be
    // compiled with CUDA since CudaGpuArray uses macros like __device__.  As an
    // ugly hack to just get things working, I use void*.
    //void*           CC1_array_d_;

private:
    // Limit all copies as much as possible
    TileAmrex(TileAmrex&) = delete;
    TileAmrex& operator=(TileAmrex&) = delete;
    TileAmrex& operator=(const TileAmrex&) = delete;

    //amrex::Box*   interior_;
    //amrex::Box*   GC_;
};

}

#endif

