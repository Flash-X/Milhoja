#ifndef TILE_H__
#define TILE_H__

#include <AMReX.H>
#include <AMReX_Box.H>
#include <AMReX_MFIter.H>
#include "Grid_IntVect.h"

#include "DataItem.h"

namespace orchestration {

/**
 * TODO: The construction of this class should be well done.  In particular, we
 * want to make sure that we limit creating copies as much as possible in terms
 * of data members here and it terms of granting client code to Grid data -
 * prefer references as much as possible.  Also, the tile objects are added to
 * tile lists (in data packets) and passed through queues.  Therefore, it would
 * be good to maximize use of move semantics where possible.
 */
class Tile : public DataItem {
public:
    Tile(amrex::MFIter& itor, const unsigned int level);
    ~Tile(void);

    Tile(Tile&&);
    Tile& operator=(Tile&&);

    bool                 isNull(void) const;

    int                  gridIndex(void) const  { return gridIdx_; }
    unsigned int         level(void) const      { return level_; }

    // TODO: Get rid of the amrex::Dim3/XDim3 so that the interface is more
    // general.  Can we get references instead?
    IntVect              lo(void) const;
    IntVect              hi(void) const;

    amrex::Dim3          loGC(void) const;
    amrex::Dim3          hiGC(void) const;
    const int*           loGCVect(void) const;
    const int*           hiGCVect(void) const;

    // TODO: Do we need these in the interface?  I'd like to avoid having
    //       the AMReX data structures in the interface.
    //       If needed, can they be made protected or private?
    const amrex::Box&    interior(void) const;
    const amrex::Box&    interiorAndGC(void) const;

    amrex::XDim3         deltas(void) const;

    std::size_t                nSubItems(void) const override;
    std::shared_ptr<DataItem>  popSubItem(void) override;
    DataItem*                  getSubItem(const std::size_t i) override;
    void                       addSubItem(std::shared_ptr<DataItem>&& dataItem) override;

    // Pointers to source data in the original data structures in the host
    // memory
    amrex::Real*    CC_h_;

    // TODO: Replace amrex::Real and double with Real ??

    // If a Tile object has been added to a DataPacket, then these pointers will
    // point to the location of useful data in the DataPacket's pinned memory
    // buffer in the host's memory.
    double*         CC1_p_; 
    double*         CC2_p_; 
    amrex::Dim3*    loGC_p_;
    amrex::Dim3*    hiGC_p_;

    // If a Tile object has been added to a DataPacket, then these pointers will
    // point to the location of useful data in the DataPacket's device memory
    // buffer in the GPU's memory.
    double*         CC1_d_; 
    double*         CC2_d_; 
    amrex::Dim3*    loGC_d_;
    amrex::Dim3*    hiGC_d_;
    // FIXME: I would like to specify the type of the following pointer as
    // CudaGpuArray.  However, this would then mean the this class would be
    // CudaTile. It also means that every code that includes Tile.h needs to be
    // compiled with CUDA since CudaGpuArray uses macros like __device__.  As an
    // ugly hack to just get things working, I use void*.
    void*           CC1_array_d_;

private:
    // Limit all copies as much as possible
    Tile(Tile&) = delete;
    Tile(const Tile&) = delete;
    Tile& operator=(Tile&) = delete;
    Tile& operator=(const Tile&) = delete;

    int           gridIdx_;
    unsigned int  level_;
    amrex::Box*   interior_;
    amrex::Box*   GC_;
};

}

#endif

