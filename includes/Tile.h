#ifndef TILE_H__
#define TILE_H__

#include "DataItem.h"

#include "FArray4D.h"
#include "Grid_IntVect.h"

// TODO remove dependency on AMReX
#include <AMReX.H>

namespace orchestration {

/**
 * \brief Provides access to pointers to physical data.
 *
 * When iterating over the domain, a Tile Iterator returns
 * Tiles which store indices to the correct location in the
 * physical data arrays. Tile inherits from DataItem. Tile is
 * an abstract class, each AMR package must implement its own
 * version of most of the member functions.
 */
class Tile : public DataItem {
public:
    Tile(void);
    virtual ~Tile(void);

    Tile(Tile&&) = delete;
    Tile& operator=(Tile&&) = delete;
    Tile(Tile&) = delete;
    Tile(const Tile&) = delete;
    Tile& operator=(Tile&) = delete;
    Tile& operator=(const Tile&) = delete;

    // Overrides to DataItem
    void                       unpack(void) override;
    void*                      hostPointer(void) override;
    void*                      gpuPointer(void) override;
    std::size_t                sizeInBytes(void) override;
    std::shared_ptr<DataItem>  getTile(void) override;
    CudaStream&                stream(void) override;

    std::size_t  nSubItems(void) const override;
    std::shared_ptr<DataItem>  popSubItem(void) override;
    DataItem*    getSubItem(const std::size_t i) override;
    void         addSubItem(std::shared_ptr<DataItem>&& dataItem) override;

    // Pure virtual functions
    virtual bool         isNull(void) const = 0;
    virtual int          gridIndex(void) const = 0;
    virtual unsigned int level(void) const = 0;
    virtual IntVect      lo(void) const = 0;
    virtual IntVect      hi(void) const = 0;
    virtual IntVect      loGC(void) const = 0;
    virtual IntVect      hiGC(void) const = 0;
    // TODO: Create readonly versions of these?
    virtual FArray4D     data(void) = 0;
    virtual Real*        dataPtr(void) = 0;

    // Virtual functions with a default implementation.
    virtual RealVect     deltas(void) const;
    virtual RealVect     getCenterCoords(void) const;

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
};

}

#endif

