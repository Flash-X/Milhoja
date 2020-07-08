#include "CudaDataPacket.h"

#include <stdexcept>

#include <AMReX_Dim3.H>

#include "Flash.h"
#include "constants.h"
#include "CudaStreamManager.h"
#include "CudaGpuArray.h"

namespace orchestration {

/**
 * Create a CudaDataPacket object that null.
 */
CudaDataPacket::CudaDataPacket(void)
    : start_p_{nullptr},
      copyIn_p_{nullptr},
      start_d_{nullptr},
      copyIn_d_{nullptr},
      stream_{},
      nBytes_{0},
      idx_{NULL_DATA_PACKET_ID},
      nDataPerBlock_{0},
      tileList_{}
{
    assert(isNull());
}

/**
 *
 */
CudaDataPacket::~CudaDataPacket(void) {
    nullify();
}

/**
 *  Determine if the CudaDataPacket is null, which is true if
 *    - ...
 */
bool  CudaDataPacket::isNull(void) const {
    // TODO: Shouldn't CudaStream have an isNull member function?
    return (   (start_p_ == nullptr)
            && (copyIn_p_ == nullptr)
            && (start_d_ == nullptr)
            && (copyIn_d_ == nullptr)
            && (stream_.id == CudaStream::NULL_STREAM_ID)
            && (stream_.object == nullptr)
            && (nBytes_ == 0)
            && (idx_ == NULL_DATA_PACKET_ID)
            && (nDataPerBlock_ == 0)
            && (tileList_.size() == 0) );
}

/**
 *  Clean-up the CudaDataPacket's allocated resources (if any), and set the
 *  CudaDataPacket into the null state.
 *
 *  It is acceptable, but potentially wasteful to call this method for a
 *  CudaDataPacket that is already null.
 */
void CudaDataPacket::nullify(void) {
    if (stream_.object) {
        CudaStreamManager::instance().releaseStream(stream_);
        assert(stream_.object == nullptr);
    } else if (start_p_) {
        cudaError_t   cErr = cudaFreeHost(start_p_);
        if (cErr != cudaSuccess) {
            std::string  errMsg = "[CudaDataPacket::nullify] ";
            errMsg += "Unable to deallocate pinned memory\n";
            errMsg += "CUDA error - " + std::string(cudaGetErrorName(cErr)) + "\n";
            errMsg += std::string(cudaGetErrorString(cErr)) + "\n";
            throw std::runtime_error(errMsg);
        }
    } else if (start_d_) {
        cudaError_t   cErr = cudaFree(start_d_);
        if (cErr != cudaSuccess) {
            std::string  errMsg = "[CudaDataPacket::nullify] ";
            errMsg += "Unable to deallocate device memory\n";
            errMsg += "CUDA error - " + std::string(cudaGetErrorName(cErr)) + "\n";
            errMsg += std::string(cudaGetErrorString(cErr)) + "\n";
            throw std::runtime_error(errMsg);
        }
    }

    start_p_        = nullptr;
    copyIn_p_       = nullptr;
    start_d_        = nullptr;
    copyIn_d_       = nullptr;
    stream_.id      = CudaStream::NULL_STREAM_ID;
    stream_.object  = nullptr;
    nBytes_         = 0;
    idx_            = NULL_DATA_PACKET_ID;
    nDataPerBlock_  = 0;
    // TODO: What do we have to do to nullify the TileList?
    //       Ideally, we don't want to release pre-allocated memory if the
    //       container allows us to keep it around.
    tileList_.clear();

    assert(isNull());
}

/**
 *
 */
std::size_t  CudaDataPacket::nDataItems(void) const {
    return tileList_.size();
}

/**
 *  It is intended that the implementation of this method should not request any
 *  resources.  Rather, request allocation should occur with
 *  prepareForTransfer().  See the documentation for that method for more
 *  details.
 */
void   CudaDataPacket::addDataItem(Tile&& dataItem) {
    // If the memory buffers already exist, then they are not sized to accept
    // another data item
    if (start_p_ || start_d_) {
        throw std::logic_error("[CudaDataPacket::addDataItem (move)] "
                               "Cannot add data item after packet has been prepared");
    }

    tileList_.push_back( std::move(dataItem) );
}

/**
 *  It is intended that the implementation of this method should not request any
 *  resources.  Rather, request allocation should occur with
 *  prepareForTransfer().  See the documentation for that method for more
 *  details.
 */
void   CudaDataPacket::addDataItem(const Tile& dataItem) {
    // If the memory buffers already exist, then they are not sized to accept
    // another data item
    if (start_p_ || start_d_) {
        throw std::logic_error("[CudaDataPacket::addDataItem (copy)] "
                               "Cannot add data item after packet has been prepared");
    }

    tileList_.push_back( dataItem );
}

/**
 *  This method
 *    - acquires a stream to be used for scheduling data movements and
 *      kernel launches,
 *    - allocates pinned memory for staging the cell-centered data for
 *      asynchronous transfer to device memory,
 *    - packs the CC data from all tiles in the packet into the single,
 *      contiguous block of pinned memory, and
 *    - allocates device memory into which the CC data shall be transferred.
 *
 *  It is intended that client code call this method once it has added all tiles
 *  to the packet and just before transferring the packet.  One motivation for
 *  this is that we can isolate resource allocation to this method so that we
 *  only request resources when we know that they will be needed.  This avoids
 *  the scenario of eagerly acquiring resources up front and then having to
 *  remember to release them if the CudaDataPacket is never given tiles and
 *  therefore never used.
 *
 *  This method also exists so that client code can add in tiles one at a time
 *  without requiring that the data of the tiles be immediatly packed into the
 *  data packet.  Rather, we allow for intelligent packing here once the
 *  CudaDataPacket knows how many tiles it will need to transfer.
 */
void  CudaDataPacket::prepareForTransfer(const std::size_t nBytesIn) {
    if (tileList_.size() == 0) {
        throw std::logic_error("[CudaDataPacket::prepareForTransfer] "
                               "No sense in preparing an empty packet for transfer");
    } else if (stream_.object || (stream_.id != CudaStream::NULL_STREAM_ID)) {
        throw std::logic_error("[CudaDataPacket::prepareForTransfer] "
                               "CUDA stream already acquired");
    } else if (start_p_) {
        throw std::logic_error("[CudaDataPacket::prepareForTransfer] "
                               "Pinned memory buffer has already been allocated");
    } else if (start_d_) {
        throw std::logic_error("[CudaDataPacket::prepareForTransfer] "
                               "Device memory buffer has already been allocated");
    }

    // FIXME: Assume for the moment that cudaMalloc and cudaMallocHost allocate
    // memory that is properly aligned for double (or that we can eventually
    // accomplish this).  Then the copy-in buffer size should be a multiple of the
    // size of double so that the buffer after copy-in is correctly aligned.
    std::size_t  nBytesInAligned = nBytesIn;
    if ((nBytesIn % sizeof(double)) != 0) {
        nBytesInAligned =   sizeof(double)
                          * (floor(nBytesIn / ((float)sizeof(double))) + 1);
    }

    CudaStreamManager&   sm = CudaStreamManager::instance();
    stream_ = sm.requestStream(true);

    // FIXME: Presently written with the expectation that we will only ever use
    // packets of blocks rather than packets of proper tiles.
    std::size_t   nBlocks = tileList_.size();
    nDataPerBlock_ =   (NXB + 2 * NGUARD * K1D)
                     * (NYB + 2 * NGUARD * K2D)
                     * (NZB + 2 * NGUARD * K3D)
                     * NUNKVAR;
    std::size_t   blockSizeInBytes = nDataPerBlock_ * sizeof(double);
    std::size_t   pointSizeInBytes = sizeof(amrex::Dim3);
    std::size_t   arraySizeInBytes = sizeof(CudaGpuArray);

    // Pack the data into the data packet
    // The data packet is structured as follows:
    // 1) ...
    nBytes_ =   nBytesInAligned
              + nBlocks * (  N_CC_BUFFERS*blockSizeInBytes
                           +              arraySizeInBytes
                           +            2*pointSizeInBytes);

    // TODO: Are the returned buffers aligned in memory in accord with doubles?
    // TODO: We should have pools of preallocated pinned and device memory so
    // that we need not incur the overhead of repeated memory allocation.
    // TODO: What to do about amrex::Real within infrastructure and the static
    // Fortran code, which should just use something like double?
    cudaError_t    cErr = cudaErrorInvalidValue;
    cErr = cudaMallocHost(&start_p_, nBytes_);
    if (cErr != cudaSuccess) {
        std::string  errMsg = "[CudaDataPacket::prepareForTransfer] ";
        errMsg += "Unable to allocate pinned memory\n";
        errMsg += "Cuda error - " + std::string(cudaGetErrorName(cErr)) + "\n";
        errMsg += std::string(cudaGetErrorString(cErr)) + "\n";
        throw std::runtime_error(errMsg);
    }

    cErr = cudaMalloc(&start_d_, nBytes_);
    if (cErr != cudaSuccess) {
        std::string  errMsg = "[CudaDataPacket::prepareForTransfer] ";
        errMsg += "Unable to allocate device memory\n";
        errMsg += "Cuda error - " + std::string(cudaGetErrorName(cErr)) + "\n";
        errMsg += std::string(cudaGetErrorString(cErr)) + "\n";
        throw std::runtime_error(errMsg);
    }

    copyIn_p_ = static_cast<void*>(start_p_);
    copyIn_d_ = static_cast<void*>(start_d_);
    char*   pinnedBuffer = static_cast<char*>(copyIn_p_) + nBytesInAligned;
    char*   deviceBuffer = static_cast<char*>(copyIn_d_) + nBytesInAligned;
    assert(sizeof(char) == 1);
    for (Tile& dataItem : tileList_) {
        // TODO: Confirm that all pointers are null?

        // Copy lower GC point to pinned buffer and get pinned/device pointers
        amrex::Dim3   loGC_h = dataItem.loGC();
        memcpy((void *)pinnedBuffer,
               (void *)&loGC_h,
               pointSizeInBytes);
        dataItem.loGC_p_ = reinterpret_cast<amrex::Dim3*>(pinnedBuffer);
        pinnedBuffer += pointSizeInBytes;
        dataItem.loGC_d_ = reinterpret_cast<amrex::Dim3*>(deviceBuffer);
        deviceBuffer += pointSizeInBytes;

        // Copy upper GC point to pinned buffer and get pinned/device pointers
        amrex::Dim3   hiGC_h = dataItem.hiGC();
        memcpy((void *)pinnedBuffer,
               (void *)&hiGC_h,
               pointSizeInBytes);
        dataItem.hiGC_p_ = reinterpret_cast<amrex::Dim3*>(pinnedBuffer);
        pinnedBuffer += pointSizeInBytes;
        dataItem.hiGC_d_ = reinterpret_cast<amrex::Dim3*>(deviceBuffer);
        deviceBuffer += pointSizeInBytes;

        // Copy CC data from UNK (host) to CC1 data block in pinned buffer
        // and get pinned/device pointers
        // TODO: We need information to know which variables to copy over 
        //       so that we can have two actions working concurrently on
        //       disjoint sets of UNK variables.
        // TODO: Make casting of data type explicit here or add in a check at
        // startup that confirms that amrex::Real is the same type as our
        // double?
        memcpy((void *)pinnedBuffer,
               (void *)dataItem.CC_h_,
               blockSizeInBytes);
        dataItem.CC1_p_ = reinterpret_cast<double*>(pinnedBuffer);
        pinnedBuffer += blockSizeInBytes;
        dataItem.CC1_d_ = reinterpret_cast<double*>(deviceBuffer);
        deviceBuffer += blockSizeInBytes;

        // Create a Fortran-style array interface object on the host, but that
        // wraps the data in the device.
        const amrex::Dim3 begin_h = loGC_h;
        const amrex::Dim3 end_h{hiGC_h.x+1, hiGC_h.y+1, hiGC_h.z+1};
        CudaGpuArray    gpuArray_h(dataItem.CC1_d_, begin_h, end_h, NUNKVAR);
        memcpy((void *)pinnedBuffer,
               (void *)(&gpuArray_h),
               arraySizeInBytes);
        pinnedBuffer += arraySizeInBytes;
        // TODO: Hopefully this cast to void* will be temporary. See Tile.h for
        // more info.
        dataItem.CC1_array_d_ = static_cast<void*>(deviceBuffer);
        deviceBuffer += arraySizeInBytes;

        // TODO: memset CC2 to zero?  Do we need a flag in the CudaDataPacket 
        //       to specify which has the valid data?
        dataItem.CC2_p_ = reinterpret_cast<double*>(pinnedBuffer);
        pinnedBuffer += blockSizeInBytes;
        dataItem.CC2_d_ = reinterpret_cast<double*>(deviceBuffer);
        deviceBuffer += blockSizeInBytes;
    }
}

/**
 * This method is a helper function that is intended to be used to transfer
 * results from the pinned memory to the original source location in host memory
 * once a DataPacket has been transferred back to the host memory and will
 * subsequently be destroyed.  Therefore, this routine should not be called if
 * the packet is empty or hasn't yet been prepared for transfer.
 *
 */
void  CudaDataPacket::moveDataFromPinnedToSource(void) {
    // TODO: This needs to be updated/evaluated completely.
    if (tileList_.size() == 0) {
        throw std::logic_error("[CudaDataPacket::moveDataFromPinnedToSource] "
                               "No data to move for empty packet");
    } else if (nBytes_ == 0) {
        throw std::logic_error("[CudaDataPacket::moveDataFromPinnedToSource] "
                               "No bytes in packet");
    }

    std::size_t   nBlocks = tileList_.size();
    nDataPerBlock_ =   (NXB + 2 * NGUARD * K1D)
                     * (NYB + 2 * NGUARD * K2D)
                     * (NZB + 2 * NGUARD * K3D)
                     * NUNKVAR;
    size_t nCells = nBlocks * nDataPerBlock_;

    for (Tile& dataItem : tileList_) {
        if (!dataItem.CC1_p_) {
            throw std::logic_error("[CudaDataPacket::moveDataFromPinnedToSource] "
                                   "No address in pinned memory for CC1 data");
        } else if (!dataItem.CC_h_) {
            throw std::logic_error("[CudaDataPacket::moveDataFromPinnedToSource] "
                                   "No address in source memory given for CC data");
        }

        // TODO: We need a switch to know in which CCX to find the data
        // TODO: We need information to know which variable in UNK to write 
        //       this so that we can have two actions working concurrently 
        //       on disjoint sets of UNK variables.
        // TODO: Make casting of data type explicit here or add in a check at
        // startup that confirms that amrex::Real is the same type as our
        // double?  Make a custom type and for AMReX set type equal to
        // amrex::Real?
        memcpy((void *)dataItem.CC_h_,
               (void *)dataItem.CC1_p_,
               nCells * sizeof(double));
    }
}

/**
 *
 */
Tile&       CudaDataPacket::operator[](const std::size_t idx) {
    // Insist on array bounds checking
    return tileList_.at(idx);
}

/**
 *
 */
const Tile& CudaDataPacket::operator[](const std::size_t idx) const {
    // Insist on array bounds checking
    return tileList_.at(idx);
}

}
