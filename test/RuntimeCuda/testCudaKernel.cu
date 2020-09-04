#include "Tile.h"
#include "Grid.h"
#include "FArray4D.h"
#include "CudaRuntime.h"
#include "CudaStreamManager.h"

#include "Flash.h"
#include "constants.h"
#include "gpuKernel.h"

constexpr unsigned int   LEVEL = 0;
constexpr unsigned int   N_THREAD_TEAMS = 1;
constexpr unsigned int   MAX_THREADS = 5;

void setInitialConditions_block(const int tId, void* dataItem) {
    using namespace orchestration;

    Tile*  tileDesc = static_cast<Tile*>(dataItem);

    const IntVect   loGC = tileDesc->loGC();
    const IntVect   hiGC = tileDesc->hiGC();
    FArray4D        f    = tileDesc->data();

    for         (int k = loGC.K(); k <= hiGC.K(); ++k) {
        for     (int j = loGC.J(); j <= hiGC.J(); ++j) {
            for (int i = loGC.I(); i <= hiGC.I(); ++i) {
                f(i, j, k, DENS_VAR_C) = i;
                f(i, j, k, ENER_VAR_C) = 2.0 * j;
            }
        }
    }
}

int   main(int argc, char* argv[]) {
    using namespace orchestration;

    constexpr std::size_t   N_CELLS =   (NXB + 2 * NGUARD * K1D)
                                      * (NYB + 2 * NGUARD * K2D)
                                      * (NZB + 2 * NGUARD * K3D)
                                      * NUNKVAR;
    constexpr std::size_t   N_BYTES_PER_PACKET =   2 * MDIM * sizeof(int) 
                                                 +        1 * sizeof(FArray4D)
                                                 +  N_CELLS * sizeof(Real);
    constexpr std::size_t   N_BLOCKS = N_BLOCKS_X * N_BLOCKS_Y * N_BLOCKS_Z;

    CudaRuntime::setNumberThreadTeams(N_THREAD_TEAMS);
    CudaRuntime::setMaxThreadsPerTeam(MAX_THREADS);
    CudaRuntime::setLogFilename("DeleteMe.log");
    std::cout << "\n";
    std::cout << "----------------------------------------------------------\n";
    CudaRuntime::instance().printGpuInformation();
    std::cout << "----------------------------------------------------------\n";
    std::cout << std::endl;

    CudaStreamManager::setMaxNumberStreams(N_BLOCKS);
    CudaStream    streamFull;
    cudaStream_t  stream;
    int           streamId;

    // Initialize Grid unit/AMReX
    Grid::instantiate();
    Grid&    grid = Grid::instance();
    grid.initDomain(setInitialConditions_block);

    // TODO: Errors should try to release acquired resources if possible.

    // Created pinned and device memory pools
    void*         buffer_p = nullptr;
    cudaError_t   cErr = cudaMallocHost(&buffer_p, N_BYTES_PER_PACKET * N_BLOCKS);
    if (cErr != cudaSuccess) {
        std::string  errMsg = "[CudaDataPacket::prepareForTransfer] ";
        errMsg += "Unable to allocate pinned memory\n";
        errMsg += "Cuda error - " + std::string(cudaGetErrorName(cErr)) + "\n";
        errMsg += std::string(cudaGetErrorString(cErr)) + "\n";
        throw std::runtime_error(errMsg);
    }

    void*         buffer_d = nullptr;
    cErr = cudaMalloc(&buffer_d, N_BYTES_PER_PACKET * N_BLOCKS);
    if (cErr != cudaSuccess) {
        std::string  errMsg = "[CudaDataPacket::prepareForTransfer] ";
        errMsg += "Unable to allocate device memory\n";
        errMsg += "Cuda error - " + std::string(cudaGetErrorName(cErr)) + "\n";
        errMsg += std::string(cudaGetErrorString(cErr)) + "\n";
        throw std::runtime_error(errMsg);
    }

    // Run the kernel in the CPU at first
    std::size_t          n        = 0;
    Real*                data_h   = nullptr;
    Real*                data_p   = nullptr;
    Real*                data_d   = nullptr;
    void*                packet_p = nullptr;
    void*                packet_d = nullptr;
    char*                ptr_p    = static_cast<char*>(buffer_p);
    char*                ptr_d    = static_cast<char*>(buffer_d);
    std::vector<Real*>   hostPtrs{N_BLOCKS};
    std::vector<Real*>   pinnedPtrs{N_BLOCKS};
    assert(sizeof(char) == 1);
    for (auto ti = grid.buildTileIter(LEVEL); ti->isValid(); ti->next(), ++n) {
        std::unique_ptr<Tile>   tileDesc = ti->buildCurrentTile();

        const IntVect   loGC = tileDesc->loGC();
        const IntVect   hiGC = tileDesc->hiGC();
        data_h               = tileDesc->dataPtr();

        // Keep pointer to start of this packet
        packet_p = ptr_p;
        packet_d = ptr_d;

        // Pack data for single tile data packet
        int   tmp = loGC.I();
        std::memcpy((void*)ptr_p, (void*)&tmp, sizeof(int));
        ptr_p += sizeof(int);
        ptr_d += sizeof(int);
        tmp = loGC.J();
        std::memcpy((void*)ptr_p, (void*)&tmp, sizeof(int));
        ptr_p += sizeof(int);
        ptr_d += sizeof(int);
        tmp = loGC.K();
        std::memcpy((void*)ptr_p, (void*)&tmp, sizeof(int));
        ptr_p += sizeof(int);
        ptr_d += sizeof(int);

        tmp = hiGC.I();
        std::memcpy((void*)ptr_p, (void*)&tmp, sizeof(int));
        ptr_p += sizeof(int);
        ptr_d += sizeof(int);
        tmp = hiGC.J();
        std::memcpy((void*)ptr_p, (void*)&tmp, sizeof(int));
        ptr_p += sizeof(int);
        ptr_d += sizeof(int);
        tmp = hiGC.K();
        std::memcpy((void*)ptr_p, (void*)&tmp, sizeof(int));
        ptr_p += sizeof(int);
        ptr_d += sizeof(int);

        data_p = reinterpret_cast<Real*>(ptr_p);
        data_d = reinterpret_cast<Real*>(ptr_d);
        std::memcpy((void*)ptr_p, (void*)data_h, N_CELLS*sizeof(Real));
        ptr_p += N_CELLS*sizeof(Real);
        ptr_d += N_CELLS*sizeof(Real);

        // Create an FArray4D object in host memory but that already points
        // to where its data will be in device memory (i.e. the device object
        // will already be attached to its data in device memory).
        // The object in host memory should never be used then.
        // IMPORTANT: When this local object is destroyed, we don't want it to
        // affect the use of the copies (e.g. release memory).
        FArray4D   f_d{data_d, loGC, hiGC, NUNKVAR};
        std::memcpy((void*)ptr_p, (void*)&f_d, sizeof(FArray4D));
        ptr_p += sizeof(FArray4D);
        ptr_d += sizeof(FArray4D);

        // Cache pointers to data in same order for later copyback
        hostPtrs[n]   = data_h;
        pinnedPtrs[n] = data_p;

        // Do data transfers and execute kernel
        streamFull = CudaStreamManager::instance().requestStream(false);
        stream = *(streamFull.object);
        streamId = streamFull.id;

        cErr = cudaMemcpyAsync(packet_d, packet_p, N_BYTES_PER_PACKET,
                               cudaMemcpyHostToDevice, stream);
        if (cErr != cudaSuccess) {
            std::string  errMsg = "[CudaRuntime::executeGpuTasks] ";
            errMsg += "Unable to execute H-to-D transfer\n";
            errMsg += "CUDA error - " + std::string(cudaGetErrorName(cErr)) + "\n";
            errMsg += std::string(cudaGetErrorString(cErr)) + "\n";
            throw std::runtime_error(errMsg);
        }

        gpuKernel::kernel(packet_d, streamId);

        cErr = cudaMemcpyAsync(packet_p, packet_d, N_BYTES_PER_PACKET,
                               cudaMemcpyDeviceToHost, stream);
        if (cErr != cudaSuccess) {
            std::string  errMsg = "[CudaRuntime::executeGpuTasks] ";
            errMsg += "Unable to execute D-to-H transfer\n";
            errMsg += "CUDA error - " + std::string(cudaGetErrorName(cErr)) + "\n";
            errMsg += std::string(cudaGetErrorString(cErr)) + "\n";
            throw std::runtime_error(errMsg);
        }
        cudaStreamSynchronize(stream);
        CudaStreamManager::instance().releaseStream(streamFull);

        data_h = nullptr;
        data_p = nullptr;
        data_d = nullptr;
        packet_p = nullptr;
        packet_d = nullptr;
    }
    ptr_p = nullptr;
    ptr_d = nullptr;

    // Transfer data back to standard location in Grid host data structures
    // TODO: Could this be handled with the other asynchronous tasks using 
    // events or callbacks?
    for (n=0; n<hostPtrs.size(); ++n) {
        std::memcpy((void*)hostPtrs[n], (void*)pinnedPtrs[n], N_CELLS*sizeof(Real));
    }

    // Release buffers
    cErr = cudaFree(buffer_d);
    if (cErr != cudaSuccess) {
        std::string  errMsg = "[CudaDataPacket::prepareForTransfer] ";
        errMsg += "Unable to free device memory\n";
        errMsg += "Cuda error - " + std::string(cudaGetErrorName(cErr)) + "\n";
        errMsg += std::string(cudaGetErrorString(cErr)) + "\n";
        throw std::runtime_error(errMsg);
    }
    buffer_d = nullptr;

    cErr = cudaFreeHost(buffer_p);
    if (cErr != cudaSuccess) {
        std::string  errMsg = "[CudaDataPacket::prepareForTransfer] ";
        errMsg += "Unable to free pinned memory\n";
        errMsg += "Cuda error - " + std::string(cudaGetErrorName(cErr)) + "\n";
        errMsg += std::string(cudaGetErrorString(cErr)) + "\n";
        throw std::runtime_error(errMsg);
    }
    buffer_p = nullptr;

    // Check that kernel ran correctly    
    for (auto ti = grid.buildTileIter(LEVEL); ti->isValid(); ti->next()) {
        std::unique_ptr<Tile>   tileDesc = ti->buildCurrentTile();

        const IntVect   loGC = tileDesc->loGC();
        const IntVect   hiGC = tileDesc->hiGC();
        const FArray4D  f    = tileDesc->data();

        double absErr = 0.0;
        double fExpected = 0.0;
        for         (int k = loGC.K(); k <= hiGC.K(); ++k) {
            for     (int j = loGC.J(); j <= hiGC.J(); ++j) {
                for (int i = loGC.I(); i <= hiGC.I(); ++i) {
                    fExpected = 2.1;
                    absErr = fabs(f(i, j, k, DENS_VAR_C) - fExpected);
                    if (absErr > 1.0e-12) {
                        std::cout << "Bad DENS at ("
                                  << i << "," << j << "," << k << ") - "
                                  << f(i, j, k, DENS_VAR_C) << " instead of "
                                  << fExpected << "\n";
                    }
                    fExpected = 3.1;
                    absErr = fabs(f(i, j, k, ENER_VAR_C) - fExpected);
                    if (absErr > 1.0e-12) {
                        std::cout << "Bad ENER at ("
                                  << i << "," << j << "," << k << ") - "
                                  << f(i, j, k, ENER_VAR_C) << " instead of "
                                  << fExpected << "\n";
                    }
                }
            }
        }
    }

    // Clean-up
    Grid::instance().destroyDomain();
}

