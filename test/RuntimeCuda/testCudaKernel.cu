#include "Tile.h"
#include "Grid.h"
#include "FArray4D.h"
#include "CudaRuntime.h"

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

    CudaRuntime::setNumberThreadTeams(N_THREAD_TEAMS);
    CudaRuntime::setMaxThreadsPerTeam(MAX_THREADS);
    CudaRuntime::setLogFilename("DeleteMe.log");
    std::cout << "\n";
    std::cout << "----------------------------------------------------------\n";
    CudaRuntime::instance().printGpuInformation();
    std::cout << "----------------------------------------------------------\n";
    std::cout << std::endl;

    // Initialize Grid unit/AMReX
    Grid::instantiate();
    Grid&    grid = Grid::instance();
    grid.initDomain(setInitialConditions_block);

    // Determine size of and allocate memory for data packet
    constexpr std::size_t   N_CELLS =   (NXB + 2 * NGUARD * K1D)
                                      * (NYB + 2 * NGUARD * K2D)
                                      * (NZB + 2 * NGUARD * K3D)
                                      * NUNKVAR;
    constexpr std::size_t   N_BYTES =   2 * MDIM * sizeof(int) 
                                      +  N_CELLS * sizeof(Real);

    void*         packet_p = nullptr;
    cudaError_t   cErr = cudaMallocHost(&packet_p, N_BYTES);
    if (cErr != cudaSuccess) {
        std::string  errMsg = "[CudaDataPacket::prepareForTransfer] ";
        errMsg += "Unable to allocate pinned memory\n";
        errMsg += "Cuda error - " + std::string(cudaGetErrorName(cErr)) + "\n";
        errMsg += std::string(cudaGetErrorString(cErr)) + "\n";
        throw std::runtime_error(errMsg);
    }

    void*         packet_d = nullptr;
    cErr = cudaMalloc(&packet_d, N_BYTES);
    if (cErr != cudaSuccess) {
        std::string  errMsg = "[CudaDataPacket::prepareForTransfer] ";
        errMsg += "Unable to allocate device memory\n";
        errMsg += "Cuda error - " + std::string(cudaGetErrorName(cErr)) + "\n";
        errMsg += std::string(cudaGetErrorString(cErr)) + "\n";
        throw std::runtime_error(errMsg);
    }

    // Run the kernel in the CPU at first
    Real*   data_h = nullptr;
    Real*   data_p = nullptr;
    Real*   data_d = nullptr;
    char*   ptr_p  = nullptr;
    char*   ptr_d  = nullptr;
    assert(sizeof(char) == 1);
    for (auto ti = grid.buildTileIter(LEVEL); ti->isValid(); ti->next()) {
        std::unique_ptr<Tile>   tileDesc = ti->buildCurrentTile();

        const IntVect   loGC = tileDesc->loGC();
        const IntVect   hiGC = tileDesc->hiGC();
        data_h               = tileDesc->dataPtr();

        // Marshall data for single tile data packet
        ptr_p = static_cast<char*>(packet_p);
        ptr_d = static_cast<char*>(packet_d);

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

        cErr = cudaMemcpy(packet_d, packet_p, N_BYTES, cudaMemcpyHostToDevice);
        if (cErr != cudaSuccess) {
            std::string  errMsg = "[CudaRuntime::executeGpuTasks] ";
            errMsg += "Unable to execute H-to-D transfer\n";
            errMsg += "CUDA error - " + std::string(cudaGetErrorName(cErr)) + "\n";
            errMsg += std::string(cudaGetErrorString(cErr)) + "\n";
            throw std::runtime_error(errMsg);
        }

        gpuKernel::kernel(packet_d);

        cErr = cudaMemcpy(packet_p, packet_d, N_BYTES, cudaMemcpyDeviceToHost);
        if (cErr != cudaSuccess) {
            std::string  errMsg = "[CudaRuntime::executeGpuTasks] ";
            errMsg += "Unable to execute D-to-H transfer\n";
            errMsg += "CUDA error - " + std::string(cudaGetErrorName(cErr)) + "\n";
            errMsg += std::string(cudaGetErrorString(cErr)) + "\n";
            throw std::runtime_error(errMsg);
        }

        std::memcpy((void*)data_h, (void*)data_p, N_CELLS*sizeof(Real));

        data_h = nullptr;
        data_p = nullptr;
        data_d = nullptr;
        ptr_p  = nullptr;
        ptr_d  = nullptr;
    }

    // Release buffers
    cErr = cudaFree(packet_d);
    if (cErr != cudaSuccess) {
        std::string  errMsg = "[CudaDataPacket::prepareForTransfer] ";
        errMsg += "Unable to free device memory\n";
        errMsg += "Cuda error - " + std::string(cudaGetErrorName(cErr)) + "\n";
        errMsg += std::string(cudaGetErrorString(cErr)) + "\n";
        throw std::runtime_error(errMsg);
    }
    packet_d = nullptr;

    cErr = cudaFreeHost(packet_p);
    if (cErr != cudaSuccess) {
        std::string  errMsg = "[CudaDataPacket::prepareForTransfer] ";
        errMsg += "Unable to free pinned memory\n";
        errMsg += "Cuda error - " + std::string(cudaGetErrorName(cErr)) + "\n";
        errMsg += std::string(cudaGetErrorString(cErr)) + "\n";
        throw std::runtime_error(errMsg);
    }
    packet_p = nullptr;

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
                    fExpected = i + 2.1;
                    absErr = fabs(f(i, j, k, DENS_VAR_C) - fExpected);
                    if (absErr > 1.0e-12) {
                        std::cout << "Bad DENS at ("
                                  << i << "," << j << "," << k << ") - "
                                  << f(i, j, k, DENS_VAR_C) << " instead of "
                                  << fExpected << "\n";
                    }
                    fExpected = 2.0*j + 2.1;
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

