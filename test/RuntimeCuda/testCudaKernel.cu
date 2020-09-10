#include "Tile.h"
#include "Grid.h"
#include "FArray4D.h"
#include "CudaRuntime.h"
#include "CudaStreamManager.h"
#include "CudaDataPacket.h"
#include "CudaMoverUnpacker.h"

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

    // Initialize Grid unit/AMReX
    Grid::instantiate();
    Grid&    grid = Grid::instance();
    grid.initDomain(setInitialConditions_block);

    CudaMoverUnpacker                 unpacker{};
    std::shared_ptr<CudaDataPacket>   dataItem_gpu{};
    assert(dataItem_gpu == nullptr);
    assert(dataItem_gpu.use_count() == 0);

    // TODO: Errors should try to release acquired resources if possible.
    cudaStream_t  stream;
    int           streamId = 0;
    cudaError_t   cErr = cudaSuccess;
    for (auto ti = grid.buildTileIter(LEVEL); ti->isValid(); ti->next()) {
        dataItem_gpu = std::make_shared<CudaDataPacket>( ti->buildCurrentTile() );

        dataItem_gpu->pack();

        CudaStream&  streamFull = dataItem_gpu->stream();
        stream = *(streamFull.object);
        streamId = streamFull.id;

        cErr = cudaMemcpyAsync(dataItem_gpu->gpuPointer(), dataItem_gpu->hostPointer(),
                               dataItem_gpu->sizeInBytes(),
                               cudaMemcpyHostToDevice, stream);
        if (cErr != cudaSuccess) {
            std::string  errMsg = "[CudaRuntime::executeGpuTasks] ";
            errMsg += "Unable to execute H-to-D transfer\n";
            errMsg += "CUDA error - " + std::string(cudaGetErrorName(cErr)) + "\n";
            errMsg += std::string(cudaGetErrorString(cErr)) + "\n";
            throw std::runtime_error(errMsg);
        }

        gpuKernel::kernel(dataItem_gpu->gpuPointer(), streamId);

        unpacker.enqueue( std::move(dataItem_gpu) );
        assert(dataItem_gpu == nullptr);
        assert(dataItem_gpu.use_count() == 0);
    }

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
                    fExpected = i + 2.1*j; 
                    absErr = fabs(f(i, j, k, DENS_VAR_C) - fExpected);
                    if (absErr > 1.0e-12) {
                        std::cout << "Bad DENS at ("
                                  << i << "," << j << "," << k << ") - "
                                  << f(i, j, k, DENS_VAR_C) << " instead of "
                                  << fExpected << "\n";
                    }
                    fExpected = -i + 2.0*j;
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

