#include "CudaRuntime.h"

#include <cmath>
#include <stdexcept>
#include <iostream>
#include <iomanip>

#include "ThreadTeam.h"
#include "Grid.h"
#include "OrchestrationLogger.h"
#include "CudaStreamManager.h"
#include "CudaDataPacket.h"
#include "CudaMoverUnpacker.h"

#include "Flash.h"

namespace orchestration {

unsigned int    CudaRuntime::nTeams_            = 0;
unsigned int    CudaRuntime::maxThreadsPerTeam_ = 0;
bool            CudaRuntime::instantiated_      = false;

/**
 * 
 *
 * \return 
 */
CudaRuntime& CudaRuntime::instance(void) {
    static CudaRuntime     orSingleton;
    return orSingleton;
}

/**
 * 
 *
 * \return 
 */
void CudaRuntime::setLogFilename(const std::string& filename) {
    orchestration::Logger::setLogFilename(filename);
}

/**
 * 
 *
 * \return 
 */
void CudaRuntime::setNumberThreadTeams(const unsigned int nTeams) {
    if (instantiated_) {
        throw std::logic_error("[CudaRuntime::setNumberThreadTeams] "
                               "Set only when runtime does not exist");
    } else if (nTeams == 0) {
        throw std::invalid_argument("[CudaRuntime::setNumberThreadTeams] "
                                    "Need at least one ThreadTeam");
    }

    nTeams_ = nTeams;
}

/**
 * 
 *
 * \return 
 */
void CudaRuntime::setMaxThreadsPerTeam(const unsigned int nThreads) {
    if (instantiated_) {
        throw std::logic_error("[CudaRuntime::setMaxThreadsPerTeam] "
                               "Set only when runtime does not exist");
    } else if (nThreads == 0) {
        throw std::invalid_argument("[CudaRuntime::setMaxThreadsPerTeam] "
                                    "Need at least one thread per team");
    }

    maxThreadsPerTeam_ = nThreads;
}

/**
 * 
 *
 * \return 
 */
CudaRuntime::CudaRuntime(void)
    : teams_{nullptr},
      gpuDeviceName_{""},
      gpuCompMajor_{-1},
      gpuCompMinor_{-1},
      gpuMaxGridSize_{-1, -1, -1},
      gpuMaxThreadDim_{-1, -1, -1},
      gpuMaxThreadsPerBlock_{-1},
      gpuWarpSize_{-1},
      gpuClockRateHz_{-1.0},
      gpuMemClockRateHz_{-1.0},
      gpuMemBusWidthBytes_{-1},
      gpuTotalGlobalMemBytes_{0},
      gpuL2CacheSizeBytes_{-1},
      gpuSupportsL1Caching_{false},
      gpuNumMultiprocessors_{-1},
      gpuMaxConcurrentKernels_{-1}
{
    Logger::instance().log("[CudaRuntime] Initializing...");

    if (nTeams_ <= 0) {
        throw std::invalid_argument("[CudaRuntime::CudaRuntime] "
                                    "Need to create at least one team");
    }

    teams_ = new ThreadTeam*[nTeams_];
    for (unsigned int i=0; i<nTeams_; ++i) {
        teams_[i] = new ThreadTeam(maxThreadsPerTeam_, i);
    }

    int numDevices = 0;
    cudaGetDeviceCount(&numDevices);
    if (numDevices != 1) {
        throw std::runtime_error("[CudaRuntime::CudaRuntime] "
                                 "We insist upon 1 GPU per MPI task");
    }

    cudaDeviceProp  prop;
    cudaGetDeviceProperties(&prop, 0);

    if (prop.concurrentKernels != 1) {
        throw std::runtime_error("[CudaRuntime::CudaRuntime] "
                                 "GPU kernel concurrency is required");
    }

    gpuDeviceName_          = std::string(prop.name);
    gpuCompMajor_           = prop.major;
    gpuCompMinor_           = prop.minor;
    gpuMaxThreadsPerBlock_  = prop.maxThreadsPerBlock;
    gpuWarpSize_            = prop.warpSize;
    gpuClockRateHz_         = prop.clockRate * 1000;
    gpuMemClockRateHz_      = prop.memoryClockRate * 1000;
    gpuMemBusWidthBytes_    = round(prop.memoryBusWidth * 0.125);
    gpuTotalGlobalMemBytes_ = prop.totalGlobalMem;
    gpuL2CacheSizeBytes_    = prop.l2CacheSize;
    gpuSupportsL1Caching_   = (prop.localL1CacheSupported == 1);
    gpuNumMultiprocessors_  = prop.multiProcessorCount;
    for (unsigned int i=0; i<3; ++i) {
        gpuMaxGridSize_[i]  = prop.maxGridSize[i];
        gpuMaxThreadDim_[i] = prop.maxThreadsDim[i];
    }

    if (gpuCompMajor_ != 7) {
        throw std::runtime_error("[CudaRuntime::CudaRuntime] "
                                 "We assume GPU compute capability 7.X");
    }

    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications__technical-specifications-per-compute-capability
    gpuMaxConcurrentKernels_ = 128;

    CudaStreamManager::setMaxNumberStreams(gpuMaxConcurrentKernels_);

    instantiated_ = true;

    Logger::instance().log("[CudaRuntime] Created and ready for use");
}

/**
 * 
 *
 * \return 
 */
CudaRuntime::~CudaRuntime(void) {
    Logger::instance().log("[CudaRuntime] Finalizing...");

    instantiated_ = false;

    for (unsigned int i=0; i<nTeams_; ++i) {
        delete teams_[i];
        teams_[i] = nullptr;
    }
    delete [] teams_;
    teams_ = nullptr;

    Logger::instance().log("[CudaRuntime] Finalized");
}

unsigned int CudaRuntime::numberFreeStreams(void) const {
    return CudaStreamManager::instance().numberFreeStreams();
}

/**
 * 
 *
 * \return 
 */
void CudaRuntime::executeCpuTasks(const std::string& actionName,
                                  const RuntimeAction& cpuAction) {
    if (cpuAction.teamType != ThreadTeamDataType::BLOCK) {
        throw std::logic_error("[CudaRuntime::executeCpuTasks] "
                               "Given CPU action should run on block-based "
                               "thread team, which is not in configuration");
    } else if (cpuAction.nTilesPerPacket != 0) {
        throw std::invalid_argument("[CudaRuntime::executeCpuTasks] "
                                    "CPU tiles/packet should be zero since it is tile-based");
    } else if (nTeams_ < 1) {
        throw std::logic_error("[CudaRuntime::executeCpuTasks] "
                               "Need at least one ThreadTeam in runtime");
    }

    ThreadTeam*   cpuTeam = teams_[0];

    Logger::instance().log("[CudaRuntime] Start single CPU action");

    cpuTeam->startCycle(cpuAction, "CPU_Block_Team");

    unsigned int   level = 0;
    Grid&   grid = Grid::instance();
    for (auto ti = grid.buildTileIter(level); ti->isValid(); ti->next()) {
        cpuTeam->enqueue( ti->buildCurrentTile() );
    }
    cpuTeam->closeQueue();
    cpuTeam->wait();

    Logger::instance().log("[CudaRuntime] End single CPU action");
}

/**
 * 
 *
 * \return 
 */
void CudaRuntime::executeGpuTasks(const std::string& bundleName,
                                  const RuntimeAction& gpuAction) {
    if (gpuAction.teamType != ThreadTeamDataType::SET_OF_BLOCKS) {
        throw std::logic_error("[CudaRuntime::executeGpuTasks] "
                               "Given GPU action should run on a thread team "
                               "that works with data packets of blocks");
    } else if (gpuAction.nTilesPerPacket <= 0) {
        throw std::invalid_argument("[CudaRuntime::executeGpuTasks] "
                                    "Need at least one tile per packet");
    } else if (nTeams_ < 1) {
        throw std::logic_error("[CudaRuntime::executeGpuTasks] "
                               "Need at least one ThreadTeam in runtime");
    }

    ThreadTeam*   gpuTeam = teams_[0];

    Logger::instance().log("[CudaRuntime] Start single GPU action");

    CudaMoverUnpacker         gpuToHost{};
    gpuTeam->attachDataReceiver(&gpuToHost);

    // TODO: Good idea to call this as early as possible so that all threads
    //       go to wait while this this is setting everything up?  Hides
    //       TT wind-up cost.
    gpuTeam->startCycle(gpuAction, "GPU_PacketOfBlocks_Team");

    // We allocate the CUD memory here and pass the pointers around
    // with the data packet so that a work subscriber will
    // eventually deallocate the memory.
    auto packet_gpu = std::shared_ptr<CudaDataPacket>{};
    assert(packet_gpu == nullptr);
    assert(packet_gpu.use_count() == 0);

    cudaStream_t   stream;
    cudaError_t    cErr = cudaErrorInvalidValue;
    unsigned int   level = 0;
    Grid&   grid = Grid::instance();
    for (auto ti = grid.buildTileIter(level); ti->isValid(); ti->next()) {
        packet_gpu = std::make_shared<CudaDataPacket>( ti->buildCurrentTile() );
        packet_gpu->pack();

        stream = *(packet_gpu->stream().object);
        cErr = cudaMemcpyAsync(packet_gpu->gpuPointer(), packet_gpu->hostPointer(),
                               packet_gpu->sizeInBytes(),
                               cudaMemcpyHostToDevice, stream);
        if (cErr != cudaSuccess) {
            std::string  errMsg = "[CudaRuntime::executeGpuTasks] ";
            errMsg += "Unable to execute H-to-D transfer\n";
            errMsg += "CUDA error - " + std::string(cudaGetErrorName(cErr)) + "\n";
            errMsg += std::string(cudaGetErrorString(cErr)) + "\n";
            throw std::runtime_error(errMsg);
        }

        gpuTeam->enqueue( std::move(packet_gpu) );
        assert(packet_gpu == nullptr);
        assert(packet_gpu.use_count() == 0);
    }

    gpuTeam->closeQueue();
    gpuTeam->wait();

    gpuTeam->detachDataReceiver();

    Logger::instance().log("[CudaRuntime] End single GPU action");
}

/**
 * 
 *
 * \return 
 */
void CudaRuntime::executeTasks_FullPacket(const std::string& bundleName,
                                          const RuntimeAction& cpuAction,
                                          const RuntimeAction& gpuAction,
                                          const RuntimeAction& postGpuAction) {
    if        (cpuAction.teamType != ThreadTeamDataType::BLOCK) {
        throw std::logic_error("[CudaRuntime::executeTasks_FullPacket] "
                               "Given CPU action should run on tile-based "
                               "thread team, which is not in configuration");
    } else if (cpuAction.nTilesPerPacket != 0) {
        throw std::invalid_argument("[CudaRuntime::executeTasks_FullPacket] "
                                    "CPU tiles/packet should be zero since it is tile-based");
    } else if (gpuAction.teamType != ThreadTeamDataType::SET_OF_BLOCKS) {
        throw std::logic_error("[CudaRuntime::executeTasks_FullPacket] "
                               "Given GPU action should run on packet-based "
                               "thread team, which is not in configuration");
    } else if (gpuAction.nTilesPerPacket <= 0) {
        throw std::invalid_argument("[CudaRuntime::executeTasks_FullPacket] "
                                    "Need at least one tile per GPU packet");
    } else if (postGpuAction.teamType != ThreadTeamDataType::BLOCK) {
        throw std::logic_error("[CudaRuntime::executeTasks_FullPacket] "
                               "Given post-GPU action should run on tile-based "
                               "thread team, which is not in configuration");
    } else if (postGpuAction.nTilesPerPacket != 0) {
        throw std::invalid_argument("[CudaRuntime::executeTasks_FullPacket] "
                                    "Post-GPU should have zero tiles/packet as "
                                    "client code cannot control this");
    } else if (nTeams_ < 3) {
        throw std::logic_error("[CudaRuntime::executeTasks_FullPacket] "
                               "Need at least three ThreadTeams in runtime");
    }

    ThreadTeam*        cpuTeam     = teams_[0];
    ThreadTeam*        gpuTeam     = teams_[1];
    ThreadTeam*        postGpuTeam = teams_[2];
    CudaMoverUnpacker  gpuToHost{};

    unsigned int nTotalThreads =       cpuAction.nInitialThreads
                                 +     gpuAction.nInitialThreads
                                 + postGpuAction.nInitialThreads;
    if (nTotalThreads > postGpuTeam->nMaximumThreads()) {
        throw std::logic_error("[CudaRuntime::executeTasks_FullPacket] "
                                "Post-GPU could receive too many thread "
                                "activation calls from CPU and GPU teams");
    }

    //***** Construct thread and work pipelines
    cpuTeam->attachThreadReceiver(postGpuTeam);
    gpuTeam->attachThreadReceiver(postGpuTeam);
    gpuTeam->attachDataReceiver(&gpuToHost);
    gpuToHost.attachDataReceiver(postGpuTeam);

    std::shared_ptr<Tile>   tile_cpu{};
    std::shared_ptr<Tile>   tile_gpu{};
    auto                    packet_gpu = std::shared_ptr<CudaDataPacket>{};

    Logger::instance().log("[CudaRuntime] Start CPU/GPU action");

    cpuTeam->startCycle(cpuAction, "Concurrent_CPU_Block_Team");
    gpuTeam->startCycle(gpuAction, "Concurrent_GPU_Packet_Team");
    postGpuTeam->startCycle(postGpuAction, "Post_GPU_Packet_Team");

    // Data is enqueued for both the concurrent CPU and concurrent GPU
    // thread teams.  When a data item is finished on the GPU, the data item
    // is enqueued automatically with the post-GPU team.
    cudaStream_t   stream;
    cudaError_t    cErr = cudaErrorInvalidValue;
    unsigned int   level = 0;
    Grid&          grid = Grid::instance();
    for (auto ti = grid.buildTileIter(level); ti->isValid(); ti->next()) {
        assert(tile_cpu == nullptr);
        assert(tile_cpu.use_count() == 0);
        assert(tile_gpu == nullptr);
        assert(tile_gpu.use_count() == 0);
        assert(packet_gpu == nullptr);
        assert(packet_gpu.use_count() == 0);

        tile_cpu = ti->buildCurrentTile();
        tile_gpu = tile_cpu;
        assert(tile_cpu.get() == tile_gpu.get());
        assert(tile_cpu.use_count() == 2);

        packet_gpu = std::make_shared<CudaDataPacket>( std::move(tile_gpu) );
        assert(tile_gpu == nullptr);
        packet_gpu->pack();

        stream = *(packet_gpu->stream().object);
        cErr = cudaMemcpyAsync(packet_gpu->gpuPointer(), packet_gpu->hostPointer(),
                               packet_gpu->sizeInBytes(),
                               cudaMemcpyHostToDevice, stream);
        if (cErr != cudaSuccess) {
            std::string  errMsg = "[CudaRuntime::executeTasks_FullPacket] ";
            errMsg += "Unable to execute H-to-D transfer\n";
            errMsg += "CUDA error - " + std::string(cudaGetErrorName(cErr)) + "\n";
            errMsg += std::string(cudaGetErrorString(cErr)) + "\n";
            throw std::runtime_error(errMsg);
        }

        cpuTeam->enqueue( std::move(tile_cpu) );
        gpuTeam->enqueue( std::move(packet_gpu) );
    }
    gpuTeam->closeQueue();
    cpuTeam->closeQueue();

    // TODO: We could give subscribers a pointer to the publisher so that during
    // the subscriber's wait() it can determine if it should terminate yet.  The
    // goal of this would be to allow client code to call the wait() methods in
    // any order.  I doubt that the design/code complexity is worth this minor
    // gain.

    cpuTeam->wait();
    gpuTeam->wait();
    postGpuTeam->wait();

    Logger::instance().log("[CudaRuntime] End CPU/GPU action");

    cpuTeam->detachThreadReceiver();
    gpuTeam->detachThreadReceiver();
    gpuTeam->detachDataReceiver();
    gpuToHost.detachDataReceiver();
}

void CudaRuntime::printGpuInformation(void) const {
    std::cout << "Lead MPI Processor GPU Information\n";
    std::cout << "  Name                    "
              <<  gpuDeviceName_ << "\n";
    std::cout << "  Clock Rate              "
              << (gpuClockRateHz_ * 1.0e-9) << " GHz\n";
    std::cout << "  Memory Clock Rate       "
              << (gpuMemClockRateHz_ * 1.0e-9) << " GHz\n";
    std::cout << "  Memory Bus Width        "
              <<  gpuMemBusWidthBytes_ << " bytes\n";
    std::cout << "  Total Global Memory     "
              << (gpuTotalGlobalMemBytes_ / std::pow(1024.0, 3.0)) << " GB\n";
    std::cout << "  L2 Cache Size           "
              << (gpuL2CacheSizeBytes_ / std::pow(1024.0, 2.0)) << " MB\n";
    std::cout << "  Supports local L1 Cache "
              <<  (gpuSupportsL1Caching_ ? 'T' : 'F') << "\n";
    std::cout << "  Compute Capability      "
              <<  gpuCompMajor_ << "." << gpuCompMinor_ << "\n";
    std::cout << "  Max Grid Size           "
              <<  gpuMaxGridSize_[0] << " x "
              <<  gpuMaxGridSize_[1] << " x "
              <<  gpuMaxGridSize_[2] << "\n";
    std::cout << "  Max Thread Dims         "
              <<  gpuMaxThreadDim_[0] << " x "
              <<  gpuMaxThreadDim_[1] << " x "
              <<  gpuMaxThreadDim_[2] << "\n";
    std::cout << "  Max Threads/Block       "
              <<  gpuMaxThreadsPerBlock_ << "\n";
    std::cout << "  Warp Size               "
              <<  gpuWarpSize_ << "\n";
    std::cout << "  Num Multiprocessors     "
              <<  gpuNumMultiprocessors_ << "\n";
    std::cout << "  Max Concurrent Kernels  "
              <<  gpuMaxConcurrentKernels_ << "\n";
}

}

