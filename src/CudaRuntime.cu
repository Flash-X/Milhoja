#include "CudaRuntime.h"

#include <cmath>
#include <stdexcept>
#include <iostream>

#include "ThreadTeam.h"
#include "Grid.h"
#include "OrchestrationLogger.h"
#include "CudaStreamManager.h"

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
    Logger::instance().log("[CudaRuntime] Initializing");

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

    Logger::instance().log("[CudaRuntime] Initialized");
}

/**
 * 
 *
 * \return 
 */
CudaRuntime::~CudaRuntime(void) {
    Logger::instance().log("[CudaRuntime] Finalizing");

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

//void CudaRuntime::executeGpuTasks(const std::string& bundleName,
//                                  const RuntimeAction& gpuAction) {
//    if (gpuAction.teamType != ThreadTeamDataType::SET_OF_BLOCKS) {
//        throw std::logic_error("[CudaRuntime::executeGpuTasks] "
//                               "Given GPU action should run on a thread team "
//                               "that works with data packets of blocks");
//    } else if (gpuAction.nTilesPerPacket <= 0) {
//        throw std::invalid_argument("[CudaRuntime::executeGpuTasks] "
//                                    "Need at least one tile per packet");
//    } else if (nTeams_ < 1) {
//        throw std::logic_error("[CudaRuntime::executeGpuTasks] "
//                               "Need at least one ThreadTeam in runtime");
//    }
//
//    ThreadTeam*   gpuTeam = teams_[0];
//
//    CudaTransferSubsciber     gpuToHost(SRC_GPU, DEST_HOST);
//    gpuTeam->attachWorkReceiver(&gpuToHost);
//
//    // TODO: Good idea to call this as early as possible so that all threads
//    //       go to wait while this this is setting everything up?  Hides
//    //       TT wind-up cost.
//    gpuTeam->startCycle(gpuAction, "GPU_PacketOfBlocks_Team");
//
//    // We allocate the CUD memory here and pass the pointers around
//    // with the data packet so that a work subscriber will
//    // eventually deallocate the memory.
//    auto packet = std::shared_ptr<DataItem>{ new DataPacket{} };
//    if (!packet->isNull()) {
//        throw std::runtime_error("[CudaRuntime::executeGpuTasks] "
//                                 "Data packet not null on instantiation");
//    }
//
//    cudaError_t    cErr = cudaErrorInvalidValue;
//    unsigned int   level = 0;
//    Grid&   grid = Grid::instance();
//    for (auto ti = grid.buildTileIter(0); ti->isValid(); ti->next()) {
//        packet->addSubItem( ti->buildCurrentTile() );
//
//        if (packet->nSubItems() >= gpuAction.nTilesPerPacket) {
//            // TODO: Put the move into DataPacket?  It seems somewhat nice, but
//            // beyond the scope of a simple DataPacket object.  If we do this,
//            // then this class might be unrelated to Cuda!  We might need a
//            // factory to generate things like CudaTransferSuscriber.
//            packet.prepareForTransfer();
//            cErr = cudaMemcpyAsync(packet->devicePtr,
//                                   packet->pinnedPtr, 
//                                   packet->size*sizeof(double), 
//                                   cudaMemcpyHostToDevice, 
//                                   packet->stream);
//            if (cErr != cudaSuccess) {
//                std::string  errMsg = "[CudaRuntime::executeGpuTasks] ";
//                errMsg += "Unable to initiate asynchronous transfer of data packet\n";
//                errMsg += "CUDA error - " + std::string(cudaGetErrorName(cErr)) + "\n";
//                errMsg += std::string(cudaGetErrorString(cErr)) + "\n";
//                throw std::runtime_error(errMsg);
//            }
//
//            gpuTeam->enqueue( std::move(packet) );
//            if (!packet->isNull()) {
//                throw std::runtime_error("[CudaRuntime::executeGpuTasks] "
//                                         "Data packet not null after move");
//            }
//            packet = std::shared_ptr<DataItem>( new DataPacket{} );
//        }
//    }
//
//    if (packet->nSubItems() != 0) {
//        packet->prepareForTransfer();
//        cErr = cudaMemcpyAsync(packet->devicePtr,
//                               packet->pinnedPtr, 
//                               packet->size*sizeof(double), 
//                               cudaMemcpyHostToDevice, 
//                               packet->stream);
//        if (cErr != cudaSuccess) {
//            std::string  errMsg = "[CudaRuntime::executeGpuTasks] ";
//            errMsg += "Unable to initiate asynchronous transfer of last data packet\n";
//            errMsg += "CUDA error - " + std::string(cudaGetErrorName(cErr)) + "\n";
//            errMsg += std::string(cudaGetErrorString(cErr)) + "\n";
//            throw std::runtime_error(errMsg);
//        }
//
//        gpuTeam->enqueue( std::move(packet) );
//    }
//
//    if (!packet->isNull()) {
//        throw std::runtime_error("[CudaRuntime::executeGpuTasks] "
//                                 "Data packet not null after move");
//    }
//    gpuTeam->closeQueue();
//    gpuTeam->wait();
//
//    gpuTeam->detachWorkReceiver();
//}

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

