/**
 * \file    Milhoja_Runtime.h
 *
 * \brief 
 *
 */

#ifndef MILHOJA_RUNTIME_H__
#define MILHOJA_RUNTIME_H__

#include <string>

#include <mpi.h>

#include "Milhoja_ThreadTeam.h"
#include "Milhoja_DataPacket.h"
#include "Milhoja_RuntimeAction.h"

#if defined(MILHOJA_USE_CUDA_BACKEND)
#include "Milhoja_MoverUnpacker.h"
#endif

namespace milhoja {

class Runtime {
public:
    ~Runtime(void);

    Runtime(Runtime&)                  = delete;
    Runtime(const Runtime&)            = delete;
    Runtime(Runtime&&)                 = delete;
    Runtime& operator=(Runtime&)       = delete;
    Runtime& operator=(const Runtime&) = delete;
    Runtime& operator=(Runtime&&)      = delete;

    static void          initialize(const unsigned int nTeams,
                                     const unsigned int nThreadsPerTeam,
                                     const unsigned int nStreams,
                                     const std::size_t nBytesInMemoryPools);
    static Runtime&      instance(void);
    void                 finalize(void);

    void executeCpuTasks(const std::string& actionName,
                         const RuntimeAction& cpuAction);
#if defined(MILHOJA_USE_CUDA_BACKEND)
    void executeGpuTasks(const std::string& actionName,
                         const unsigned int nDistributorThreads,
                         const unsigned int stagger_usec,
                         const RuntimeAction& gpuAction,
                         const DataPacket& packetPrototype);
    void executeGpuTasks_timed(const std::string& actionName,
                               const unsigned int nDistributorThreads,
                               const unsigned int stagger_usec,
                               const RuntimeAction& gpuAction,
                               const DataPacket& packetPrototype,
                               const unsigned int stepNumber,
                               const MPI_Comm comm);
    void executeCpuGpuTasks(const std::string& bundleName,
                            const RuntimeAction& cpuAction,
                            const RuntimeAction& gpuAction,
                            const DataPacket& packetPrototype);
    void executeExtendedGpuTasks(const std::string& bundleName,
                                 const unsigned int nDistributorThreads,
                                 const RuntimeAction& gpuAction,
                                 const RuntimeAction& postGpuAction,
                                 const DataPacket& packetPrototype);
    void executeCpuGpuSplitTasks(const std::string& bundleName,
                                 const unsigned int nDistributorThreads,
                                 const unsigned int stagger_usec,
                                 const RuntimeAction& cpuAction,
                                 const RuntimeAction& gpuAction,
                                 const DataPacket& packetPrototype,
                                 const unsigned int nTilesPerCpuTurn);
    void executeCpuGpuSplitTasks_timed(const std::string& bundleName,
                                       const unsigned int nDistributorThreads,
                                       const unsigned int stagger_usec,
                                       const RuntimeAction& cpuAction,
                                       const RuntimeAction& gpuAction,
                                       const DataPacket& packetPrototype,
                                       const unsigned int nTilesPerCpuTurn,
                                       const unsigned int stepNumber,
                                       const MPI_Comm comm);
    void executeExtendedCpuGpuSplitTasks(const std::string& bundleName,
                                         const unsigned int nDistributorThreads,
                                         const RuntimeAction& actionA_cpu,
                                         const RuntimeAction& actionA_gpu,
                                         const RuntimeAction& postActionB_cpu,
                                         const DataPacket& packetPrototype,
                                         const unsigned int nTilesPerCpuTurn);
    void executeCpuGpuWowzaTasks(const std::string& bundleName,
                                 const RuntimeAction& actionA_cpu,
                                 const RuntimeAction& actionA_gpu,
                                 const RuntimeAction& actionB_gpu,
                                 const DataPacket& packetPrototypeA,
                                 const DataPacket& packetPrototypeB,
                                 const unsigned int nTilesPerCpuTurn);
    void executeTasks_FullPacket(const std::string& bundleName,
                                 const RuntimeAction& cpuAction,
                                 const RuntimeAction& gpuAction,
                                 const RuntimeAction& postGpuAction,
                                 const DataPacket& packetPrototype);
#endif

private:
    Runtime(void);

    static unsigned int    nTeams_; 
    static unsigned int    maxThreadsPerTeam_;
    static bool            initialized_;
    static bool            finalized_;

    ThreadTeam**     teams_;

#if defined(MILHOJA_USE_CUDA_BACKEND)
    MoverUnpacker    gpuToHost1_;
    MoverUnpacker    gpuToHost2_;
#endif
};

}

#endif

