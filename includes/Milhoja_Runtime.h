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

#include "Milhoja.h"
#include "Milhoja_ThreadTeam.h"
#ifndef RUNTIME_MUST_USE_TILEITER
#include "Milhoja_FlashxrTileRaw.h"
#endif
#include "Milhoja_TileWrapper.h"
#include "Milhoja_DataPacket.h"
#include "Milhoja_RuntimeAction.h"

#ifdef RUNTIME_SUPPORT_DATAPACKETS
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
                                     const std::size_t nBytesInCpuMemoryPool,
                                     const std::size_t nBytesInGpuMemoryPools);
    static Runtime&      instance(void);
    void                 finalize(void);

    unsigned int   nMaxThreadsPerTeam(void) const {
        return maxThreadsPerTeam_;
    }

#ifndef RUNTIME_MUST_USE_TILEITER
    void setupPipelineForCpuTasks(const std::string& actionName,
                                  const RuntimeAction& cpuAction);
    void pushTileToPipeline(const std::string& actionName,
                            const TileWrapper& prototype,
                            const FlashxrTileRawPtrs& tP,
                            const FlashxTileRawInts& tI,
                            const FlashxTileRawReals& tR);
    void teardownPipelineForCpuTasks(const std::string& actionName);
#endif
    void executeCpuTasks(const std::string& actionName,
                         const RuntimeAction& cpuAction,
                         const TileWrapper& prototype);
#ifdef RUNTIME_SUPPORT_DATAPACKETS
#  ifndef RUNTIME_MUST_USE_TILEITER
    void setupPipelineForGpuTasks(const std::string& bundleName,
                                  const unsigned int stagger_usec,
                                  const RuntimeAction& gpuAction,
                                  const DataPacket& packetPrototype);
    void pushTileToGpuPipeline(const std::string& bundleName,
                               const DataPacket& packetPrototype,
                               const FlashxrTileRawPtrs& tP,
                               const FlashxTileRawInts& tI,
                               const FlashxTileRawReals& tR);
    void teardownPipelineForGpuTasks(const std::string& bundleName);
#  endif
    void executeGpuTasks(const std::string& actionName,
                         const unsigned int nDistributorThreads,
                         const unsigned int stagger_usec,
                         const RuntimeAction& gpuAction,
                         const DataPacket& packetPrototype);
#  ifdef MILHOJA_TIMED_PIPELINE_CONFIGS
    void executeGpuTasks_timed(const std::string& actionName,
                               const unsigned int nDistributorThreads,
                               const unsigned int stagger_usec,
                               const RuntimeAction& gpuAction,
                               const DataPacket& packetPrototype,
                               const unsigned int stepNumber,
                               const MPI_Comm comm);
#  endif
#  ifdef MILHOJA_ADDTL_PIPELINE_CONFIGS
    void executeCpuGpuTasks(const std::string& bundleName,
                            const RuntimeAction& cpuAction,
                            const TileWrapper& tilePrototype,
                            const RuntimeAction& gpuAction,
                            const DataPacket& packetPrototype);
#  endif
#  ifndef RUNTIME_MUST_USE_TILEITER
    void setupPipelineForExtGpuTasks(const std::string& bundleName,
                                     const RuntimeAction& gpuAction,
                                     const RuntimeAction& postGpuAction,
                                     const DataPacket& packetPrototype,
                                     const TileWrapper& tilePrototype);
    void pushTileToExtGpuPipeline(const std::string& bundleName,
                                  const DataPacket& packetPrototype,
                                  const FlashxrTileRawPtrs& tP,
                                  const FlashxTileRawInts& tI,
                                  const FlashxTileRawReals& tR);
    void teardownPipelineForExtGpuTasks(const std::string& bundleName);
#  endif
    void executeExtendedGpuTasks(const std::string& bundleName,
                                 const unsigned int nDistributorThreads,
                                 const RuntimeAction& gpuAction,
                                 const RuntimeAction& postGpuAction,
                                 const DataPacket& packetPrototype,
                                 const TileWrapper& tilePrototype);
#  ifdef MILHOJA_ADDTL_PIPELINE_CONFIGS
    void executeCpuGpuSplitTasks(const std::string& bundleName,
                                 const unsigned int nDistributorThreads,
                                 const unsigned int stagger_usec,
                                 const RuntimeAction& cpuAction,
                                 const TileWrapper& tilePrototype,
                                 const RuntimeAction& gpuAction,
                                 const DataPacket& packetPrototype,
                                 const unsigned int nTilesPerCpuTurn);
#    ifdef MILHOJA_TIMED_PIPELINE_CONFIGS
    void executeCpuGpuSplitTasks_timed(const std::string& bundleName,
                                       const unsigned int nDistributorThreads,
                                       const unsigned int stagger_usec,
                                       const RuntimeAction& cpuAction,
                                       const TileWrapper& tilePrototype,
                                       const RuntimeAction& gpuAction,
                                       const DataPacket& packetPrototype,
                                       const unsigned int nTilesPerCpuTurn,
                                       const unsigned int stepNumber,
                                       const MPI_Comm comm);
#    endif
    void executeExtendedCpuGpuSplitTasks(const std::string& bundleName,
                                         const unsigned int nDistributorThreads,
                                         const RuntimeAction& actionA_cpu,
                                         const RuntimeAction& actionA_gpu,
                                         const RuntimeAction& postActionB_cpu,
                                         const DataPacket& packetPrototype,
                                         const unsigned int nTilesPerCpuTurn);
    void executeCpuGpuWowzaTasks(const std::string& bundleName,
                                 const RuntimeAction& actionA_cpu,
                                 const TileWrapper& tilePrototype,
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
#  endif
#endif

private:
    Runtime(void);

    static unsigned int    nTeams_; 
    static unsigned int    maxThreadsPerTeam_;
    static bool            initialized_;
    static bool            finalized_;
#if defined(RUNTIME_SUPPORT_DATAPACKETS) && defined(RUNTIME_SUPPORT_PUSH)
    int                    nTilesPerPacket_;
#endif
#ifndef RUNTIME_MUST_USE_TILEITER
#  ifdef RUNTIME_SUPPORT_DATAPACKETS
    std::shared_ptr<DataPacket> packet_gpu_;
#  else
    std::unique_ptr<DataPacket> packet_gpu_;
#  endif
#endif

    ThreadTeam**     teams_;

#ifdef RUNTIME_SUPPORT_DATAPACKETS
    MoverUnpacker    gpuToHost1_;
#ifdef MILHOJA_ADDTL_PIPELINE_CONFIGS
    MoverUnpacker    gpuToHost2_;
#endif
#endif
};

}

#endif

