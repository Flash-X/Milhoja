#include "Tile_cpu_tf_IQ.h"

#include <Milhoja_Runtime.h>
#include <Milhoja_RuntimeBackend.h>
#ifdef DEBUG_RUNTIME
#include <Milhoja_Logger.h>
#endif

void*  Tile_cpu_tf_IQ::MH_INTERNAL_cellVolumes_ = nullptr;

void Tile_cpu_tf_IQ::acquireScratch(void) {
    const unsigned int  nThreads = milhoja::Runtime::instance().nMaxThreadsPerTeam();

    if (MH_INTERNAL_cellVolumes_) {
        throw std::logic_error("[Tile_cpu_tf_IQ::acquireScratch] MH_INTERNAL_cellVolumes_ scratch already allocated");
    }

    const std::size_t nBytes = nThreads
                    * Tile_cpu_tf_IQ::MH_INTERNAL_CELLVOLUMES_SIZE_
                    * sizeof(milhoja::Real);

    milhoja::RuntimeBackend::instance().requestCpuMemory(nBytes, &MH_INTERNAL_cellVolumes_);

#ifdef DEBUG_RUNTIME
    std::string   msg = "[Tile_cpu_tf_IQ::acquireScratch] Acquired"
                    + std::to_string(nThreads)
                    + " MH_INTERNAL_cellVolumes_ scratch blocks"
    milhoja::Logger::instance().log(msg);
#endif
}

void Tile_cpu_tf_IQ::releaseScratch(void) {
    if (!MH_INTERNAL_cellVolumes_) {
        throw std::logic_error("[Tile_cpu_tf_IQ::releaseScratch] MH_INTERNAL_cellVolumes_ scratch not allocated");
    }

    milhoja::RuntimeBackend::instance().releaseCpuMemory(&MH_INTERNAL_cellVolumes_);
    MH_INTERNAL_cellVolumes_ = nullptr;

#ifdef DEBUG_RUNTIME
    std::string   msg = "[Tile_cpu_tf_IQ::releaseScratch] Released MH_INTERNAL_cellVolumes_ scratch"
    milhoja::Logger::instance().log(msg);
#endif
}

Tile_cpu_tf_IQ::Tile_cpu_tf_IQ(void)
    : milhoja::TileWrapper{}
{
}

Tile_cpu_tf_IQ::~Tile_cpu_tf_IQ(void) {
#ifdef DEBUG_RUNTIME
    std::string   msg = "[~Tile_cpu_tf_IQ] Destroying wrapper object";
    milhoja::Logger::instance().log(msg);
#endif
}

std::unique_ptr<milhoja::TileWrapper> Tile_cpu_tf_IQ::clone(std::shared_ptr<milhoja::Tile>&& tileToWrap) const {
    Tile_cpu_tf_IQ* ptr = new Tile_cpu_tf_IQ{};

    if (ptr->tile_) {
        throw std::logic_error("[Tile_cpu_tf_IQ::clone] Internal tile_ member not null");
    }
    ptr->tile_ = std::move(tileToWrap);
    if (!(ptr->tile_) || tileToWrap) {
        throw std::logic_error("[Tile_cpu_tf_IQ::clone] Wrapper did not take ownership of tile");
    }

    return std::unique_ptr<milhoja::TileWrapper>{ptr};
}
