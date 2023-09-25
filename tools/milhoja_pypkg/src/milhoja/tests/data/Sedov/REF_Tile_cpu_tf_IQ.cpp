#include "Tile_cpu_tf_IQ.h"

#include <Milhoja_Runtime.h>
#include <Milhoja_RuntimeBackend.h>
#ifdef DEBUG_RUNTIME
#include <Milhoja_Logger.h>
#endif

void*  Tile_cpu_tf_IQ::_mh_internal_volumes_ = nullptr;

void Tile_cpu_tf_IQ::acquireScratch(void) {
    const unsigned int  nThreads = milhoja::Runtime::instance().nMaxThreadsPerTeam();

    if (_mh_internal_volumes_) {
        throw std::logic_error("[Tile_cpu_tf_IQ::acquireScratch] _mh_internal_volumes scratch already allocated");
    }

    const std::size_t nBytes = nThreads
                    * Tile_cpu_tf_IQ::_MH_INTERNAL_VOLUMES_SIZE_
                    * sizeof(milhoja::Real);

    milhoja::RuntimeBackend::instance().requestCpuMemory(nBytes, &_mh_internal_volumes_);

#ifdef DEBUG_RUNTIME
    std::string   msg = "[Tile_cpu_tf_IQ::acquireScratch] Acquired"
                    + std::to_string(nThreads)
                    + " _mh_internal_volumes scratch blocks"
    milhoja::Logger::instance().log(msg);
#endif
}

void Tile_cpu_tf_IQ::releaseScratch(void) {
    if (!_mh_internal_volumes_) {
        throw std::logic_error("[Tile_cpu_tf_IQ::releaseScratch] _mh_internal_volumes scratch not allocated");
    }

    milhoja::RuntimeBackend::instance().releaseCpuMemory(&_mh_internal_volumes_);
    _mh_internal_volumes_ = nullptr;

#ifdef DEBUG_RUNTIME
    std::string   msg = "[Tile_cpu_tf_IQ::releaseScratch] Released _mh_internal_volumes scratch"
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
