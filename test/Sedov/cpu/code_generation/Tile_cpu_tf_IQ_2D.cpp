#include "Tile_cpu_tf_IQ_2D.h"

#include <Milhoja_Runtime.h>
#include <Milhoja_RuntimeBackend.h>
#ifdef DEBUG_RUNTIME
#include <Milhoja_Logger.h>
#endif

void*  Tile_cpu_tf_IQ_2D::_mh_internal_volumes_ = nullptr;

void Tile_cpu_tf_IQ_2D::acquireScratch(void) {
    const unsigned int  nThreads = milhoja::Runtime::instance().nMaxThreadsPerTeam();

    if (_mh_internal_volumes_) {
        throw std::logic_error("[Tile_cpu_tf_IQ_2D::acquireScratch] _mh_internal_volumes scratch already allocated");
    }

    const std::size_t nBytes = nThreads
                    * Tile_cpu_tf_IQ_2D::_MH_INTERNAL_VOLUMES_SIZE_
                    * sizeof(milhoja::Real);

    milhoja::RuntimeBackend::instance().requestCpuMemory(nBytes, &_mh_internal_volumes_);

#ifdef DEBUG_RUNTIME
    std::string   msg = "[Tile_cpu_tf_IQ_2D::acquireScratch] Acquired"
                    + std::to_string(nThreads)
                    + " _mh_internal_volumes scratch blocks"
    milhoja::Logger::instance().log(msg);
#endif
}

void Tile_cpu_tf_IQ_2D::releaseScratch(void) {
    if (!_mh_internal_volumes_) {
        throw std::logic_error("[Tile_cpu_tf_IQ_2D::releaseScratch] _mh_internal_volumes scratch not allocated");
    }

    milhoja::RuntimeBackend::instance().releaseCpuMemory(&_mh_internal_volumes_);
    _mh_internal_volumes_ = nullptr;

#ifdef DEBUG_RUNTIME
    std::string   msg = "[Tile_cpu_tf_IQ_2D::releaseScratch] Released _mh_internal_volumes scratch"
    milhoja::Logger::instance().log(msg);
#endif
}

Tile_cpu_tf_IQ_2D::Tile_cpu_tf_IQ_2D(void)
    : milhoja::TileWrapper{}
{
}

Tile_cpu_tf_IQ_2D::~Tile_cpu_tf_IQ_2D(void) {
#ifdef DEBUG_RUNTIME
    std::string   msg = "[~Tile_cpu_tf_IQ_2D] Destroying wrapper object";
    milhoja::Logger::instance().log(msg);
#endif
}

std::unique_ptr<milhoja::TileWrapper> Tile_cpu_tf_IQ_2D::clone(std::unique_ptr<milhoja::Tile>&& tileToWrap) const {
    Tile_cpu_tf_IQ_2D* ptr = new Tile_cpu_tf_IQ_2D{};

    if (ptr->tile_) {
        throw std::logic_error("[Tile_cpu_tf_IQ_2D::clone] Internal tile_ member not null");
    }
    ptr->tile_ = std::move(tileToWrap);
    if (!(ptr->tile_) || tileToWrap) {
        throw std::logic_error("[Tile_cpu_tf_IQ_2D::clone] Wrapper did not take ownership of tile");
    }

    return std::unique_ptr<milhoja::TileWrapper>{ptr};
}
