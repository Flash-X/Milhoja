#include "Tile_cpu_tf_dens.h"

#include <Milhoja_Runtime.h>
#include <Milhoja_RuntimeBackend.h>
#ifdef DEBUG_RUNTIME
#include <Milhoja_Logger.h>
#endif

void*  Tile_cpu_tf_dens::base_op1_scratch_ = nullptr;

void Tile_cpu_tf_dens::acquireScratch(void) {
    const unsigned int  nThreads = milhoja::Runtime::instance().nMaxThreadsPerTeam();

    if (base_op1_scratch_) {
        throw std::logic_error("[Tile_cpu_tf_dens::acquireScratch] base_op1_scratch_ scratch already allocated");
    }

    const std::size_t nBytes = nThreads
                    * Tile_cpu_tf_dens::BASE_OP1_SCRATCH_SIZE_
                    * sizeof(milhoja::Real);

    milhoja::RuntimeBackend::instance().requestCpuMemory(nBytes, &base_op1_scratch_);

#ifdef DEBUG_RUNTIME
    std::string   msg = "[Tile_cpu_tf_dens::acquireScratch] Acquired"
                    + std::to_string(nThreads)
                    + " base_op1_scratch_ scratch blocks"
    milhoja::Logger::instance().log(msg);
#endif
}

void Tile_cpu_tf_dens::releaseScratch(void) {
    if (!base_op1_scratch_) {
        throw std::logic_error("[Tile_cpu_tf_dens::releaseScratch] base_op1_scratch_ scratch not allocated");
    }

    milhoja::RuntimeBackend::instance().releaseCpuMemory(&base_op1_scratch_);
    base_op1_scratch_ = nullptr;

#ifdef DEBUG_RUNTIME
    std::string   msg = "[Tile_cpu_tf_dens::releaseScratch] Released base_op1_scratch_ scratch"
    milhoja::Logger::instance().log(msg);
#endif
}

Tile_cpu_tf_dens::Tile_cpu_tf_dens(void)
    : milhoja::TileWrapper{}
{
#ifdef DEBUG_RUNTIME
    std::string   msg = "[Tile_cpu_tf_dens] Creating wrapper object";
    milhoja::Logger::instance().log(msg);
#endif
}

Tile_cpu_tf_dens::~Tile_cpu_tf_dens(void) {
#ifdef DEBUG_RUNTIME
    std::string   msg = "[~Tile_cpu_tf_dens] Destroying wrapper object";
    milhoja::Logger::instance().log(msg);
#endif
}

std::unique_ptr<milhoja::TileWrapper> Tile_cpu_tf_dens::clone(std::shared_ptr<milhoja::Tile>&& tileToWrap) const {
    Tile_cpu_tf_dens* ptr = new Tile_cpu_tf_dens{};

    if (ptr->tile_) {
        throw std::logic_error("[Tile_cpu_tf_dens::clone] Internal tile_ member not null");
    }
    ptr->tile_ = std::move(tileToWrap);
    if (!(ptr->tile_) || tileToWrap) {
        throw std::logic_error("[Tile_cpu_tf_dens::clone] Wrapper did not take ownership of tile");
    }

    return std::unique_ptr<milhoja::TileWrapper>{ptr};
}
