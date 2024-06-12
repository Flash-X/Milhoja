#include "Tile_cpu_tf_dens.h"

#include <Milhoja_Runtime.h>
#include <Milhoja_RuntimeBackend.h>
#ifdef DEBUG_RUNTIME
#include <Milhoja_Logger.h>
#endif

void*  Tile_cpu_tf_dens::scratch_base_op1_scratch3D_ = nullptr;

void Tile_cpu_tf_dens::acquireScratch(void) {
    const unsigned int  nThreads = milhoja::Runtime::instance().nMaxThreadsPerTeam();

    if (scratch_base_op1_scratch3D_) {
        throw std::logic_error("[Tile_cpu_tf_dens::acquireScratch] scratch_base_op1_scratch3D_ scratch already allocated");
    }

    const std::size_t nBytes_scratch_base_op1_scratch3D = nThreads
                    * Tile_cpu_tf_dens::SCRATCH_BASE_OP1_SCRATCH3D_SIZE_
                    * sizeof(milhoja::Real);

    milhoja::RuntimeBackend::instance().requestCpuMemory(nBytes_scratch_base_op1_scratch3D, &scratch_base_op1_scratch3D_);

#ifdef DEBUG_RUNTIME
    std::string   msg = "[Tile_cpu_tf_dens::acquireScratch] Acquired"
                    + std::to_string(nThreads)
                    + " scratch_base_op1_scratch3D_ scratch blocks"
    milhoja::Logger::instance().log(msg);
#endif
}

void Tile_cpu_tf_dens::releaseScratch(void) {
    if (!scratch_base_op1_scratch3D_) {
        throw std::logic_error("[Tile_cpu_tf_dens::releaseScratch] scratch_base_op1_scratch3D_ scratch not allocated");
    }

    milhoja::RuntimeBackend::instance().releaseCpuMemory(&scratch_base_op1_scratch3D_);
    scratch_base_op1_scratch3D_ = nullptr;

#ifdef DEBUG_RUNTIME
    std::string   msg = "[Tile_cpu_tf_dens::releaseScratch] Released scratch_base_op1_scratch3D_ scratch"
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
