#include "Tile_cpu_tf_hydro.h"

#include <Milhoja_Runtime.h>
#include <Milhoja_RuntimeBackend.h>
#ifdef DEBUG_RUNTIME
#include <Milhoja_Logger.h>
#endif

void*  Tile_cpu_tf_hydro::scratch_hydro_op1_auxC_ = nullptr;

void Tile_cpu_tf_hydro::acquireScratch(void) {
    const unsigned int  nThreads = milhoja::Runtime::instance().nMaxThreadsPerTeam();

    if (scratch_hydro_op1_auxC_) {
        throw std::logic_error("[Tile_cpu_tf_hydro::acquireScratch] scratch_hydro_op1_auxC_ scratch already allocated");
    }

    const std::size_t nBytes_scratch_hydro_op1_auxC = nThreads
                    * Tile_cpu_tf_hydro::SCRATCH_HYDRO_OP1_AUXC_SIZE_
                    * sizeof(milhoja::Real);

    milhoja::RuntimeBackend::instance().requestCpuMemory(nBytes_scratch_hydro_op1_auxC, &scratch_hydro_op1_auxC_);

#ifdef DEBUG_RUNTIME
    {
        std::string   msg = "[Tile_cpu_tf_hydro::acquireScratch] Acquired"
                        + std::to_string(nThreads)
                        + " scratch_hydro_op1_auxC_ scratch blocks";
        milhoja::Logger::instance().log(msg);
    }
#endif
}

void Tile_cpu_tf_hydro::releaseScratch(void) {
    if (!scratch_hydro_op1_auxC_) {
        throw std::logic_error("[Tile_cpu_tf_hydro::releaseScratch] scratch_hydro_op1_auxC_ scratch not allocated");
    }

    milhoja::RuntimeBackend::instance().releaseCpuMemory(&scratch_hydro_op1_auxC_);
    scratch_hydro_op1_auxC_ = nullptr;

#ifdef DEBUG_RUNTIME
    {
        std::string   msg = "[Tile_cpu_tf_hydro::releaseScratch] Released scratch_hydro_op1_auxC_ scratch";
        milhoja::Logger::instance().log(msg);
    }
#endif
}

Tile_cpu_tf_hydro::Tile_cpu_tf_hydro(
    const milhoja::Real external_hydro_op1_dt
)
    : milhoja::TileWrapper{},
      external_hydro_op1_dt_{external_hydro_op1_dt}
{
#ifdef DEBUG_RUNTIME
    {
        std::string   msg = "[Tile_cpu_tf_hydro] Creating wrapper object";
        milhoja::Logger::instance().log(msg);
    }
#endif
}

Tile_cpu_tf_hydro::~Tile_cpu_tf_hydro(void) {
#ifdef DEBUG_RUNTIME
    {
        std::string   msg = "[~Tile_cpu_tf_hydro] Destroying wrapper object";
        milhoja::Logger::instance().log(msg);
    }
#endif
}

std::unique_ptr<milhoja::TileWrapper> Tile_cpu_tf_hydro::clone(std::shared_ptr<milhoja::Tile>&& tileToWrap) const {
    Tile_cpu_tf_hydro* ptr = new Tile_cpu_tf_hydro{external_hydro_op1_dt_};

    if (ptr->tile_) {
        throw std::logic_error("[Tile_cpu_tf_hydro::clone] Internal tile_ member not null");
    }
    ptr->tile_ = std::move(tileToWrap);
    if (!(ptr->tile_) || tileToWrap) {
        throw std::logic_error("[Tile_cpu_tf_hydro::clone] Wrapper did not take ownership of tile");
    }

    return std::unique_ptr<milhoja::TileWrapper>{ptr};
}
