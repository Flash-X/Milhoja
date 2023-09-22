#include "Tile_cpu_tf_ic_2D.h"

#include <Milhoja_Runtime.h>
#include <Milhoja_RuntimeBackend.h>
#ifdef DEBUG_RUNTIME
#include <Milhoja_Logger.h>
#endif


void Tile_cpu_tf_ic_2D::acquireScratch(void) {
    const unsigned int  nThreads = milhoja::Runtime::instance().nMaxThreadsPerTeam();

}

void Tile_cpu_tf_ic_2D::releaseScratch(void) {
}

Tile_cpu_tf_ic_2D::Tile_cpu_tf_ic_2D(void)
    : milhoja::TileWrapper{}
{
}

Tile_cpu_tf_ic_2D::~Tile_cpu_tf_ic_2D(void) {
#ifdef DEBUG_RUNTIME
    std::string   msg = "[~Tile_cpu_tf_ic_2D] Destroying wrapper object";
    milhoja::Logger::instance().log(msg);
#endif
}

std::unique_ptr<milhoja::TileWrapper> Tile_cpu_tf_ic_2D::clone(std::unique_ptr<milhoja::Tile>&& tileToWrap) const {
    Tile_cpu_tf_ic_2D* ptr = new Tile_cpu_tf_ic_2D{};

    if (ptr->tile_) {
        throw std::logic_error("[Tile_cpu_tf_ic_2D::clone] Internal tile_ member not null");
    }
    ptr->tile_ = std::move(tileToWrap);
    if (!(ptr->tile_) || tileToWrap) {
        throw std::logic_error("[Tile_cpu_tf_ic_2D::clone] Wrapper did not take ownership of tile");
    }

    return std::unique_ptr<milhoja::TileWrapper>{ptr};
}
