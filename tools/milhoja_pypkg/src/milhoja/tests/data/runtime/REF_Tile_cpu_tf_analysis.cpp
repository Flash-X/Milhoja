#include "Tile_cpu_tf_analysis.h"

#include <Milhoja_Runtime.h>
#include <Milhoja_RuntimeBackend.h>
#ifdef DEBUG_RUNTIME
#include <Milhoja_Logger.h>
#endif


void Tile_cpu_tf_analysis::acquireScratch(void) {
}

void Tile_cpu_tf_analysis::releaseScratch(void) {
}

Tile_cpu_tf_analysis::Tile_cpu_tf_analysis(void)
    : milhoja::TileWrapper{}
{
#ifdef DEBUG_RUNTIME
    std::string   msg = "[Tile_cpu_tf_analysis] Creating wrapper object";
    milhoja::Logger::instance().log(msg);
#endif
}

Tile_cpu_tf_analysis::~Tile_cpu_tf_analysis(void) {
#ifdef DEBUG_RUNTIME
    std::string   msg = "[~Tile_cpu_tf_analysis] Destroying wrapper object";
    milhoja::Logger::instance().log(msg);
#endif
}

std::unique_ptr<milhoja::TileWrapper> Tile_cpu_tf_analysis::clone(std::shared_ptr<milhoja::Tile>&& tileToWrap) const {
    Tile_cpu_tf_analysis* ptr = new Tile_cpu_tf_analysis{};

    if (ptr->tile_) {
        throw std::logic_error("[Tile_cpu_tf_analysis::clone] Internal tile_ member not null");
    }
    ptr->tile_ = std::move(tileToWrap);
    if (!(ptr->tile_) || tileToWrap) {
        throw std::logic_error("[Tile_cpu_tf_analysis::clone] Wrapper did not take ownership of tile");
    }

    return std::unique_ptr<milhoja::TileWrapper>{ptr};
}
