#ifndef CGKIT_GENERATED_TILE_CPU_TF_IC_H__
#define CGKIT_GENERATED_TILE_CPU_TF_IC_H__

#include <Milhoja_TileWrapper.h>

struct Tile_cpu_tf_ic : public milhoja::TileWrapper {
    Tile_cpu_tf_ic(void);
    ~Tile_cpu_tf_ic(void);

    Tile_cpu_tf_ic(Tile_cpu_tf_ic&)                  = delete;
    Tile_cpu_tf_ic(const Tile_cpu_tf_ic&)            = delete;
    Tile_cpu_tf_ic(Tile_cpu_tf_ic&&)                 = delete;
    Tile_cpu_tf_ic& operator=(Tile_cpu_tf_ic&)       = delete;
    Tile_cpu_tf_ic& operator=(const Tile_cpu_tf_ic&) = delete;
    Tile_cpu_tf_ic& operator=(Tile_cpu_tf_ic&&)      = delete;

    std::unique_ptr<milhoja::TileWrapper> clone(std::shared_ptr<milhoja::Tile>&& tileToWrap) const override;


    // TODO: Can we get rid of these?
    static void acquireScratch(void);
    static void releaseScratch(void);


};

#endif
