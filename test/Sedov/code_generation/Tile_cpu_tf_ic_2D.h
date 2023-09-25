#ifndef CGKIT_GENERATED_TILE_CPU_TF_IC_2D_H__
#define CGKIT_GENERATED_TILE_CPU_TF_IC_2D_H__

#include <Milhoja_TileWrapper.h>

struct Tile_cpu_tf_ic_2D : public milhoja::TileWrapper {
    Tile_cpu_tf_ic_2D(void);
    ~Tile_cpu_tf_ic_2D(void);

    Tile_cpu_tf_ic_2D(Tile_cpu_tf_ic_2D&)                  = delete;
    Tile_cpu_tf_ic_2D(const Tile_cpu_tf_ic_2D&)            = delete;
    Tile_cpu_tf_ic_2D(Tile_cpu_tf_ic_2D&&)                 = delete;
    Tile_cpu_tf_ic_2D& operator=(Tile_cpu_tf_ic_2D&)       = delete;
    Tile_cpu_tf_ic_2D& operator=(const Tile_cpu_tf_ic_2D&) = delete;
    Tile_cpu_tf_ic_2D& operator=(Tile_cpu_tf_ic_2D&&)      = delete;

    std::unique_ptr<milhoja::TileWrapper> clone(std::shared_ptr<milhoja::Tile>&& tileToWrap) const override;


    static void acquireScratch(void);
    static void releaseScratch(void);


};

#endif
