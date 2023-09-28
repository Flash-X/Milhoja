#ifndef MILHOJA_GENERATED_TILE_CPU_TF_ANALYSIS_H__
#define MILHOJA_GENERATED_TILE_CPU_TF_ANALYSIS_H__

#include <Milhoja_TileWrapper.h>

struct Tile_cpu_tf_analysis : public milhoja::TileWrapper {
    Tile_cpu_tf_analysis(void);
    ~Tile_cpu_tf_analysis(void);

    Tile_cpu_tf_analysis(Tile_cpu_tf_analysis&)                  = delete;
    Tile_cpu_tf_analysis(const Tile_cpu_tf_analysis&)            = delete;
    Tile_cpu_tf_analysis(Tile_cpu_tf_analysis&&)                 = delete;
    Tile_cpu_tf_analysis& operator=(Tile_cpu_tf_analysis&)       = delete;
    Tile_cpu_tf_analysis& operator=(const Tile_cpu_tf_analysis&) = delete;
    Tile_cpu_tf_analysis& operator=(Tile_cpu_tf_analysis&&)      = delete;

    std::unique_ptr<milhoja::TileWrapper> clone(std::shared_ptr<milhoja::Tile>&& tileToWrap) const override;


    static void acquireScratch(void);
    static void releaseScratch(void);


};

#endif
