#ifndef CGKIT_GENERATED_TILE_CPU_TF_IQ_3D_H__
#define CGKIT_GENERATED_TILE_CPU_TF_IQ_3D_H__

#include <Milhoja_TileWrapper.h>

struct Tile_cpu_tf_IQ_3D : public milhoja::TileWrapper {
    Tile_cpu_tf_IQ_3D(void);
    ~Tile_cpu_tf_IQ_3D(void);

    Tile_cpu_tf_IQ_3D(Tile_cpu_tf_IQ_3D&)                  = delete;
    Tile_cpu_tf_IQ_3D(const Tile_cpu_tf_IQ_3D&)            = delete;
    Tile_cpu_tf_IQ_3D(Tile_cpu_tf_IQ_3D&&)                 = delete;
    Tile_cpu_tf_IQ_3D& operator=(Tile_cpu_tf_IQ_3D&)       = delete;
    Tile_cpu_tf_IQ_3D& operator=(const Tile_cpu_tf_IQ_3D&) = delete;
    Tile_cpu_tf_IQ_3D& operator=(Tile_cpu_tf_IQ_3D&&)      = delete;

    std::unique_ptr<milhoja::TileWrapper> clone(std::unique_ptr<milhoja::Tile>&& tileToWrap) const override;


    static void acquireScratch(void);
    static void releaseScratch(void);

    constexpr static std::size_t  _MH_INTERNAL_VOLUMES_SIZE_ =
                      18
                    * 18
                    * 18;

    static void* _mh_internal_volumes_;
};

#endif
