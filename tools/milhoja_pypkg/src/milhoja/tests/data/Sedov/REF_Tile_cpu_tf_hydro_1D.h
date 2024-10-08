#ifndef MILHOJA_GENERATED_TILE_CPU_TF_HYDRO_H__
#define MILHOJA_GENERATED_TILE_CPU_TF_HYDRO_H__

#include <Milhoja_TileWrapper.h>

struct Tile_cpu_tf_hydro : public milhoja::TileWrapper {
    Tile_cpu_tf_hydro(
      const milhoja::Real external_hydro_op1_dt
    );
    ~Tile_cpu_tf_hydro(void);

    Tile_cpu_tf_hydro(Tile_cpu_tf_hydro&)                  = delete;
    Tile_cpu_tf_hydro(const Tile_cpu_tf_hydro&)            = delete;
    Tile_cpu_tf_hydro(Tile_cpu_tf_hydro&&)                 = delete;
    Tile_cpu_tf_hydro& operator=(Tile_cpu_tf_hydro&)       = delete;
    Tile_cpu_tf_hydro& operator=(const Tile_cpu_tf_hydro&) = delete;
    Tile_cpu_tf_hydro& operator=(Tile_cpu_tf_hydro&&)      = delete;

    std::unique_ptr<milhoja::TileWrapper> clone(std::shared_ptr<milhoja::Tile>&& tileToWrap) const override;

    milhoja::Real  external_hydro_op1_dt_;

    static void acquireScratch(void);
    static void releaseScratch(void);

    constexpr static std::size_t  SCRATCH_HYDRO_OP1_AUXC_SIZE_ =
                      18
                    * 1
                    * 1;

    static void* scratch_hydro_op1_auxC_;
};

#endif
