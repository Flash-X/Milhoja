#ifndef MILHOJA_GENERATED_TILEWRAPPER_CPU_TASKFN_0_H__
#define MILHOJA_GENERATED_TILEWRAPPER_CPU_TASKFN_0_H__

#include <Milhoja_TileWrapper.h>

struct Tile_cpu_taskfn_0 : public milhoja::TileWrapper {
   Tile_cpu_taskfn_0(
         const milhoja::Real external_Hydro_dt,
         const milhoja::Real external_Hydro_dtOld,
         const int external_Hydro_stage
);
   ~Tile_cpu_taskfn_0(void);

   Tile_cpu_taskfn_0(Tile_cpu_taskfn_0&)                  = delete;
   Tile_cpu_taskfn_0(const Tile_cpu_taskfn_0&)            = delete;
   Tile_cpu_taskfn_0(Tile_cpu_taskfn_0&&)                 = delete;
   Tile_cpu_taskfn_0& operator=(Tile_cpu_taskfn_0&)       = delete;
   Tile_cpu_taskfn_0& operator=(const Tile_cpu_taskfn_0&) = delete;
   Tile_cpu_taskfn_0& operator=(Tile_cpu_taskfn_0&&)      = delete;

   std::unique_ptr<milhoja::TileWrapper> clone(std::shared_ptr<milhoja::Tile>&& tileToWrap) const override;

   milhoja::Real  external_Hydro_dt_;
   milhoja::Real  external_Hydro_dtOld_;
   int  external_Hydro_stage_;

   static void acquireScratch(void);
   static void releaseScratch(void);

   constexpr static std::size_t  SCRATCH_HYDRO_CVOL_FAKE_SIZE_ =
                 1
               * 1
               * 1;
   constexpr static std::size_t  SCRATCH_HYDRO_FAREAX_FAKE_SIZE_ =
                 1
               * 1
               * 1;
   constexpr static std::size_t  SCRATCH_HYDRO_FAREAY_FAKE_SIZE_ =
                 1
               * 1
               * 1;
   constexpr static std::size_t  SCRATCH_HYDRO_FAREAZ_FAKE_SIZE_ =
                 1
               * 1
               * 1;
   constexpr static std::size_t  SCRATCH_HYDRO_FLUXBUFZ_SIZE_ =
                 1
               * 1
               * 1
               * 1;
   constexpr static std::size_t  SCRATCH_HYDRO_HY_VC_SIZE_ =
                 28
               * 28
               * 1;
   constexpr static std::size_t  SCRATCH_HYDRO_HY_FLAT3D_SIZE_ =
                 28
               * 28
               * 1;
   constexpr static std::size_t  SCRATCH_HYDRO_HY_FLUX_SIZE_ =
                 28
               * 28
               * 1
               * 7;
   constexpr static std::size_t  SCRATCH_HYDRO_HY_FLX_SIZE_ =
                 28
               * 28
               * 1
               * 5;
   constexpr static std::size_t  SCRATCH_HYDRO_HY_FLY_SIZE_ =
                 28
               * 28
               * 1
               * 5;
   constexpr static std::size_t  SCRATCH_HYDRO_HY_FLZ_SIZE_ =
                 28
               * 28
               * 1
               * 5;
   constexpr static std::size_t  SCRATCH_HYDRO_HY_GRAV_SIZE_ =
                 3
               * 28
               * 28
               * 1;
   constexpr static std::size_t  SCRATCH_HYDRO_HY_ROPE_SIZE_ =
                 28
               * 28
               * 1
               * 7;
   constexpr static std::size_t  SCRATCH_HYDRO_HY_STARSTATE_SIZE_ =
                 28
               * 28
               * 1
               * 18;
   constexpr static std::size_t  SCRATCH_HYDRO_HY_TMPSTATE_SIZE_ =
                 28
               * 28
               * 1
               * 18;
   constexpr static std::size_t  SCRATCH_HYDRO_HY_UMINUS_SIZE_ =
                 28
               * 28
               * 1
               * 7;
   constexpr static std::size_t  SCRATCH_HYDRO_HY_UPLUS_SIZE_ =
                 28
               * 28
               * 1
               * 7;
   constexpr static std::size_t  SCRATCH_HYDRO_XCENTER_FAKE_SIZE_ = 1;
   constexpr static std::size_t  SCRATCH_HYDRO_XLEFT_FAKE_SIZE_ = 1;
   constexpr static std::size_t  SCRATCH_HYDRO_XRIGHT_FAKE_SIZE_ = 1;
   constexpr static std::size_t  SCRATCH_HYDRO_YCENTER_FAKE_SIZE_ = 1;
   constexpr static std::size_t  SCRATCH_HYDRO_YLEFT_FAKE_SIZE_ = 1;
   constexpr static std::size_t  SCRATCH_HYDRO_YRIGHT_FAKE_SIZE_ = 1;
   constexpr static std::size_t  SCRATCH_HYDRO_ZCENTER_FAKE_SIZE_ = 1;

   static void* scratch_Hydro_cvol_fake_;
   static void* scratch_Hydro_fareaX_fake_;
   static void* scratch_Hydro_fareaY_fake_;
   static void* scratch_Hydro_fareaZ_fake_;
   static void* scratch_Hydro_fluxBufZ_;
   static void* scratch_Hydro_hy_Vc_;
   static void* scratch_Hydro_hy_flat3d_;
   static void* scratch_Hydro_hy_flux_;
   static void* scratch_Hydro_hy_flx_;
   static void* scratch_Hydro_hy_fly_;
   static void* scratch_Hydro_hy_flz_;
   static void* scratch_Hydro_hy_grav_;
   static void* scratch_Hydro_hy_rope_;
   static void* scratch_Hydro_hy_starState_;
   static void* scratch_Hydro_hy_tmpState_;
   static void* scratch_Hydro_hy_uMinus_;
   static void* scratch_Hydro_hy_uPlus_;
   static void* scratch_Hydro_xCenter_fake_;
   static void* scratch_Hydro_xLeft_fake_;
   static void* scratch_Hydro_xRight_fake_;
   static void* scratch_Hydro_yCenter_fake_;
   static void* scratch_Hydro_yLeft_fake_;
   static void* scratch_Hydro_yRight_fake_;
   static void* scratch_Hydro_zCenter_fake_;
};

#endif
