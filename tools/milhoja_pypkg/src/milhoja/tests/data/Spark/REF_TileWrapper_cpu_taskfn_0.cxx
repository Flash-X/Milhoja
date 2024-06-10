#include "TileWrapper_cpu_taskfn_0.h"

#include <Milhoja_Runtime.h>
#include <Milhoja_RuntimeBackend.h>
#ifdef DEBUG_RUNTIME
#include <Milhoja_Logger.h>
#endif

void*  Tile_cpu_taskfn_0::scratch_Hydro_cvol_fake_ = nullptr;
void*  Tile_cpu_taskfn_0::scratch_Hydro_fareaX_fake_ = nullptr;
void*  Tile_cpu_taskfn_0::scratch_Hydro_fareaY_fake_ = nullptr;
void*  Tile_cpu_taskfn_0::scratch_Hydro_fareaZ_fake_ = nullptr;
void*  Tile_cpu_taskfn_0::scratch_Hydro_fluxBufZ_ = nullptr;
void*  Tile_cpu_taskfn_0::scratch_Hydro_hy_Vc_ = nullptr;
void*  Tile_cpu_taskfn_0::scratch_Hydro_hy_flat3d_ = nullptr;
void*  Tile_cpu_taskfn_0::scratch_Hydro_hy_flux_ = nullptr;
void*  Tile_cpu_taskfn_0::scratch_Hydro_hy_flx_ = nullptr;
void*  Tile_cpu_taskfn_0::scratch_Hydro_hy_fly_ = nullptr;
void*  Tile_cpu_taskfn_0::scratch_Hydro_hy_flz_ = nullptr;
void*  Tile_cpu_taskfn_0::scratch_Hydro_hy_grav_ = nullptr;
void*  Tile_cpu_taskfn_0::scratch_Hydro_hy_rope_ = nullptr;
void*  Tile_cpu_taskfn_0::scratch_Hydro_hy_starState_ = nullptr;
void*  Tile_cpu_taskfn_0::scratch_Hydro_hy_tmpState_ = nullptr;
void*  Tile_cpu_taskfn_0::scratch_Hydro_hy_uMinus_ = nullptr;
void*  Tile_cpu_taskfn_0::scratch_Hydro_hy_uPlus_ = nullptr;
void*  Tile_cpu_taskfn_0::scratch_Hydro_xCenter_fake_ = nullptr;
void*  Tile_cpu_taskfn_0::scratch_Hydro_xLeft_fake_ = nullptr;
void*  Tile_cpu_taskfn_0::scratch_Hydro_xRight_fake_ = nullptr;
void*  Tile_cpu_taskfn_0::scratch_Hydro_yCenter_fake_ = nullptr;
void*  Tile_cpu_taskfn_0::scratch_Hydro_yLeft_fake_ = nullptr;
void*  Tile_cpu_taskfn_0::scratch_Hydro_yRight_fake_ = nullptr;
void*  Tile_cpu_taskfn_0::scratch_Hydro_zCenter_fake_ = nullptr;

void Tile_cpu_taskfn_0::acquireScratch(void) {
   const unsigned int  nThreads = milhoja::Runtime::instance().nMaxThreadsPerTeam();

   if (scratch_Hydro_cvol_fake_) {
      throw std::logic_error("[Tile_cpu_taskfn_0::acquireScratch] scratch_Hydro_cvol_fake_ scratch already allocated");
   }

   const std::size_t nBytes_scratch_Hydro_cvol_fake = nThreads
               * Tile_cpu_taskfn_0::SCRATCH_HYDRO_CVOL_FAKE_SIZE_
               * sizeof(milhoja::Real);

   milhoja::RuntimeBackend::instance().requestCpuMemory(nBytes_scratch_Hydro_cvol_fake, &scratch_Hydro_cvol_fake_);

#ifdef DEBUG_RUNTIME
   std::string   msg = "[Tile_cpu_taskfn_0::acquireScratch] Acquired"
               + std::to_string(nThreads)
               + " scratch_Hydro_cvol_fake_ scratch blocks"
   milhoja::Logger::instance().log(msg);
#endif
   if (scratch_Hydro_fareaX_fake_) {
      throw std::logic_error("[Tile_cpu_taskfn_0::acquireScratch] scratch_Hydro_fareaX_fake_ scratch already allocated");
   }

   const std::size_t nBytes_scratch_Hydro_fareaX_fake = nThreads
               * Tile_cpu_taskfn_0::SCRATCH_HYDRO_FAREAX_FAKE_SIZE_
               * sizeof(milhoja::Real);

   milhoja::RuntimeBackend::instance().requestCpuMemory(nBytes_scratch_Hydro_fareaX_fake, &scratch_Hydro_fareaX_fake_);

#ifdef DEBUG_RUNTIME
   std::string   msg = "[Tile_cpu_taskfn_0::acquireScratch] Acquired"
               + std::to_string(nThreads)
               + " scratch_Hydro_fareaX_fake_ scratch blocks"
   milhoja::Logger::instance().log(msg);
#endif
   if (scratch_Hydro_fareaY_fake_) {
      throw std::logic_error("[Tile_cpu_taskfn_0::acquireScratch] scratch_Hydro_fareaY_fake_ scratch already allocated");
   }

   const std::size_t nBytes_scratch_Hydro_fareaY_fake = nThreads
               * Tile_cpu_taskfn_0::SCRATCH_HYDRO_FAREAY_FAKE_SIZE_
               * sizeof(milhoja::Real);

   milhoja::RuntimeBackend::instance().requestCpuMemory(nBytes_scratch_Hydro_fareaY_fake, &scratch_Hydro_fareaY_fake_);

#ifdef DEBUG_RUNTIME
   std::string   msg = "[Tile_cpu_taskfn_0::acquireScratch] Acquired"
               + std::to_string(nThreads)
               + " scratch_Hydro_fareaY_fake_ scratch blocks"
   milhoja::Logger::instance().log(msg);
#endif
   if (scratch_Hydro_fareaZ_fake_) {
      throw std::logic_error("[Tile_cpu_taskfn_0::acquireScratch] scratch_Hydro_fareaZ_fake_ scratch already allocated");
   }

   const std::size_t nBytes_scratch_Hydro_fareaZ_fake = nThreads
               * Tile_cpu_taskfn_0::SCRATCH_HYDRO_FAREAZ_FAKE_SIZE_
               * sizeof(milhoja::Real);

   milhoja::RuntimeBackend::instance().requestCpuMemory(nBytes_scratch_Hydro_fareaZ_fake, &scratch_Hydro_fareaZ_fake_);

#ifdef DEBUG_RUNTIME
   std::string   msg = "[Tile_cpu_taskfn_0::acquireScratch] Acquired"
               + std::to_string(nThreads)
               + " scratch_Hydro_fareaZ_fake_ scratch blocks"
   milhoja::Logger::instance().log(msg);
#endif
   if (scratch_Hydro_fluxBufZ_) {
      throw std::logic_error("[Tile_cpu_taskfn_0::acquireScratch] scratch_Hydro_fluxBufZ_ scratch already allocated");
   }

   const std::size_t nBytes_scratch_Hydro_fluxBufZ = nThreads
               * Tile_cpu_taskfn_0::SCRATCH_HYDRO_FLUXBUFZ_SIZE_
               * sizeof(milhoja::Real);

   milhoja::RuntimeBackend::instance().requestCpuMemory(nBytes_scratch_Hydro_fluxBufZ, &scratch_Hydro_fluxBufZ_);

#ifdef DEBUG_RUNTIME
   std::string   msg = "[Tile_cpu_taskfn_0::acquireScratch] Acquired"
               + std::to_string(nThreads)
               + " scratch_Hydro_fluxBufZ_ scratch blocks"
   milhoja::Logger::instance().log(msg);
#endif
   if (scratch_Hydro_hy_Vc_) {
      throw std::logic_error("[Tile_cpu_taskfn_0::acquireScratch] scratch_Hydro_hy_Vc_ scratch already allocated");
   }

   const std::size_t nBytes_scratch_Hydro_hy_Vc = nThreads
               * Tile_cpu_taskfn_0::SCRATCH_HYDRO_HY_VC_SIZE_
               * sizeof(milhoja::Real);

   milhoja::RuntimeBackend::instance().requestCpuMemory(nBytes_scratch_Hydro_hy_Vc, &scratch_Hydro_hy_Vc_);

#ifdef DEBUG_RUNTIME
   std::string   msg = "[Tile_cpu_taskfn_0::acquireScratch] Acquired"
               + std::to_string(nThreads)
               + " scratch_Hydro_hy_Vc_ scratch blocks"
   milhoja::Logger::instance().log(msg);
#endif
   if (scratch_Hydro_hy_flat3d_) {
      throw std::logic_error("[Tile_cpu_taskfn_0::acquireScratch] scratch_Hydro_hy_flat3d_ scratch already allocated");
   }

   const std::size_t nBytes_scratch_Hydro_hy_flat3d = nThreads
               * Tile_cpu_taskfn_0::SCRATCH_HYDRO_HY_FLAT3D_SIZE_
               * sizeof(milhoja::Real);

   milhoja::RuntimeBackend::instance().requestCpuMemory(nBytes_scratch_Hydro_hy_flat3d, &scratch_Hydro_hy_flat3d_);

#ifdef DEBUG_RUNTIME
   std::string   msg = "[Tile_cpu_taskfn_0::acquireScratch] Acquired"
               + std::to_string(nThreads)
               + " scratch_Hydro_hy_flat3d_ scratch blocks"
   milhoja::Logger::instance().log(msg);
#endif
   if (scratch_Hydro_hy_flux_) {
      throw std::logic_error("[Tile_cpu_taskfn_0::acquireScratch] scratch_Hydro_hy_flux_ scratch already allocated");
   }

   const std::size_t nBytes_scratch_Hydro_hy_flux = nThreads
               * Tile_cpu_taskfn_0::SCRATCH_HYDRO_HY_FLUX_SIZE_
               * sizeof(milhoja::Real);

   milhoja::RuntimeBackend::instance().requestCpuMemory(nBytes_scratch_Hydro_hy_flux, &scratch_Hydro_hy_flux_);

#ifdef DEBUG_RUNTIME
   std::string   msg = "[Tile_cpu_taskfn_0::acquireScratch] Acquired"
               + std::to_string(nThreads)
               + " scratch_Hydro_hy_flux_ scratch blocks"
   milhoja::Logger::instance().log(msg);
#endif
   if (scratch_Hydro_hy_flx_) {
      throw std::logic_error("[Tile_cpu_taskfn_0::acquireScratch] scratch_Hydro_hy_flx_ scratch already allocated");
   }

   const std::size_t nBytes_scratch_Hydro_hy_flx = nThreads
               * Tile_cpu_taskfn_0::SCRATCH_HYDRO_HY_FLX_SIZE_
               * sizeof(milhoja::Real);

   milhoja::RuntimeBackend::instance().requestCpuMemory(nBytes_scratch_Hydro_hy_flx, &scratch_Hydro_hy_flx_);

#ifdef DEBUG_RUNTIME
   std::string   msg = "[Tile_cpu_taskfn_0::acquireScratch] Acquired"
               + std::to_string(nThreads)
               + " scratch_Hydro_hy_flx_ scratch blocks"
   milhoja::Logger::instance().log(msg);
#endif
   if (scratch_Hydro_hy_fly_) {
      throw std::logic_error("[Tile_cpu_taskfn_0::acquireScratch] scratch_Hydro_hy_fly_ scratch already allocated");
   }

   const std::size_t nBytes_scratch_Hydro_hy_fly = nThreads
               * Tile_cpu_taskfn_0::SCRATCH_HYDRO_HY_FLY_SIZE_
               * sizeof(milhoja::Real);

   milhoja::RuntimeBackend::instance().requestCpuMemory(nBytes_scratch_Hydro_hy_fly, &scratch_Hydro_hy_fly_);

#ifdef DEBUG_RUNTIME
   std::string   msg = "[Tile_cpu_taskfn_0::acquireScratch] Acquired"
               + std::to_string(nThreads)
               + " scratch_Hydro_hy_fly_ scratch blocks"
   milhoja::Logger::instance().log(msg);
#endif
   if (scratch_Hydro_hy_flz_) {
      throw std::logic_error("[Tile_cpu_taskfn_0::acquireScratch] scratch_Hydro_hy_flz_ scratch already allocated");
   }

   const std::size_t nBytes_scratch_Hydro_hy_flz = nThreads
               * Tile_cpu_taskfn_0::SCRATCH_HYDRO_HY_FLZ_SIZE_
               * sizeof(milhoja::Real);

   milhoja::RuntimeBackend::instance().requestCpuMemory(nBytes_scratch_Hydro_hy_flz, &scratch_Hydro_hy_flz_);

#ifdef DEBUG_RUNTIME
   std::string   msg = "[Tile_cpu_taskfn_0::acquireScratch] Acquired"
               + std::to_string(nThreads)
               + " scratch_Hydro_hy_flz_ scratch blocks"
   milhoja::Logger::instance().log(msg);
#endif
   if (scratch_Hydro_hy_grav_) {
      throw std::logic_error("[Tile_cpu_taskfn_0::acquireScratch] scratch_Hydro_hy_grav_ scratch already allocated");
   }

   const std::size_t nBytes_scratch_Hydro_hy_grav = nThreads
               * Tile_cpu_taskfn_0::SCRATCH_HYDRO_HY_GRAV_SIZE_
               * sizeof(milhoja::Real);

   milhoja::RuntimeBackend::instance().requestCpuMemory(nBytes_scratch_Hydro_hy_grav, &scratch_Hydro_hy_grav_);

#ifdef DEBUG_RUNTIME
   std::string   msg = "[Tile_cpu_taskfn_0::acquireScratch] Acquired"
               + std::to_string(nThreads)
               + " scratch_Hydro_hy_grav_ scratch blocks"
   milhoja::Logger::instance().log(msg);
#endif
   if (scratch_Hydro_hy_rope_) {
      throw std::logic_error("[Tile_cpu_taskfn_0::acquireScratch] scratch_Hydro_hy_rope_ scratch already allocated");
   }

   const std::size_t nBytes_scratch_Hydro_hy_rope = nThreads
               * Tile_cpu_taskfn_0::SCRATCH_HYDRO_HY_ROPE_SIZE_
               * sizeof(milhoja::Real);

   milhoja::RuntimeBackend::instance().requestCpuMemory(nBytes_scratch_Hydro_hy_rope, &scratch_Hydro_hy_rope_);

#ifdef DEBUG_RUNTIME
   std::string   msg = "[Tile_cpu_taskfn_0::acquireScratch] Acquired"
               + std::to_string(nThreads)
               + " scratch_Hydro_hy_rope_ scratch blocks"
   milhoja::Logger::instance().log(msg);
#endif
   if (scratch_Hydro_hy_starState_) {
      throw std::logic_error("[Tile_cpu_taskfn_0::acquireScratch] scratch_Hydro_hy_starState_ scratch already allocated");
   }

   const std::size_t nBytes_scratch_Hydro_hy_starState = nThreads
               * Tile_cpu_taskfn_0::SCRATCH_HYDRO_HY_STARSTATE_SIZE_
               * sizeof(milhoja::Real);

   milhoja::RuntimeBackend::instance().requestCpuMemory(nBytes_scratch_Hydro_hy_starState, &scratch_Hydro_hy_starState_);

#ifdef DEBUG_RUNTIME
   std::string   msg = "[Tile_cpu_taskfn_0::acquireScratch] Acquired"
               + std::to_string(nThreads)
               + " scratch_Hydro_hy_starState_ scratch blocks"
   milhoja::Logger::instance().log(msg);
#endif
   if (scratch_Hydro_hy_tmpState_) {
      throw std::logic_error("[Tile_cpu_taskfn_0::acquireScratch] scratch_Hydro_hy_tmpState_ scratch already allocated");
   }

   const std::size_t nBytes_scratch_Hydro_hy_tmpState = nThreads
               * Tile_cpu_taskfn_0::SCRATCH_HYDRO_HY_TMPSTATE_SIZE_
               * sizeof(milhoja::Real);

   milhoja::RuntimeBackend::instance().requestCpuMemory(nBytes_scratch_Hydro_hy_tmpState, &scratch_Hydro_hy_tmpState_);

#ifdef DEBUG_RUNTIME
   std::string   msg = "[Tile_cpu_taskfn_0::acquireScratch] Acquired"
               + std::to_string(nThreads)
               + " scratch_Hydro_hy_tmpState_ scratch blocks"
   milhoja::Logger::instance().log(msg);
#endif
   if (scratch_Hydro_hy_uMinus_) {
      throw std::logic_error("[Tile_cpu_taskfn_0::acquireScratch] scratch_Hydro_hy_uMinus_ scratch already allocated");
   }

   const std::size_t nBytes_scratch_Hydro_hy_uMinus = nThreads
               * Tile_cpu_taskfn_0::SCRATCH_HYDRO_HY_UMINUS_SIZE_
               * sizeof(milhoja::Real);

   milhoja::RuntimeBackend::instance().requestCpuMemory(nBytes_scratch_Hydro_hy_uMinus, &scratch_Hydro_hy_uMinus_);

#ifdef DEBUG_RUNTIME
   std::string   msg = "[Tile_cpu_taskfn_0::acquireScratch] Acquired"
               + std::to_string(nThreads)
               + " scratch_Hydro_hy_uMinus_ scratch blocks"
   milhoja::Logger::instance().log(msg);
#endif
   if (scratch_Hydro_hy_uPlus_) {
      throw std::logic_error("[Tile_cpu_taskfn_0::acquireScratch] scratch_Hydro_hy_uPlus_ scratch already allocated");
   }

   const std::size_t nBytes_scratch_Hydro_hy_uPlus = nThreads
               * Tile_cpu_taskfn_0::SCRATCH_HYDRO_HY_UPLUS_SIZE_
               * sizeof(milhoja::Real);

   milhoja::RuntimeBackend::instance().requestCpuMemory(nBytes_scratch_Hydro_hy_uPlus, &scratch_Hydro_hy_uPlus_);

#ifdef DEBUG_RUNTIME
   std::string   msg = "[Tile_cpu_taskfn_0::acquireScratch] Acquired"
               + std::to_string(nThreads)
               + " scratch_Hydro_hy_uPlus_ scratch blocks"
   milhoja::Logger::instance().log(msg);
#endif
   if (scratch_Hydro_xCenter_fake_) {
      throw std::logic_error("[Tile_cpu_taskfn_0::acquireScratch] scratch_Hydro_xCenter_fake_ scratch already allocated");
   }

   const std::size_t nBytes_scratch_Hydro_xCenter_fake = nThreads
               * Tile_cpu_taskfn_0::SCRATCH_HYDRO_XCENTER_FAKE_SIZE_
               * sizeof(milhoja::Real);

   milhoja::RuntimeBackend::instance().requestCpuMemory(nBytes_scratch_Hydro_xCenter_fake, &scratch_Hydro_xCenter_fake_);

#ifdef DEBUG_RUNTIME
   std::string   msg = "[Tile_cpu_taskfn_0::acquireScratch] Acquired"
               + std::to_string(nThreads)
               + " scratch_Hydro_xCenter_fake_ scratch blocks"
   milhoja::Logger::instance().log(msg);
#endif
   if (scratch_Hydro_xLeft_fake_) {
      throw std::logic_error("[Tile_cpu_taskfn_0::acquireScratch] scratch_Hydro_xLeft_fake_ scratch already allocated");
   }

   const std::size_t nBytes_scratch_Hydro_xLeft_fake = nThreads
               * Tile_cpu_taskfn_0::SCRATCH_HYDRO_XLEFT_FAKE_SIZE_
               * sizeof(milhoja::Real);

   milhoja::RuntimeBackend::instance().requestCpuMemory(nBytes_scratch_Hydro_xLeft_fake, &scratch_Hydro_xLeft_fake_);

#ifdef DEBUG_RUNTIME
   std::string   msg = "[Tile_cpu_taskfn_0::acquireScratch] Acquired"
               + std::to_string(nThreads)
               + " scratch_Hydro_xLeft_fake_ scratch blocks"
   milhoja::Logger::instance().log(msg);
#endif
   if (scratch_Hydro_xRight_fake_) {
      throw std::logic_error("[Tile_cpu_taskfn_0::acquireScratch] scratch_Hydro_xRight_fake_ scratch already allocated");
   }

   const std::size_t nBytes_scratch_Hydro_xRight_fake = nThreads
               * Tile_cpu_taskfn_0::SCRATCH_HYDRO_XRIGHT_FAKE_SIZE_
               * sizeof(milhoja::Real);

   milhoja::RuntimeBackend::instance().requestCpuMemory(nBytes_scratch_Hydro_xRight_fake, &scratch_Hydro_xRight_fake_);

#ifdef DEBUG_RUNTIME
   std::string   msg = "[Tile_cpu_taskfn_0::acquireScratch] Acquired"
               + std::to_string(nThreads)
               + " scratch_Hydro_xRight_fake_ scratch blocks"
   milhoja::Logger::instance().log(msg);
#endif
   if (scratch_Hydro_yCenter_fake_) {
      throw std::logic_error("[Tile_cpu_taskfn_0::acquireScratch] scratch_Hydro_yCenter_fake_ scratch already allocated");
   }

   const std::size_t nBytes_scratch_Hydro_yCenter_fake = nThreads
               * Tile_cpu_taskfn_0::SCRATCH_HYDRO_YCENTER_FAKE_SIZE_
               * sizeof(milhoja::Real);

   milhoja::RuntimeBackend::instance().requestCpuMemory(nBytes_scratch_Hydro_yCenter_fake, &scratch_Hydro_yCenter_fake_);

#ifdef DEBUG_RUNTIME
   std::string   msg = "[Tile_cpu_taskfn_0::acquireScratch] Acquired"
               + std::to_string(nThreads)
               + " scratch_Hydro_yCenter_fake_ scratch blocks"
   milhoja::Logger::instance().log(msg);
#endif
   if (scratch_Hydro_yLeft_fake_) {
      throw std::logic_error("[Tile_cpu_taskfn_0::acquireScratch] scratch_Hydro_yLeft_fake_ scratch already allocated");
   }

   const std::size_t nBytes_scratch_Hydro_yLeft_fake = nThreads
               * Tile_cpu_taskfn_0::SCRATCH_HYDRO_YLEFT_FAKE_SIZE_
               * sizeof(milhoja::Real);

   milhoja::RuntimeBackend::instance().requestCpuMemory(nBytes_scratch_Hydro_yLeft_fake, &scratch_Hydro_yLeft_fake_);

#ifdef DEBUG_RUNTIME
   std::string   msg = "[Tile_cpu_taskfn_0::acquireScratch] Acquired"
               + std::to_string(nThreads)
               + " scratch_Hydro_yLeft_fake_ scratch blocks"
   milhoja::Logger::instance().log(msg);
#endif
   if (scratch_Hydro_yRight_fake_) {
      throw std::logic_error("[Tile_cpu_taskfn_0::acquireScratch] scratch_Hydro_yRight_fake_ scratch already allocated");
   }

   const std::size_t nBytes_scratch_Hydro_yRight_fake = nThreads
               * Tile_cpu_taskfn_0::SCRATCH_HYDRO_YRIGHT_FAKE_SIZE_
               * sizeof(milhoja::Real);

   milhoja::RuntimeBackend::instance().requestCpuMemory(nBytes_scratch_Hydro_yRight_fake, &scratch_Hydro_yRight_fake_);

#ifdef DEBUG_RUNTIME
   std::string   msg = "[Tile_cpu_taskfn_0::acquireScratch] Acquired"
               + std::to_string(nThreads)
               + " scratch_Hydro_yRight_fake_ scratch blocks"
   milhoja::Logger::instance().log(msg);
#endif
   if (scratch_Hydro_zCenter_fake_) {
      throw std::logic_error("[Tile_cpu_taskfn_0::acquireScratch] scratch_Hydro_zCenter_fake_ scratch already allocated");
   }

   const std::size_t nBytes_scratch_Hydro_zCenter_fake = nThreads
               * Tile_cpu_taskfn_0::SCRATCH_HYDRO_ZCENTER_FAKE_SIZE_
               * sizeof(milhoja::Real);

   milhoja::RuntimeBackend::instance().requestCpuMemory(nBytes_scratch_Hydro_zCenter_fake, &scratch_Hydro_zCenter_fake_);

#ifdef DEBUG_RUNTIME
   std::string   msg = "[Tile_cpu_taskfn_0::acquireScratch] Acquired"
               + std::to_string(nThreads)
               + " scratch_Hydro_zCenter_fake_ scratch blocks"
   milhoja::Logger::instance().log(msg);
#endif
}

void Tile_cpu_taskfn_0::releaseScratch(void) {
   if (!scratch_Hydro_cvol_fake_) {
      throw std::logic_error("[Tile_cpu_taskfn_0::releaseScratch] scratch_Hydro_cvol_fake_ scratch not allocated");
   }

   milhoja::RuntimeBackend::instance().releaseCpuMemory(&scratch_Hydro_cvol_fake_);
   scratch_Hydro_cvol_fake_ = nullptr;

#ifdef DEBUG_RUNTIME
   std::string   msg = "[Tile_cpu_taskfn_0::releaseScratch] Released scratch_Hydro_cvol_fake_ scratch"
   milhoja::Logger::instance().log(msg);
#endif
   if (!scratch_Hydro_fareaX_fake_) {
      throw std::logic_error("[Tile_cpu_taskfn_0::releaseScratch] scratch_Hydro_fareaX_fake_ scratch not allocated");
   }

   milhoja::RuntimeBackend::instance().releaseCpuMemory(&scratch_Hydro_fareaX_fake_);
   scratch_Hydro_fareaX_fake_ = nullptr;

#ifdef DEBUG_RUNTIME
   std::string   msg = "[Tile_cpu_taskfn_0::releaseScratch] Released scratch_Hydro_fareaX_fake_ scratch"
   milhoja::Logger::instance().log(msg);
#endif
   if (!scratch_Hydro_fareaY_fake_) {
      throw std::logic_error("[Tile_cpu_taskfn_0::releaseScratch] scratch_Hydro_fareaY_fake_ scratch not allocated");
   }

   milhoja::RuntimeBackend::instance().releaseCpuMemory(&scratch_Hydro_fareaY_fake_);
   scratch_Hydro_fareaY_fake_ = nullptr;

#ifdef DEBUG_RUNTIME
   std::string   msg = "[Tile_cpu_taskfn_0::releaseScratch] Released scratch_Hydro_fareaY_fake_ scratch"
   milhoja::Logger::instance().log(msg);
#endif
   if (!scratch_Hydro_fareaZ_fake_) {
      throw std::logic_error("[Tile_cpu_taskfn_0::releaseScratch] scratch_Hydro_fareaZ_fake_ scratch not allocated");
   }

   milhoja::RuntimeBackend::instance().releaseCpuMemory(&scratch_Hydro_fareaZ_fake_);
   scratch_Hydro_fareaZ_fake_ = nullptr;

#ifdef DEBUG_RUNTIME
   std::string   msg = "[Tile_cpu_taskfn_0::releaseScratch] Released scratch_Hydro_fareaZ_fake_ scratch"
   milhoja::Logger::instance().log(msg);
#endif
   if (!scratch_Hydro_fluxBufZ_) {
      throw std::logic_error("[Tile_cpu_taskfn_0::releaseScratch] scratch_Hydro_fluxBufZ_ scratch not allocated");
   }

   milhoja::RuntimeBackend::instance().releaseCpuMemory(&scratch_Hydro_fluxBufZ_);
   scratch_Hydro_fluxBufZ_ = nullptr;

#ifdef DEBUG_RUNTIME
   std::string   msg = "[Tile_cpu_taskfn_0::releaseScratch] Released scratch_Hydro_fluxBufZ_ scratch"
   milhoja::Logger::instance().log(msg);
#endif
   if (!scratch_Hydro_hy_Vc_) {
      throw std::logic_error("[Tile_cpu_taskfn_0::releaseScratch] scratch_Hydro_hy_Vc_ scratch not allocated");
   }

   milhoja::RuntimeBackend::instance().releaseCpuMemory(&scratch_Hydro_hy_Vc_);
   scratch_Hydro_hy_Vc_ = nullptr;

#ifdef DEBUG_RUNTIME
   std::string   msg = "[Tile_cpu_taskfn_0::releaseScratch] Released scratch_Hydro_hy_Vc_ scratch"
   milhoja::Logger::instance().log(msg);
#endif
   if (!scratch_Hydro_hy_flat3d_) {
      throw std::logic_error("[Tile_cpu_taskfn_0::releaseScratch] scratch_Hydro_hy_flat3d_ scratch not allocated");
   }

   milhoja::RuntimeBackend::instance().releaseCpuMemory(&scratch_Hydro_hy_flat3d_);
   scratch_Hydro_hy_flat3d_ = nullptr;

#ifdef DEBUG_RUNTIME
   std::string   msg = "[Tile_cpu_taskfn_0::releaseScratch] Released scratch_Hydro_hy_flat3d_ scratch"
   milhoja::Logger::instance().log(msg);
#endif
   if (!scratch_Hydro_hy_flux_) {
      throw std::logic_error("[Tile_cpu_taskfn_0::releaseScratch] scratch_Hydro_hy_flux_ scratch not allocated");
   }

   milhoja::RuntimeBackend::instance().releaseCpuMemory(&scratch_Hydro_hy_flux_);
   scratch_Hydro_hy_flux_ = nullptr;

#ifdef DEBUG_RUNTIME
   std::string   msg = "[Tile_cpu_taskfn_0::releaseScratch] Released scratch_Hydro_hy_flux_ scratch"
   milhoja::Logger::instance().log(msg);
#endif
   if (!scratch_Hydro_hy_flx_) {
      throw std::logic_error("[Tile_cpu_taskfn_0::releaseScratch] scratch_Hydro_hy_flx_ scratch not allocated");
   }

   milhoja::RuntimeBackend::instance().releaseCpuMemory(&scratch_Hydro_hy_flx_);
   scratch_Hydro_hy_flx_ = nullptr;

#ifdef DEBUG_RUNTIME
   std::string   msg = "[Tile_cpu_taskfn_0::releaseScratch] Released scratch_Hydro_hy_flx_ scratch"
   milhoja::Logger::instance().log(msg);
#endif
   if (!scratch_Hydro_hy_fly_) {
      throw std::logic_error("[Tile_cpu_taskfn_0::releaseScratch] scratch_Hydro_hy_fly_ scratch not allocated");
   }

   milhoja::RuntimeBackend::instance().releaseCpuMemory(&scratch_Hydro_hy_fly_);
   scratch_Hydro_hy_fly_ = nullptr;

#ifdef DEBUG_RUNTIME
   std::string   msg = "[Tile_cpu_taskfn_0::releaseScratch] Released scratch_Hydro_hy_fly_ scratch"
   milhoja::Logger::instance().log(msg);
#endif
   if (!scratch_Hydro_hy_flz_) {
      throw std::logic_error("[Tile_cpu_taskfn_0::releaseScratch] scratch_Hydro_hy_flz_ scratch not allocated");
   }

   milhoja::RuntimeBackend::instance().releaseCpuMemory(&scratch_Hydro_hy_flz_);
   scratch_Hydro_hy_flz_ = nullptr;

#ifdef DEBUG_RUNTIME
   std::string   msg = "[Tile_cpu_taskfn_0::releaseScratch] Released scratch_Hydro_hy_flz_ scratch"
   milhoja::Logger::instance().log(msg);
#endif
   if (!scratch_Hydro_hy_grav_) {
      throw std::logic_error("[Tile_cpu_taskfn_0::releaseScratch] scratch_Hydro_hy_grav_ scratch not allocated");
   }

   milhoja::RuntimeBackend::instance().releaseCpuMemory(&scratch_Hydro_hy_grav_);
   scratch_Hydro_hy_grav_ = nullptr;

#ifdef DEBUG_RUNTIME
   std::string   msg = "[Tile_cpu_taskfn_0::releaseScratch] Released scratch_Hydro_hy_grav_ scratch"
   milhoja::Logger::instance().log(msg);
#endif
   if (!scratch_Hydro_hy_rope_) {
      throw std::logic_error("[Tile_cpu_taskfn_0::releaseScratch] scratch_Hydro_hy_rope_ scratch not allocated");
   }

   milhoja::RuntimeBackend::instance().releaseCpuMemory(&scratch_Hydro_hy_rope_);
   scratch_Hydro_hy_rope_ = nullptr;

#ifdef DEBUG_RUNTIME
   std::string   msg = "[Tile_cpu_taskfn_0::releaseScratch] Released scratch_Hydro_hy_rope_ scratch"
   milhoja::Logger::instance().log(msg);
#endif
   if (!scratch_Hydro_hy_starState_) {
      throw std::logic_error("[Tile_cpu_taskfn_0::releaseScratch] scratch_Hydro_hy_starState_ scratch not allocated");
   }

   milhoja::RuntimeBackend::instance().releaseCpuMemory(&scratch_Hydro_hy_starState_);
   scratch_Hydro_hy_starState_ = nullptr;

#ifdef DEBUG_RUNTIME
   std::string   msg = "[Tile_cpu_taskfn_0::releaseScratch] Released scratch_Hydro_hy_starState_ scratch"
   milhoja::Logger::instance().log(msg);
#endif
   if (!scratch_Hydro_hy_tmpState_) {
      throw std::logic_error("[Tile_cpu_taskfn_0::releaseScratch] scratch_Hydro_hy_tmpState_ scratch not allocated");
   }

   milhoja::RuntimeBackend::instance().releaseCpuMemory(&scratch_Hydro_hy_tmpState_);
   scratch_Hydro_hy_tmpState_ = nullptr;

#ifdef DEBUG_RUNTIME
   std::string   msg = "[Tile_cpu_taskfn_0::releaseScratch] Released scratch_Hydro_hy_tmpState_ scratch"
   milhoja::Logger::instance().log(msg);
#endif
   if (!scratch_Hydro_hy_uMinus_) {
      throw std::logic_error("[Tile_cpu_taskfn_0::releaseScratch] scratch_Hydro_hy_uMinus_ scratch not allocated");
   }

   milhoja::RuntimeBackend::instance().releaseCpuMemory(&scratch_Hydro_hy_uMinus_);
   scratch_Hydro_hy_uMinus_ = nullptr;

#ifdef DEBUG_RUNTIME
   std::string   msg = "[Tile_cpu_taskfn_0::releaseScratch] Released scratch_Hydro_hy_uMinus_ scratch"
   milhoja::Logger::instance().log(msg);
#endif
   if (!scratch_Hydro_hy_uPlus_) {
      throw std::logic_error("[Tile_cpu_taskfn_0::releaseScratch] scratch_Hydro_hy_uPlus_ scratch not allocated");
   }

   milhoja::RuntimeBackend::instance().releaseCpuMemory(&scratch_Hydro_hy_uPlus_);
   scratch_Hydro_hy_uPlus_ = nullptr;

#ifdef DEBUG_RUNTIME
   std::string   msg = "[Tile_cpu_taskfn_0::releaseScratch] Released scratch_Hydro_hy_uPlus_ scratch"
   milhoja::Logger::instance().log(msg);
#endif
   if (!scratch_Hydro_xCenter_fake_) {
      throw std::logic_error("[Tile_cpu_taskfn_0::releaseScratch] scratch_Hydro_xCenter_fake_ scratch not allocated");
   }

   milhoja::RuntimeBackend::instance().releaseCpuMemory(&scratch_Hydro_xCenter_fake_);
   scratch_Hydro_xCenter_fake_ = nullptr;

#ifdef DEBUG_RUNTIME
   std::string   msg = "[Tile_cpu_taskfn_0::releaseScratch] Released scratch_Hydro_xCenter_fake_ scratch"
   milhoja::Logger::instance().log(msg);
#endif
   if (!scratch_Hydro_xLeft_fake_) {
      throw std::logic_error("[Tile_cpu_taskfn_0::releaseScratch] scratch_Hydro_xLeft_fake_ scratch not allocated");
   }

   milhoja::RuntimeBackend::instance().releaseCpuMemory(&scratch_Hydro_xLeft_fake_);
   scratch_Hydro_xLeft_fake_ = nullptr;

#ifdef DEBUG_RUNTIME
   std::string   msg = "[Tile_cpu_taskfn_0::releaseScratch] Released scratch_Hydro_xLeft_fake_ scratch"
   milhoja::Logger::instance().log(msg);
#endif
   if (!scratch_Hydro_xRight_fake_) {
      throw std::logic_error("[Tile_cpu_taskfn_0::releaseScratch] scratch_Hydro_xRight_fake_ scratch not allocated");
   }

   milhoja::RuntimeBackend::instance().releaseCpuMemory(&scratch_Hydro_xRight_fake_);
   scratch_Hydro_xRight_fake_ = nullptr;

#ifdef DEBUG_RUNTIME
   std::string   msg = "[Tile_cpu_taskfn_0::releaseScratch] Released scratch_Hydro_xRight_fake_ scratch"
   milhoja::Logger::instance().log(msg);
#endif
   if (!scratch_Hydro_yCenter_fake_) {
      throw std::logic_error("[Tile_cpu_taskfn_0::releaseScratch] scratch_Hydro_yCenter_fake_ scratch not allocated");
   }

   milhoja::RuntimeBackend::instance().releaseCpuMemory(&scratch_Hydro_yCenter_fake_);
   scratch_Hydro_yCenter_fake_ = nullptr;

#ifdef DEBUG_RUNTIME
   std::string   msg = "[Tile_cpu_taskfn_0::releaseScratch] Released scratch_Hydro_yCenter_fake_ scratch"
   milhoja::Logger::instance().log(msg);
#endif
   if (!scratch_Hydro_yLeft_fake_) {
      throw std::logic_error("[Tile_cpu_taskfn_0::releaseScratch] scratch_Hydro_yLeft_fake_ scratch not allocated");
   }

   milhoja::RuntimeBackend::instance().releaseCpuMemory(&scratch_Hydro_yLeft_fake_);
   scratch_Hydro_yLeft_fake_ = nullptr;

#ifdef DEBUG_RUNTIME
   std::string   msg = "[Tile_cpu_taskfn_0::releaseScratch] Released scratch_Hydro_yLeft_fake_ scratch"
   milhoja::Logger::instance().log(msg);
#endif
   if (!scratch_Hydro_yRight_fake_) {
      throw std::logic_error("[Tile_cpu_taskfn_0::releaseScratch] scratch_Hydro_yRight_fake_ scratch not allocated");
   }

   milhoja::RuntimeBackend::instance().releaseCpuMemory(&scratch_Hydro_yRight_fake_);
   scratch_Hydro_yRight_fake_ = nullptr;

#ifdef DEBUG_RUNTIME
   std::string   msg = "[Tile_cpu_taskfn_0::releaseScratch] Released scratch_Hydro_yRight_fake_ scratch"
   milhoja::Logger::instance().log(msg);
#endif
   if (!scratch_Hydro_zCenter_fake_) {
      throw std::logic_error("[Tile_cpu_taskfn_0::releaseScratch] scratch_Hydro_zCenter_fake_ scratch not allocated");
   }

   milhoja::RuntimeBackend::instance().releaseCpuMemory(&scratch_Hydro_zCenter_fake_);
   scratch_Hydro_zCenter_fake_ = nullptr;

#ifdef DEBUG_RUNTIME
   std::string   msg = "[Tile_cpu_taskfn_0::releaseScratch] Released scratch_Hydro_zCenter_fake_ scratch"
   milhoja::Logger::instance().log(msg);
#endif
}

Tile_cpu_taskfn_0::Tile_cpu_taskfn_0(
         const milhoja::Real external_Hydro_dt,
         const milhoja::Real external_Hydro_dtOld,
         const int external_Hydro_stage
)
   : milhoja::TileWrapper{},
     external_Hydro_dt_{external_Hydro_dt},
     external_Hydro_dtOld_{external_Hydro_dtOld},
     external_Hydro_stage_{external_Hydro_stage}
{
#ifdef DEBUG_RUNTIME
   std::string   msg = "[Tile_cpu_taskfn_0] Creating wrapper object";
   milhoja::Logger::instance().log(msg);
#endif
}

Tile_cpu_taskfn_0::~Tile_cpu_taskfn_0(void) {
#ifdef DEBUG_RUNTIME
   std::string   msg = "[~Tile_cpu_taskfn_0] Destroying wrapper object";
   milhoja::Logger::instance().log(msg);
#endif
}

std::unique_ptr<milhoja::TileWrapper> Tile_cpu_taskfn_0::clone(std::shared_ptr<milhoja::Tile>&& tileToWrap) const {
   Tile_cpu_taskfn_0* ptr = new Tile_cpu_taskfn_0{
               external_Hydro_dt_,
               external_Hydro_dtOld_,
               external_Hydro_stage_};

   if (ptr->tile_) {
      throw std::logic_error("[Tile_cpu_taskfn_0::clone] Internal tile_ member not null");
   }
   ptr->tile_ = std::move(tileToWrap);
   if (!(ptr->tile_) || tileToWrap) {
      throw std::logic_error("[Tile_cpu_taskfn_0::clone] Wrapper did not take ownership of tile");
   }

   return std::unique_ptr<milhoja::TileWrapper>{ptr};
}
