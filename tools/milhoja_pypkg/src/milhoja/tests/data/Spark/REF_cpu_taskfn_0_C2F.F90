#include "Milhoja.h"

subroutine cpu_taskfn_0_C2F( &
   C_external_Hydro_dt, &
   C_external_Hydro_dtOld, &
   C_external_Hydro_stage, &
   C_tile_arrayBounds, &
   C_tile_deltas, &
   C_tile_interior, &
   C_tile_lbound, &
   C_tile_lo, &
   C_CC_1, &
   C_FLX_1, &
   C_FLY_1, &
   C_scratch_Hydro_cvol_fake, &
   C_scratch_Hydro_fareaX_fake, &
   C_scratch_Hydro_fareaY_fake, &
   C_scratch_Hydro_fareaZ_fake, &
   C_scratch_Hydro_fluxBufZ, &
   C_scratch_Hydro_hy_Vc, &
   C_scratch_Hydro_hy_flat3d, &
   C_scratch_Hydro_hy_flux, &
   C_scratch_Hydro_hy_flx, &
   C_scratch_Hydro_hy_fly, &
   C_scratch_Hydro_hy_flz, &
   C_scratch_Hydro_hy_grav, &
   C_scratch_Hydro_hy_rope, &
   C_scratch_Hydro_hy_starState, &
   C_scratch_Hydro_hy_tmpState, &
   C_scratch_Hydro_hy_uMinus, &
   C_scratch_Hydro_hy_uPlus, &
   C_scratch_Hydro_xCenter_fake, &
   C_scratch_Hydro_xLeft_fake, &
   C_scratch_Hydro_xRight_fake, &
   C_scratch_Hydro_yCenter_fake, &
   C_scratch_Hydro_yLeft_fake, &
   C_scratch_Hydro_yRight_fake, &
   C_scratch_Hydro_zCenter_fake &
)bind(c, name="cpu_taskfn_0_C2F")
   use iso_c_binding, ONLY : C_PTR, C_F_POINTER
   use milhoja_types_mod, ONLY : MILHOJA_INT, MILHOJA_REAL
   use cpu_taskfn_0_mod, ONLY : cpu_taskfn_0_Fortran
   implicit none

   real(MILHOJA_REAL), intent(IN), value :: C_external_Hydro_dt
   real(MILHOJA_REAL), intent(IN), value :: C_external_Hydro_dtOld
   integer(MILHOJA_INT), intent(IN), value :: C_external_Hydro_stage
   type(C_PTR), intent(IN), value :: C_tile_arrayBounds
   type(C_PTR), intent(IN), value :: C_tile_deltas
   type(C_PTR), intent(IN), value :: C_tile_interior
   type(C_PTR), intent(IN), value :: C_tile_lbound
   type(C_PTR), intent(IN), value :: C_tile_lo
   type(C_PTR), intent(IN), value :: C_CC_1
   type(C_PTR), intent(IN), value :: C_FLX_1
   type(C_PTR), intent(IN), value :: C_FLY_1
   type(C_PTR), intent(IN), value :: C_scratch_Hydro_cvol_fake
   type(C_PTR), intent(IN), value :: C_scratch_Hydro_fareaX_fake
   type(C_PTR), intent(IN), value :: C_scratch_Hydro_fareaY_fake
   type(C_PTR), intent(IN), value :: C_scratch_Hydro_fareaZ_fake
   type(C_PTR), intent(IN), value :: C_scratch_Hydro_fluxBufZ
   type(C_PTR), intent(IN), value :: C_scratch_Hydro_hy_Vc
   type(C_PTR), intent(IN), value :: C_scratch_Hydro_hy_flat3d
   type(C_PTR), intent(IN), value :: C_scratch_Hydro_hy_flux
   type(C_PTR), intent(IN), value :: C_scratch_Hydro_hy_flx
   type(C_PTR), intent(IN), value :: C_scratch_Hydro_hy_fly
   type(C_PTR), intent(IN), value :: C_scratch_Hydro_hy_flz
   type(C_PTR), intent(IN), value :: C_scratch_Hydro_hy_grav
   type(C_PTR), intent(IN), value :: C_scratch_Hydro_hy_rope
   type(C_PTR), intent(IN), value :: C_scratch_Hydro_hy_starState
   type(C_PTR), intent(IN), value :: C_scratch_Hydro_hy_tmpState
   type(C_PTR), intent(IN), value :: C_scratch_Hydro_hy_uMinus
   type(C_PTR), intent(IN), value :: C_scratch_Hydro_hy_uPlus
   type(C_PTR), intent(IN), value :: C_scratch_Hydro_xCenter_fake
   type(C_PTR), intent(IN), value :: C_scratch_Hydro_xLeft_fake
   type(C_PTR), intent(IN), value :: C_scratch_Hydro_xRight_fake
   type(C_PTR), intent(IN), value :: C_scratch_Hydro_yCenter_fake
   type(C_PTR), intent(IN), value :: C_scratch_Hydro_yLeft_fake
   type(C_PTR), intent(IN), value :: C_scratch_Hydro_yRight_fake
   type(C_PTR), intent(IN), value :: C_scratch_Hydro_zCenter_fake

   real :: F_external_Hydro_dt
   real :: F_external_Hydro_dtOld
   integer :: F_external_Hydro_stage
   integer, pointer :: F_tile_arrayBounds(:,:)
   real, pointer :: F_tile_deltas(:)
   integer, pointer :: F_tile_interior(:,:)
   integer, pointer :: F_tile_lbound(:)
   integer, pointer :: F_tile_lo(:)
   real, pointer :: F_CC_1(:,:,:,:)
   real, pointer :: F_FLX_1(:,:,:,:)
   real, pointer :: F_FLY_1(:,:,:,:)
   real, pointer :: F_scratch_Hydro_cvol_fake(:,:,:)
   real, pointer :: F_scratch_Hydro_fareaX_fake(:,:,:)
   real, pointer :: F_scratch_Hydro_fareaY_fake(:,:,:)
   real, pointer :: F_scratch_Hydro_fareaZ_fake(:,:,:)
   real, pointer :: F_scratch_Hydro_fluxBufZ(:,:,:,:)
   real, pointer :: F_scratch_Hydro_hy_Vc(:,:,:)
   real, pointer :: F_scratch_Hydro_hy_flat3d(:,:,:)
   real, pointer :: F_scratch_Hydro_hy_flux(:,:,:,:)
   real, pointer :: F_scratch_Hydro_hy_flx(:,:,:,:)
   real, pointer :: F_scratch_Hydro_hy_fly(:,:,:,:)
   real, pointer :: F_scratch_Hydro_hy_flz(:,:,:,:)
   real, pointer :: F_scratch_Hydro_hy_grav(:,:,:,:)
   real, pointer :: F_scratch_Hydro_hy_rope(:,:,:,:)
   real, pointer :: F_scratch_Hydro_hy_starState(:,:,:,:)
   real, pointer :: F_scratch_Hydro_hy_tmpState(:,:,:,:)
   real, pointer :: F_scratch_Hydro_hy_uMinus(:,:,:,:)
   real, pointer :: F_scratch_Hydro_hy_uPlus(:,:,:,:)
   real, pointer :: F_scratch_Hydro_xCenter_fake(:)
   real, pointer :: F_scratch_Hydro_xLeft_fake(:)
   real, pointer :: F_scratch_Hydro_xRight_fake(:)
   real, pointer :: F_scratch_Hydro_yCenter_fake(:)
   real, pointer :: F_scratch_Hydro_yLeft_fake(:)
   real, pointer :: F_scratch_Hydro_yRight_fake(:)
   real, pointer :: F_scratch_Hydro_zCenter_fake(:)

   F_external_Hydro_dt = REAL(C_external_Hydro_dt)
   F_external_Hydro_dtOld = REAL(C_external_Hydro_dtOld)
   F_external_Hydro_stage = INT(C_external_Hydro_stage)
   CALL C_F_POINTER(C_tile_arrayBounds, F_tile_arrayBounds, shape=[2,MILHOJA_MDIM])
   CALL C_F_POINTER(C_tile_deltas, F_tile_deltas, shape=[MILHOJA_MDIM])
   CALL C_F_POINTER(C_tile_interior, F_tile_interior, shape=[2,MILHOJA_MDIM])
   CALL C_F_POINTER(C_tile_lbound, F_tile_lbound, shape=[MILHOJA_MDIM])
   CALL C_F_POINTER(C_tile_lo, F_tile_lo, shape=[MILHOJA_MDIM])
   CALL C_F_POINTER(C_CC_1, F_CC_1, shape=[16 + 2 * 6 * MILHOJA_K1D,16 + 2 * 6 * MILHOJA_K2D,1 + 2 * 6 * MILHOJA_K3D,18])
   CALL C_F_POINTER(C_FLX_1, F_FLX_1, shape=[(16 + 1),16,1,5])
   CALL C_F_POINTER(C_FLY_1, F_FLY_1, shape=[16,(16 + 1),1,5])
   CALL C_F_POINTER(C_scratch_Hydro_cvol_fake, F_scratch_Hydro_cvol_fake, shape=[1,1,1])
   CALL C_F_POINTER(C_scratch_Hydro_fareaX_fake, F_scratch_Hydro_fareaX_fake, shape=[1,1,1])
   CALL C_F_POINTER(C_scratch_Hydro_fareaY_fake, F_scratch_Hydro_fareaY_fake, shape=[1,1,1])
   CALL C_F_POINTER(C_scratch_Hydro_fareaZ_fake, F_scratch_Hydro_fareaZ_fake, shape=[1,1,1])
   CALL C_F_POINTER(C_scratch_Hydro_fluxBufZ, F_scratch_Hydro_fluxBufZ, shape=[1,1,1,1])
   CALL C_F_POINTER(C_scratch_Hydro_hy_Vc, F_scratch_Hydro_hy_Vc, shape=[28,28,1])
   CALL C_F_POINTER(C_scratch_Hydro_hy_flat3d, F_scratch_Hydro_hy_flat3d, shape=[28,28,1])
   CALL C_F_POINTER(C_scratch_Hydro_hy_flux, F_scratch_Hydro_hy_flux, shape=[28,28,1,7])
   CALL C_F_POINTER(C_scratch_Hydro_hy_flx, F_scratch_Hydro_hy_flx, shape=[28,28,1,5])
   CALL C_F_POINTER(C_scratch_Hydro_hy_fly, F_scratch_Hydro_hy_fly, shape=[28,28,1,5])
   CALL C_F_POINTER(C_scratch_Hydro_hy_flz, F_scratch_Hydro_hy_flz, shape=[28,28,1,5])
   CALL C_F_POINTER(C_scratch_Hydro_hy_grav, F_scratch_Hydro_hy_grav, shape=[3,28,28,1])
   CALL C_F_POINTER(C_scratch_Hydro_hy_rope, F_scratch_Hydro_hy_rope, shape=[28,28,1,7])
   CALL C_F_POINTER(C_scratch_Hydro_hy_starState, F_scratch_Hydro_hy_starState, shape=[28,28,1,18])
   CALL C_F_POINTER(C_scratch_Hydro_hy_tmpState, F_scratch_Hydro_hy_tmpState, shape=[28,28,1,18])
   CALL C_F_POINTER(C_scratch_Hydro_hy_uMinus, F_scratch_Hydro_hy_uMinus, shape=[28,28,1,7])
   CALL C_F_POINTER(C_scratch_Hydro_hy_uPlus, F_scratch_Hydro_hy_uPlus, shape=[28,28,1,7])
   CALL C_F_POINTER(C_scratch_Hydro_xCenter_fake, F_scratch_Hydro_xCenter_fake, shape=[1])
   CALL C_F_POINTER(C_scratch_Hydro_xLeft_fake, F_scratch_Hydro_xLeft_fake, shape=[1])
   CALL C_F_POINTER(C_scratch_Hydro_xRight_fake, F_scratch_Hydro_xRight_fake, shape=[1])
   CALL C_F_POINTER(C_scratch_Hydro_yCenter_fake, F_scratch_Hydro_yCenter_fake, shape=[1])
   CALL C_F_POINTER(C_scratch_Hydro_yLeft_fake, F_scratch_Hydro_yLeft_fake, shape=[1])
   CALL C_F_POINTER(C_scratch_Hydro_yRight_fake, F_scratch_Hydro_yRight_fake, shape=[1])
   CALL C_F_POINTER(C_scratch_Hydro_zCenter_fake, F_scratch_Hydro_zCenter_fake, shape=[1])

   CALL cpu_taskfn_0_Fortran( &
      F_external_Hydro_dt, &
      F_external_Hydro_dtOld, &
      F_external_Hydro_stage, &
      F_tile_arrayBounds, &
      F_tile_deltas, &
      F_tile_interior, &
      F_tile_lbound, &
      F_tile_lo, &
      F_CC_1, &
      F_FLX_1, &
      F_FLY_1, &
      F_scratch_Hydro_cvol_fake, &
      F_scratch_Hydro_fareaX_fake, &
      F_scratch_Hydro_fareaY_fake, &
      F_scratch_Hydro_fareaZ_fake, &
      F_scratch_Hydro_fluxBufZ, &
      F_scratch_Hydro_hy_Vc, &
      F_scratch_Hydro_hy_flat3d, &
      F_scratch_Hydro_hy_flux, &
      F_scratch_Hydro_hy_flx, &
      F_scratch_Hydro_hy_fly, &
      F_scratch_Hydro_hy_flz, &
      F_scratch_Hydro_hy_grav, &
      F_scratch_Hydro_hy_rope, &
      F_scratch_Hydro_hy_starState, &
      F_scratch_Hydro_hy_tmpState, &
      F_scratch_Hydro_hy_uMinus, &
      F_scratch_Hydro_hy_uPlus, &
      F_scratch_Hydro_xCenter_fake, &
      F_scratch_Hydro_xLeft_fake, &
      F_scratch_Hydro_xRight_fake, &
      F_scratch_Hydro_yCenter_fake, &
      F_scratch_Hydro_yLeft_fake, &
      F_scratch_Hydro_yRight_fake, &
      F_scratch_Hydro_zCenter_fake)
end subroutine cpu_taskfn_0_C2F
