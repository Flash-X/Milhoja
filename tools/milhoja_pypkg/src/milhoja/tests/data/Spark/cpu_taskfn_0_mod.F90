#include "Milhoja.h"

module cpu_taskfn_0_mod
   implicit none
   private

   public :: cpu_taskfn_0_Fortran
   public :: cpu_taskfn_0_Cpp2C

   interface
      subroutine cpu_taskfn_0_Cpp2C(C_threadIndex, C_dataItemPtr) &
      bind(c, name="cpu_taskfn_0_Cpp2C")
         use iso_c_binding, ONLY : C_PTR
         use milhoja_types_mod, ONLY : MILHOJA_INT
         integer(MILHOJA_INT), intent(IN), value :: C_threadIndex
         type(C_PTR), intent(IN), value :: C_dataItemPtr
      end subroutine cpu_taskfn_0_Cpp2C
   end interface

contains

   subroutine cpu_taskfn_0_Fortran( &
      external_Hydro_dt, &
      external_Hydro_dtOld, &
      external_Hydro_stage, &
      tile_arrayBounds, &
      tile_deltas, &
      tile_interior, &
      tile_lbound, &
      tile_lo, &
      CC_1, &
      scratch_Hydro_cvol_fake, &
      scratch_Hydro_fareaX_fake, &
      scratch_Hydro_fareaY_fake, &
      scratch_Hydro_fareaZ_fake, &
      scratch_Hydro_hy_Vc, &
      scratch_Hydro_hy_flat3d, &
      scratch_Hydro_hy_flux, &
      scratch_Hydro_hy_fluxBufX, &
      scratch_Hydro_hy_fluxBufY, &
      scratch_Hydro_hy_fluxBufZ, &
      scratch_Hydro_hy_flx, &
      scratch_Hydro_hy_fly, &
      scratch_Hydro_hy_flz, &
      scratch_Hydro_hy_grav, &
      scratch_Hydro_hy_rope, &
      scratch_Hydro_hy_starState, &
      scratch_Hydro_hy_tmpState, &
      scratch_Hydro_hy_uMinus, &
      scratch_Hydro_hy_uPlus, &
      scratch_Hydro_xCenter_fake, &
      scratch_Hydro_xLeft_fake, &
      scratch_Hydro_xRight_fake, &
      scratch_Hydro_yCenter_fake, &
      scratch_Hydro_yLeft_fake, &
      scratch_Hydro_yRight_fake, &
      scratch_Hydro_zCenter_fake &
   )
      use iso_c_binding, ONLY : C_PTR
      use Hydro_interface, ONLY : Hydro_prepBlock
      use Hydro_interface, ONLY : Hydro_advance


      implicit none

      real, intent(IN) :: external_Hydro_dt
      real, intent(IN) :: external_Hydro_dtOld
      integer, intent(IN) :: external_Hydro_stage
      integer, intent(IN) :: tile_arrayBounds(:, :)
      real, intent(IN) :: tile_deltas(:)
      integer, intent(IN) :: tile_interior(:, :)
      integer, intent(IN) :: tile_lbound(:)
      integer, intent(IN) :: tile_lo(:)
      real, intent(INOUT) :: CC_1(:, :, :, :)
      real, intent(IN) :: scratch_Hydro_cvol_fake(:, :, :)
      real, intent(IN) :: scratch_Hydro_fareaX_fake(:, :, :)
      real, intent(IN) :: scratch_Hydro_fareaY_fake(:, :, :)
      real, intent(IN) :: scratch_Hydro_fareaZ_fake(:, :, :)
      real, intent(IN) :: scratch_Hydro_hy_Vc(:, :, :)
      real, intent(IN) :: scratch_Hydro_hy_flat3d(:, :, :)
      real, intent(IN) :: scratch_Hydro_hy_flux(:, :, :, :)
      real, intent(IN) :: scratch_Hydro_hy_fluxBufX(:, :, :, :)
      real, intent(IN) :: scratch_Hydro_hy_fluxBufY(:, :, :, :)
      real, intent(IN) :: scratch_Hydro_hy_fluxBufZ(:, :, :, :)
      real, intent(IN) :: scratch_Hydro_hy_flx(:, :, :, :)
      real, intent(IN) :: scratch_Hydro_hy_fly(:, :, :, :)
      real, intent(IN) :: scratch_Hydro_hy_flz(:, :, :, :)
      real, intent(IN) :: scratch_Hydro_hy_grav(:, :, :, :)
      real, intent(IN) :: scratch_Hydro_hy_rope(:, :, :, :)
      real, intent(IN) :: scratch_Hydro_hy_starState(:, :, :, :)
      real, intent(IN) :: scratch_Hydro_hy_tmpState(:, :, :, :)
      real, intent(IN) :: scratch_Hydro_hy_uMinus(:, :, :, :)
      real, intent(IN) :: scratch_Hydro_hy_uPlus(:, :, :, :)
      real, intent(IN) :: scratch_Hydro_xCenter_fake(:)
      real, intent(IN) :: scratch_Hydro_xLeft_fake(:)
      real, intent(IN) :: scratch_Hydro_xRight_fake(:)
      real, intent(IN) :: scratch_Hydro_yCenter_fake(:)
      real, intent(IN) :: scratch_Hydro_yLeft_fake(:)
      real, intent(IN) :: scratch_Hydro_yRight_fake(:)
      real, intent(IN) :: scratch_Hydro_zCenter_fake(:)


      CALL Hydro_prepBlock( &
         CC_1, &
         scratch_Hydro_hy_Vc, &
         tile_interior, &
         tile_arrayBounds, &
         scratch_Hydro_hy_starState, &
         scratch_Hydro_hy_tmpState, &
         external_Hydro_stage, &
         tile_lo, &
         tile_lbound &
      )
      CALL Hydro_advance( &
         external_Hydro_stage, &
         CC_1, &
         external_Hydro_dt, &
         external_Hydro_dtOld, &
         scratch_Hydro_hy_starState, &
         scratch_Hydro_hy_tmpState, &
         scratch_Hydro_hy_flx, &
         scratch_Hydro_hy_fly, &
         scratch_Hydro_hy_flz, &
         scratch_Hydro_hy_fluxBufX, &
         scratch_Hydro_hy_fluxBufY, &
         scratch_Hydro_hy_fluxBufZ, &
         scratch_Hydro_hy_grav, &
         scratch_Hydro_hy_flat3d, &
         scratch_Hydro_hy_rope, &
         scratch_Hydro_hy_flux, &
         scratch_Hydro_hy_uPlus, &
         scratch_Hydro_hy_uMinus, &
         tile_deltas, &
         tile_interior, &
         tile_arrayBounds, &
         tile_lo, &
         tile_lbound, &
         scratch_Hydro_xCenter_fake, &
         scratch_Hydro_yCenter_fake, &
         scratch_Hydro_zCenter_fake, &
         scratch_Hydro_xLeft_fake, &
         scratch_Hydro_xRight_fake, &
         scratch_Hydro_yLeft_fake, &
         scratch_Hydro_yRight_fake, &
         scratch_Hydro_fareaX_fake, &
         scratch_Hydro_fareaY_fake, &
         scratch_Hydro_fareaZ_fake, &
         scratch_Hydro_cvol_fake &
      )

   end subroutine cpu_taskfn_0_Fortran

end module cpu_taskfn_0_mod

