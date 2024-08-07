#include "Milhoja.h"

module cpu_tf_hydro_mod
    implicit none
    private

    public :: cpu_tf_hydro_Fortran
    public :: cpu_tf_hydro_Cpp2C

   interface
      subroutine cpu_tf_hydro_Cpp2C(C_threadIndex, C_dataItemPtr) &
      bind(c, name="cpu_tf_hydro_Cpp2C")
        use iso_c_binding,     ONLY : C_PTR
        use milhoja_types_mod, ONLY : MILHOJA_INT
        integer(MILHOJA_INT), intent(IN), value :: C_threadIndex
        type(C_PTR),          intent(IN), value :: C_dataItemPtr
      end subroutine cpu_tf_hydro_Cpp2C
   end interface

contains
    subroutine cpu_tf_hydro_Fortran(external_hydro_op1_dt, &
                                    external_hydro_op1_eosMode, &
                                    tile_deltas, &
                                    tile_hi, &
                                    tile_interior, &
                                    tile_lo, &
                                    CC_1, &
                                    scratch_hydro_op1_auxC, &
                                    scratch_hydro_op1_flX, &
                                    scratch_hydro_op1_flY, &
                                    scratch_hydro_op1_flZ, &
                                    lbdd_CC_1, &
                                    lbdd_scratch_hydro_op1_auxC, &
                                    lbdd_scratch_hydro_op1_flX, &
                                    lbdd_scratch_hydro_op1_flY, &
                                    lbdd_scratch_hydro_op1_flZ)

        use Hydro_advanceSolution_variants_mod, ONLY : Hydro_computeSoundSpeed_block_cpu, &
                                                       Hydro_computeFluxes_X_block_cpu, &
                                                       Hydro_computeFluxes_Y_block_cpu, &
                                                       Hydro_computeFluxes_Z_block_cpu, &
                                                       Hydro_updateSolution_block_cpu
        use Eos_interface,                      ONLY : Eos_wrapped

        real,    intent(IN)            :: external_hydro_op1_dt
        integer, intent(IN)            :: external_hydro_op1_eosMode
        real,    intent(IN)            :: tile_deltas(1:MILHOJA_MDIM)
        integer, intent(IN)            :: tile_hi(1:MILHOJA_MDIM)
        integer, intent(IN)            :: tile_interior(1:2, 1:MILHOJA_MDIM)
        integer, intent(IN)            :: tile_lo(1:MILHOJA_MDIM)
        integer, intent(IN)            :: lbdd_CC_1(1:4)
        real,    intent(INOUT), target :: CC_1(:, :, :, :)
        integer, intent(IN)            :: lbdd_scratch_hydro_op1_auxC(1:3)
        real,    intent(INOUT)         :: scratch_hydro_op1_auxC(:, :, :)
        integer, intent(IN)            :: lbdd_scratch_hydro_op1_flX(1:4)
        real,    intent(INOUT)         :: scratch_hydro_op1_flX(:, :, :, :)
        real,    intent(INOUT)         :: scratch_hydro_op1_flY(:, :, :, :)
        real,    intent(INOUT)         :: scratch_hydro_op1_flZ(:, :, :, :)

        real, pointer :: CC_1_ptr(:, :, :, :)

        NULLIFY(CC_1_ptr)

        CALL Hydro_computeSoundSpeed_block_cpu(tile_lo, tile_hi, &
                                               CC_1, &
                                               lbdd_CC_1, &
                                               scratch_hydro_op1_auxC, &
                                               lbdd_scratch_hydro_op1_auxC)
        CALL Hydro_computeFluxes_X_block_cpu(external_hydro_op1_dt, &
                                             tile_lo, tile_hi, &
                                             tile_deltas, &
                                             CC_1, &
                                             lbdd_CC_1, &
                                             scratch_hydro_op1_auxC, &
                                             lbdd_scratch_hydro_op1_auxC, &
                                             scratch_hydro_op1_flX, &
                                             lbdd_scratch_hydro_op1_flX)
        CALL Hydro_computeFluxes_Y_block_cpu(external_hydro_op1_dt, &
                                             tile_lo, tile_hi, &
                                             tile_deltas, &
                                             CC_1, &
                                             lbdd_CC_1, &
                                             scratch_hydro_op1_auxC, &
                                             lbdd_scratch_hydro_op1_auxC, &
                                             scratch_hydro_op1_flY, &
                                             lbdd_scratch_hydro_op1_flY)
        CALL Hydro_updateSolution_block_cpu(tile_lo, tile_hi, &
                                            scratch_hydro_op1_flX, &
                                            scratch_hydro_op1_flY, &
                                            scratch_hydro_op1_flZ, &
                                            lbdd_scratch_hydro_op1_flX, &
                                            CC_1, &
                                            lbdd_CC_1)

        CC_1_ptr(lbdd_CC_1(1):,lbdd_CC_1(2):,lbdd_CC_1(3):,lbdd_CC_1(4):) => CC_1
        CALL Eos_wrapped(external_hydro_op1_eosMode, tile_interior, CC_1_ptr)
        NULLIFY(CC_1_ptr)
    end subroutine cpu_tf_hydro_Fortran

end module cpu_tf_hydro_mod