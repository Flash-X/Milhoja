#include "Milhoja.h"

subroutine cpu_tf_hydro_C2F( &
                            C_external_hydro_op1_dt, &
                            C_external_hydro_op1_eosMode, &
                            C_tile_deltas, &
                            C_tile_hi, &
                            C_tile_interior, &
                            C_tile_lo, &
                            C_CC_1, &
                            C_scratch_hydro_op1_auxC, &
                            C_scratch_hydro_op1_flX, &
                            C_scratch_hydro_op1_flY, &
                            C_scratch_hydro_op1_flZ, &
                            C_lbdd_CC_1, &
                            C_lbdd_scratch_hydro_op1_auxC, &
                            C_lbdd_scratch_hydro_op1_flX, &
                            C_lbdd_scratch_hydro_op1_flY, &
                            C_lbdd_scratch_hydro_op1_flZ &
)bind(c, name="cpu_tf_hydro_C2F")
    use iso_c_binding, ONLY : C_PTR, C_F_POINTER
    use milhoja_types_mod, ONLY : MILHOJA_INT, MILHOJA_REAL
    use cpu_tf_hydro_mod, ONLY : cpu_tf_hydro_Fortran
    implicit none

    real(MILHOJA_REAL), intent(IN), value :: C_external_hydro_op1_dt
    integer(MILHOJA_INT), intent(IN), value :: C_external_hydro_op1_eosMode
    type(C_PTR), intent(IN), value :: C_tile_deltas
    type(C_PTR), intent(IN), value :: C_tile_hi
    type(C_PTR), intent(IN), value :: C_tile_interior
    type(C_PTR), intent(IN), value :: C_tile_lo
    type(C_PTR), intent(IN), value :: C_CC_1
    type(C_PTR), intent(IN), value :: C_scratch_hydro_op1_auxC
    type(C_PTR), intent(IN), value :: C_scratch_hydro_op1_flX
    type(C_PTR), intent(IN), value :: C_scratch_hydro_op1_flY
    type(C_PTR), intent(IN), value :: C_scratch_hydro_op1_flZ
    type(C_PTR), intent(IN), value :: C_lbdd_CC_1
    type(C_PTR), intent(IN), value :: C_lbdd_scratch_hydro_op1_auxC
    type(C_PTR), intent(IN), value :: C_lbdd_scratch_hydro_op1_flX
    type(C_PTR), intent(IN), value :: C_lbdd_scratch_hydro_op1_flY
    type(C_PTR), intent(IN), value :: C_lbdd_scratch_hydro_op1_flZ

    real :: F_external_hydro_op1_dt
    integer :: F_external_hydro_op1_eosMode
    real, pointer :: F_tile_deltas(:)
    integer, pointer :: F_tile_hi(:)
    integer, pointer :: F_tile_interior(:,:)
    integer, pointer :: F_tile_lo(:)
    real, pointer :: F_CC_1(:,:,:,:)
    real, pointer :: F_scratch_hydro_op1_auxC(:,:,:)
    real, pointer :: F_scratch_hydro_op1_flX(:,:,:,:)
    real, pointer :: F_scratch_hydro_op1_flY(:,:,:,:)
    real, pointer :: F_scratch_hydro_op1_flZ(:,:,:,:)
    integer, pointer :: F_lbdd_CC_1(:)
    integer, pointer :: F_lbdd_scratch_hydro_op1_auxC(:)
    integer, pointer :: F_lbdd_scratch_hydro_op1_flX(:)
    integer, pointer :: F_lbdd_scratch_hydro_op1_flY(:)
    integer, pointer :: F_lbdd_scratch_hydro_op1_flZ(:)

    F_external_hydro_op1_dt = REAL(C_external_hydro_op1_dt)
    F_external_hydro_op1_eosMode = INT(C_external_hydro_op1_eosMode)
    CALL C_F_POINTER(C_tile_deltas, F_tile_deltas, shape=[MILHOJA_MDIM])
    CALL C_F_POINTER(C_tile_hi, F_tile_hi, shape=[MILHOJA_MDIM])
    CALL C_F_POINTER(C_tile_interior, F_tile_interior, shape=[2,MILHOJA_MDIM])
    CALL C_F_POINTER(C_tile_lo, F_tile_lo, shape=[MILHOJA_MDIM])
    CALL C_F_POINTER(C_CC_1, F_CC_1, shape=[16 + 2 * 1 * MILHOJA_K1D,16 + 2 * 1 * MILHOJA_K2D,16 + 2 * 1 * MILHOJA_K3D,10])
    CALL C_F_POINTER(C_scratch_hydro_op1_auxC, F_scratch_hydro_op1_auxC, shape=[18,18,18])
    CALL C_F_POINTER(C_scratch_hydro_op1_flX, F_scratch_hydro_op1_flX, shape=[17,16,16,5])
    CALL C_F_POINTER(C_scratch_hydro_op1_flY, F_scratch_hydro_op1_flY, shape=[16,17,16,5])
    CALL C_F_POINTER(C_scratch_hydro_op1_flZ, F_scratch_hydro_op1_flZ, shape=[16,16,17,5])
    CALL C_F_POINTER(C_lbdd_CC_1, F_lbdd_CC_1, shape=[4])
    CALL C_F_POINTER(C_lbdd_scratch_hydro_op1_auxC, F_lbdd_scratch_hydro_op1_auxC, shape=[3])
    CALL C_F_POINTER(C_lbdd_scratch_hydro_op1_flX, F_lbdd_scratch_hydro_op1_flX, shape=[4])
    CALL C_F_POINTER(C_lbdd_scratch_hydro_op1_flY, F_lbdd_scratch_hydro_op1_flY, shape=[4])
    CALL C_F_POINTER(C_lbdd_scratch_hydro_op1_flZ, F_lbdd_scratch_hydro_op1_flZ, shape=[4])

    CALL cpu_tf_hydro_Fortran( &
        F_external_hydro_op1_dt, &
        F_external_hydro_op1_eosMode, &
        F_tile_deltas, &
        F_tile_hi, &
        F_tile_interior, &
        F_tile_lo, &
        F_CC_1, &
        F_scratch_hydro_op1_auxC, &
        F_scratch_hydro_op1_flX, &
        F_scratch_hydro_op1_flY, &
        F_scratch_hydro_op1_flZ, &
        F_lbdd_CC_1, &
        F_lbdd_scratch_hydro_op1_auxC, &
        F_lbdd_scratch_hydro_op1_flX, &
        F_lbdd_scratch_hydro_op1_flY, &
        F_lbdd_scratch_hydro_op1_flZ)
end subroutine cpu_tf_hydro_C2F
