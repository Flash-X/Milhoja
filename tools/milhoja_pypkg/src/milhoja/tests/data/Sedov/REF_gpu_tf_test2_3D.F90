#include "Milhoja_interface_error_codes.h"

module gpu_tf_test2_mod
    implicit none
    private

    public :: gpu_tf_test2

    ! NOTE TO MILHOJA USERS:
    ! The Fortran interfaces defined here should be used to create and manage
    ! prototype data items to be passed to Milhoja for use with this task
    ! function.
    interface
        ! Instantiate the prototype data packet
        !
        ! Arguments
        !   C_packet - Milhoja-internal handle to the data packet
        !   All others - user-specified external arguments
        function instantiate_gpu_tf_test2_packet_C( &
                    C_packet, &
                    C_dt &
                ) result(C_ierr) bind(c)
            use iso_c_binding,     ONLY : C_PTR
            use milhoja_types_mod, ONLY : MILHOJA_INT
            use milhoja_types_mod, ONLY : MILHOJA_REAL
            type(C_PTR),         intent(IN)        :: C_packet
            real(MILHOJA_REAL),  intent(IN), value :: C_dt
            integer(MILHOJA_INT)                   :: C_ierr
        end function instantiate_gpu_tf_test2_packet_C

        ! Delete the prototype data packet
        !
        ! Arguments
        !   C_packet - Milhoja-internal handle obtained when instantiating
        !               data packet
        function delete_gpu_tf_test2_packet_C(C_packet) result(C_ierr) bind(c)
            use iso_c_binding,     ONLY : C_PTR
            use milhoja_types_mod, ONLY : MILHOJA_INT
            type(C_PTR),         intent(IN), value :: C_packet
            integer(MILHOJA_INT)                   :: C_ierr
        end function delete_gpu_tf_test2_packet_C
    end interface

    ! Milhoja-internal functions.  Milhoja users should not call these.
    interface
        function release_gpu_tf_test2_extra_queue_C(C_packet, C_id) result(C_ierr) bind(c)
            use iso_c_binding,     ONLY : C_PTR
            use milhoja_types_mod, ONLY : MILHOJA_INT
            type(C_PTR),          intent(IN), value :: C_packet
            integer(MILHOJA_INT), intent(IN), value :: C_id
            integer(MILHOJA_INT)                    :: C_ierr
        end function release_gpu_tf_test2_extra_queue_C
    end interface

contains

    subroutine gpu_tf_test2(         &
                    C_packet_h,      &
                    dataQ_h,         &
                    queue2_h,        &
                    queue3_h,        &
                    nTiles_d,        &
                    dt_d,            &
                    tile_lo_d,       &
                    tile_hi_d,       &
                    tile_deltas_d,   &
                    CC_1_d,          &
                    hydro_op1_flX_d, &
                    hydro_op1_flY_d, &
                    hydro_op1_flZ_d, &
                    hydro_op1_auxc_d &
            )
        use iso_c_binding, ONLY : C_PTR
        use openacc

        use milhoja_types_mode, ONLY : MILHOJA_INT

        use dr_cg_hydroAdvance_mod, ONLY : Hydro_computeSoundSpeedHll_gpu_oacc
        use dr_cg_hydroAdvance_mod, ONLY : Hydro_computeFluxesHll_X_gpu_oacc
        use dr_cg_hydroAdvance_mod, ONLY : Hydro_computeFluxesHll_Y_gpu_oacc
        use dr_cg_hydroAdvance_mod, ONLY : Hydro_computeFluxesHll_Z_gpu_oacc

        !$acc routine (Hydro_computeSoundSpeedHll_gpu_oacc) vector
        !$acc routine (Hydro_computeFluxesHll_X_gpu_oacc) vector
        !$acc routine (Hydro_computeFluxesHll_Y_gpu_oacc) vector
        !$acc routine (Hydro_computeFluxesHll_Z_gpu_oacc) vector

        type(C_PTR),                   intent(IN)    :: C_packet_h
        integer(kind=acc_handle_kind), intent(IN)    :: dataQ_h
        integer(kind=acc_handle_kind), intent(IN)    :: queue2_h
        integer(kind=acc_handle_kind), intent(IN)    :: queue3_h
        integer,                       intent(IN)    :: nTiles_d
        real,                          intent(IN)    :: dt_d
        integer,                       intent(IN)    :: tile_lo_d(:, :)
        integer,                       intent(IN)    :: tile_hi_d(:, :)
        real,                          intent(IN)    :: tile_deltas_d(:, :)
        real,                          intent(INOUT) :: CC_1_d(:, :, :, :, :)
        real,                          intent(OUT)   :: hydro_op1_flX_d(:, :, :, :, :)
        real,                          intent(OUT)   :: hydro_op1_flY_d(:, :, :, :, :)
        real,                          intent(OUT)   :: hydro_op1_flZ_d(:, :, :, :, :)
        real,                          intent(OUT)   :: hydro_op1_auxc_d(:, :, :, :)

        integer              :: n
        integer(MILHOJA_INT) :: MH_idx
        integer(MILHOJA_INT) :: MH_ierr

        !$acc data                   &
        !$acc& deviceptr(            &
        !$acc&      nTiles_d,        &
        !$acc&      dt_d,            &
        !$acc&      tile_lo_d,       &
        !$acc&      tile_hi_d,       &
        !$acc&      tile_deltas_d,   &
        !$acc&      CC_1_d,          &
        !$acc&      hydro_op1_flX_d, &
        !$acc&      hydro_op1_flY_d, &
        !$acc&      hydro_op1_flZ_d, &
        !$acc&      hydro_op1_auxc_d &
        !$acc&  )

        !$acc parallel loop gang default(none) &
        !$acc& async(dataQ_h)
        do n = 1, nTiles_d
            CALL Hydro_computeSoundSpeedHll_gpu_oacc( &
                    tile_lo_d,                        &
                    tile_hi_d,                        &
                    CC_1_d,                           &
                    hydro_op1_auxc_d                  &
                 )
        end do
        !$acc end parallel loop

        !$acc wait(dataQ_h)

        !$acc parallel loop gang default(none) &
        !$acc& async(dataQ_h)
        do n = 1, nTiles_d
            CALL Hydro_computeFluxesHll_X_gpu_oacc( &
                    dt_d,                           &
                    tile_lo_d,                      &
                    tile_hi_d,                      &
                    tile_deltas_d,                  &
                    CC_1_d,                         &
                    hydro_op1_auxc_d,               &
                    hydro_op1_flX_d                 &
                 )
        end do
        !$acc end parallel loop

        !$acc parallel loop gang default(none) &
        !$acc& async(queue2_h)
        do n = 1, nTiles_d
            CALL Hydro_computeFluxesHll_Y_gpu_oacc( &
                    dt_d,                           &
                    tile_lo_d,                      &
                    tile_hi_d,                      &
                    tile_deltas_d,                  &
                    CC_1_d,                         &
                    hydro_op1_auxc_d,               &
                    hydro_op1_flY_d                 &
                 )
        end do
        !$acc end parallel loop

        !$acc parallel loop gang default(none) &
        !$acc& async(queue3_h)
        do n = 1, nTiles_d
            CALL Hydro_computeFluxesHll_Z_gpu_oacc( &
                    dt_d,                           &
                    tile_lo_d,                      &
                    tile_hi_d,                      &
                    tile_deltas_d,                  &
                    CC_1_d,                         &
                    hydro_op1_auxc_d,               &
                    hydro_op1_flZ_d                 &
                 )
        end do
        !$acc end parallel loop

        !$acc wait(           &
        !$acc&      dataQ_h,  &
        !$acc&      queue2_h, &
        !$acc&      queue3_h  &
        !$acc&  )
    
        MH_idx = INT(2, kind=MILHOJA_INT)
        MH_ierr = release_gpu_tf_test2_extra_queue_C(C_packet_h, MH_idx)
        if (MH_ierr /= MILHOJA_SUCCESS) then
            write(*,*) "[gpu_tf_test2] Unable to release extra OpenACC async queue 2"
            STOP
        end if
        MH_idx = INT(3, kind=MILHOJA_INT)
        MH_ierr = release_gpu_tf_test2_extra_queue_C(C_packet_h, MH_idx)
        if (MH_ierr /= MILHOJA_SUCCESS) then
            write(*,*) "[gpu_tf_test2] Unable to release extra OpenACC async queue 3"
            STOP
        end if

        !$acc end data
    end subroutine gpu_tf_test2

end module gpu_tf_test2_mod