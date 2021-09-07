subroutine Hydro_advanceSolutionHll_3_packet_oacc_C2F( &
                    C_dataQ_h,                         &
                    C_nTiles_h,                        &
                    C_nTiles_d, C_dt_d,                &
                    C_deltas_start_d,                  &
                    C_lo_start_d,   C_hi_start_d,      &
                    C_loGC_start_d, C_hiGC_start_d) bind(c)
    use iso_c_binding,   ONLY : C_INT, &
                                C_DOUBLE, &
                                C_PTR, &
                                C_F_POINTER

    use openacc,         ONLY : acc_handle_kind

    use Hydro_interface, ONLY : Hydro_advanceSolutionHll_3_packet_oacc

    implicit none

    integer(C_INT), intent(IN), value :: C_dataQ_h
    integer(C_INT), intent(IN), value :: C_nTiles_h
    type(C_PTR),    intent(IN), value :: C_nTiles_d
    type(C_PTR),    intent(IN), value :: C_dt_d
    type(C_PTR),    intent(IN), value :: C_deltas_start_d
    type(C_PTR),    intent(IN), value :: C_lo_start_d
    type(C_PTR),    intent(IN), value :: C_hi_start_d
    type(C_PTR),    intent(IN), value :: C_loGC_start_d
    type(C_PTR),    intent(IN), value :: C_hiGC_start_d

    integer(kind=acc_handle_kind) :: F_dataQ_h
    integer                       :: F_nTiles_h

    integer, pointer :: F_nTiles_d
    real,    pointer :: F_dt_d
    real,    pointer :: F_deltas_all_d(:, :)
    integer, pointer :: F_lo_all_d(:, :)
    integer, pointer :: F_hi_all_d(:, :)
    integer, pointer :: F_loGC_all_d(:, :)
    integer, pointer :: F_hiGC_all_d(:, :)

    !!!!!----- HOST-SODE DATA TYPE MANAGEMENT
    ! Explicitly cast to Fortran default kinds without checking correctness
    ! We are assuming that the runtime has confirmed already that
    !   * C_INT    = integer default kind
    !   * C_DOUBLE = real    default kind.
    ! TODO: The runtime should confirm at intialization that this cast
    !       will always be correct for the given runtime configuration.
    F_dataQ_h  = INT(C_dataQ_h, kind=acc_handle_kind)
    F_nTiles_h = INT(C_nTiles_h)

    !!!!!----- DEVICE-SIDE DATA TYPE MANAGEMENT
    ! Get Fortran view of C device pointers
    CALL C_F_POINTER(C_nTiles_d,       F_nTiles_d)
    CALL C_F_POINTER(C_dt_d,           F_dt_d)
    CALL C_F_POINTER(C_deltas_start_d, F_deltas_all_d, shape=[3, F_nTiles_h])
    CALL C_F_POINTER(C_lo_start_d,     F_lo_all_d,     shape=[3, F_nTiles_h])
    CALL C_F_POINTER(C_hi_start_d,     F_hi_all_d,     shape=[3, F_nTiles_h])
    CALL C_F_POINTER(C_loGC_start_d,   F_loGC_all_d,   shape=[3, F_nTiles_h])
    CALL C_F_POINTER(C_hiGC_start_d,   F_hiGC_all_d,   shape=[3, F_nTiles_h])

    !!!!!----- PASS DATA TO TARGET-SPECIFIC STATIC FORTRAN LAYER
    CALL Hydro_advanceSolutionHll_3_packet_oacc(F_dataQ_h,                  &
                                                F_nTiles_d, F_dt_d,         &
                                                F_deltas_all_d,             &
                                                F_lo_all_d,   F_hi_all_d,   &
                                                F_loGC_all_d, F_hiGC_all_d)
end subroutine Hydro_advanceSolutionHll_3_packet_oacc_C2F

