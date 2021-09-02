subroutine Hydro_advanceSolutionHll_3_packet_oacc_C2F( &
                    C_dataQ_h,                         &
                    C_nTiles_h, C_dt_h,                &
                    C_nTiles_d, C_dt_d) bind(c)
    use iso_c_binding,   ONLY : C_INT, &
                                C_DOUBLE, &
                                C_PTR, &
                                C_F_POINTER

    implicit none

    integer(C_INT), intent(IN), value :: C_dataQ_h
    integer(C_INT), intent(IN), value :: C_nTiles_h
    real(C_DOUBLE), intent(IN), value :: C_dt_h
    type(C_PTR),    intent(IN), value :: C_nTiles_d
    type(C_PTR),    intent(IN), value :: C_dt_d

    integer :: F_dataQ_h
    integer :: F_nTiles_h
    real    :: F_dt_h

    integer, pointer :: F_nTiles_d
    real,    pointer :: F_dt_d

    !!!!!----- HOST-SODE DATA TYPE MANAGEMENT
    ! Explicitly cast to Fortran default kinds without checking correctness
    ! We are assuming that the runtime has confirmed already that
    !   * C_INT    = integer default kind
    !   * C_DOUBLE = real    default kind.
    F_dataQ_h  = INT(C_dataQ_h)
    F_nTiles_h = INT(C_nTiles_h)
    F_dt_h     = REAL(C_dt_h)

    !!!!!----- DEVICE-SIDE DATA TYPE MANAGEMENT
    ! Get Fortran view of C device pointers
    CALL C_F_POINTER(C_nTiles_d, F_nTiles_d)
    CALL C_F_POINTER(C_dt_d,     F_dt_d)

    write(*,*) "[Hydro_advanceSolutionHll_3_packet_oacc_C2F] nTiles = ", &
               F_nTiles_h
    write(*,*) "[Hydro_advanceSolutionHll_3_packet_oacc_C2F] dt = ", &
               F_dt_h
    write(*,*) "[Hydro_advanceSolutionHll_3_packet_oacc_C2F] Q = ", &
               F_dataQ_h
end subroutine Hydro_advanceSolutionHll_3_packet_oacc_C2F

