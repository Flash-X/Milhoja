subroutine Hydro_advanceSolutionHll_3_packet_oacc(dataQ_h,                &
                                                  nTiles_d, dt_d,         &
                                                  deltas_all_d,           &
                                                  lo_all_d,   hi_all_d,   &
                                                  loGC_all_d, hiGC_all_d, &
                                                  U_all_d)
    use openacc

    implicit none

    integer(kind=acc_handle_kind), intent(IN)    :: dataQ_h
    integer,                       intent(INOUT) :: nTiles_d
    real,                          intent(INOUT) :: dt_d
    real,                          intent(INOUT) :: deltas_all_d(:, :)
    integer,                       intent(INOUT) :: lo_all_d(:, :)
    integer,                       intent(INOUT) :: hi_all_d(:, :)
    integer,                       intent(INOUT) :: loGC_all_d(:, :)
    integer,                       intent(INOUT) :: hiGC_all_d(:, :)
    real,                          intent(INOUT) :: U_all_d(:,:,:,:,:)

    integer :: n

    write(*,*) "shape(U_all_d) = ", SHAPE(U_all_d)

!$acc data deviceptr(nTiles_d, deltas_all_d, lo_all_d, hi_all_d, loGC_all_d, hiGC_all_d, U_all_d)

    !$acc  parallel loop gang default(none) &
    !$acc&                    async(dataQ_h)
    do n = 1, nTiles_d
        write(*,*) "[Hydro_advanceSolutionHll_3_packet_oacc] deltas_d = (", &
                    deltas_all_d(1, n), ",", &
                    deltas_all_d(2, n), ",", &
                    deltas_all_d(3, n), ")"
        write(*,*) "[Hydro_advanceSolutionHll_3_packet_oacc] lo_d     = (", &
                    lo_all_d(1, n), ",", &
                    lo_all_d(2, n), ",", &
                    lo_all_d(3, n), ")"
        write(*,*) "[Hydro_advanceSolutionHll_3_packet_oacc] hi_d     = (", &
                    hi_all_d(1, n), ",", &
                    hi_all_d(2, n), ",", &
                    hi_all_d(3, n), ")"
        write(*,*) "[Hydro_advanceSolutionHll_3_packet_oacc] loGC_d   = (", &
                    loGC_all_d(1, n), ",", &
                    loGC_all_d(2, n), ",", &
                    loGC_all_d(3, n), ")"
        write(*,*) "[Hydro_advanceSolutionHll_3_packet_oacc] hiGC_d   = (", &
                    hiGC_all_d(1, n), ",", &
                    hiGC_all_d(2, n), ",", &
                    hiGC_all_d(3, n), ")"
        write(*,*) "[Hydro_advanceSolutionHll_3_packet_oacc] U_d      = (", &
                    U_all_d(1, 1, 1, 1, n)
    end do
    !$acc end parallel loop
    !$acc wait(dataQ_h)

!$acc end data

end subroutine Hydro_advanceSolutionHll_3_packet_oacc

