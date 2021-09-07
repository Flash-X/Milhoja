subroutine Hydro_advanceSolutionHll_3_packet_oacc(dataQ_h,            &
                                                  nTiles_d, dt_d,     &
                                                  deltas_all_d,       &
                                                  lo_all_d, hi_all_d)
    use openacc

    implicit none

    integer(kind=acc_handle_kind), intent(IN)    :: dataQ_h
    integer,                       intent(INOUT) :: nTiles_d
    real,                          intent(INOUT) :: dt_d
    real,                          intent(INOUT) :: deltas_all_d(:, :)
    integer,                       intent(INOUT) :: lo_all_d(:, :)
    integer,                       intent(INOUT) :: hi_all_d(:, :)

    integer :: n

!$acc data deviceptr(nTiles_d, deltas_all_d, lo_all_d, hi_all_d)

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
    end do
    !$acc end parallel loop
    !$acc wait(dataQ_h)

!$acc end data

end subroutine Hydro_advanceSolutionHll_3_packet_oacc

