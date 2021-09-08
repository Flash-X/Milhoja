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
    real,                          intent(INOUT) :: U_all_d(:, :, :, :, :)

    integer :: lo_loc_n(1:3)
    integer :: hi_loc_n(1:3)
    integer :: i, j, k, var
    integer :: n

    write(*,*) "shape(U_all_d) = ", SHAPE(U_all_d)

!$acc  data create(lo_loc_n, hi_loc_n) &
!$acc&      deviceptr(nTiles_d, deltas_all_d, lo_all_d, hi_all_d, loGC_all_d, hiGC_all_d, U_all_d)

    !$acc  parallel loop gang default(none)               &
    !$acc&                    private(lo_loc_n, hi_loc_n) &
    !$acc&                    async(dataQ_h)
    do n = 1, nTiles_d
        lo_loc_n(:) = lo_all_d(:, n) - loGC_all_d(:, n) + 1
        hi_loc_n(:) = hi_all_d(:, n) - loGC_all_d(:, n) + 1

        !$acc loop vector collapse(4)
        do           var = 1, 8
            do         k = lo_loc_n(3), hi_loc_n(3)
                do     j = lo_loc_n(2), hi_loc_n(2)
                    do i = lo_loc_n(1), hi_loc_n(1)
                        U_all_d(i, j, k, var, n) = U_all_d(i, j, k, var, n) * var
                    end do
                end do
            end do
        end do
        !$acc end loop

    end do
    !$acc end parallel loop
    !$acc wait(dataQ_h)

!$acc end data

end subroutine Hydro_advanceSolutionHll_3_packet_oacc

