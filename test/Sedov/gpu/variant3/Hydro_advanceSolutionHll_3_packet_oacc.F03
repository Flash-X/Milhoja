subroutine Hydro_advanceSolutionHll_3_packet_oacc(dataQ_h,                &
                                                  nTiles_d, dt_d,         &
                                                  deltas_all_d,           &
                                                  lo_all_d,   hi_all_d,   &
                                                  loGC_all_d, hiGC_all_d, &
                                                  U_all_d,                &
                                                  auxC_all_d,             &
                                                  faceX_all_d,            &
                                                  faceY_all_d,            &
                                                  faceZ_all_d)
    use openacc

    use Hydro_interface, ONLY : hy_kernel_3_packet_oacc

    implicit none

    !$acc routine (hy_kernel_3_packet_oacc) vector

    integer(kind=acc_handle_kind), intent(IN)    :: dataQ_h
    integer,                       intent(IN)    :: nTiles_d
    real,                          intent(IN)    :: dt_d
    real,                          intent(IN)    :: deltas_all_d(:, :)
    integer,                       intent(IN)    :: lo_all_d(:, :)
    integer,                       intent(IN)    :: hi_all_d(:, :)
    integer,                       intent(IN)    :: loGC_all_d(:, :)
    integer,                       intent(IN)    :: hiGC_all_d(:, :)
    real,                          intent(INOUT) :: U_all_d(:, :, :, :, :)
    real,                          intent(INOUT) :: auxC_all_d(:, :, :, :)
    real,                          intent(INOUT) :: faceX_all_d(:, :, :, :, :)
    real,                          intent(INOUT) :: faceY_all_d(:, :, :, :, :)
    real,                          intent(INOUT) :: faceZ_all_d(:, :, :, :, :)

    integer :: lo_loc_n(1:3)
    integer :: hi_loc_n(1:3)
    integer :: i, j, k, var
    integer :: n

    write(*,*) "shape(U_all_d)     = ", SHAPE(U_all_d)
    write(*,*) "shape(auxC_all_d)  = ", SHAPE(auxC_all_d)
    write(*,*) "shape(faceX_all_d) = ", SHAPE(faceX_all_d)
    write(*,*) "shape(faceY_all_d) = ", SHAPE(faceY_all_d)
    write(*,*) "shape(faceZ_all_d) = ", SHAPE(faceZ_all_d)

!$acc  data create(lo_loc_n, hi_loc_n) &
!$acc&      deviceptr(nTiles_d, deltas_all_d, lo_all_d, hi_all_d, loGC_all_d, hiGC_all_d, U_all_d)

    !$acc  parallel loop gang default(none)               &
    !$acc&                    private(lo_loc_n, hi_loc_n) &
    !$acc&                    async(dataQ_h)
    do n = 1, nTiles_d
        lo_loc_n(:) = lo_all_d(:, n) - loGC_all_d(:, n) + 1
        hi_loc_n(:) = hi_all_d(:, n) - loGC_all_d(:, n) + 1

        CALL hy_kernel_3_packet_oacc(lo_loc_n, hi_loc_n, U_all_d(:, :, :, :, n))
    end do
    !$acc end parallel loop
    !$acc wait(dataQ_h)

!$acc end data

end subroutine Hydro_advanceSolutionHll_3_packet_oacc

