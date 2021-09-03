subroutine Hydro_advanceSolutionHll_3_packet_oacc(dataQ_h, nTiles_d)
    use openacc

    implicit none

    integer(kind=acc_handle_kind), intent(IN)    :: dataQ_h
    integer,                       intent(INOUT) :: nTiles_d

!$acc data deviceptr(nTiles_d)

    !$acc  parallel default(none) &
    !$acc&          async(dataQ_h)
    write(*,*) "[Hydro_advanceSolutionHll_3_packet_oacc] nTiles_d = ", nTiles_d
    !$acc end parallel
    !$acc wait(dataQ_h)

!$acc end data

end subroutine Hydro_advanceSolutionHll_3_packet_oacc

