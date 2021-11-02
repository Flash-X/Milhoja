#include "constants.h"

subroutine hy_advanceSolutionHll_3_packet_oacc(dataQ_h,                &
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

    use hy_cg_interface,  ONLY : hy_computeSoundSpeedHll_oacc, &
                                 hy_computeFluxesHll_X_oacc,   &
                                 hy_computeFluxesHll_Y_oacc,   &
                                 hy_computeFluxesHll_Z_oacc,   &
                                 hy_updateSolutionHll_oacc
    use eos_cg_interface, ONLY : eos_idealGammaDensIe_oacc

    implicit none

    !$acc routine (hy_computeSoundSpeedHll_oacc) vector
    !$acc routine (hy_computeFluxesHll_X_oacc)   vector
    !$acc routine (hy_computeFluxesHll_Y_oacc)   vector
    !$acc routine (hy_computeFluxesHll_Z_oacc)   vector
    !$acc routine (hy_updateSolutionHll_oacc)    vector
    !$acc routine (eos_idealGammaDensIe_oacc)    vector

    integer(kind=acc_handle_kind), intent(IN)    :: dataQ_h
    integer,                       intent(IN)    :: nTiles_d
    real,                          intent(IN)    :: dt_d
    real,                          intent(IN)    :: deltas_all_d(:, :)
    integer,                       intent(IN)    :: lo_all_d(:, :)
    integer,                       intent(IN)    :: hi_all_d(:, :)
    integer,                       intent(IN)    :: loGC_all_d(:, :)
    integer,                       intent(IN)    :: hiGC_all_d(:, :)
    real,                          intent(INOUT) :: U_all_d(:, :, :, :, :)
    real,                          intent(OUT)   :: auxC_all_d(:, :, :, :)
    real,                          intent(OUT)   :: faceX_all_d(:, :, :, :, :)
    real,                          intent(OUT)   :: faceY_all_d(:, :, :, :, :)
    real,                          intent(OUT)   :: faceZ_all_d(:, :, :, :, :)

    integer :: lo_loc_n(1:MDIM)
    integer :: hi_loc_n(1:MDIM)
    integer :: i, j, k, var
    integer :: n

!    write(*,*) "shape(U_all_d)     = ", SHAPE(U_all_d)
!    write(*,*) "shape(auxC_all_d)  = ", SHAPE(auxC_all_d)
!    write(*,*) "shape(faceX_all_d) = ", SHAPE(faceX_all_d)
!    write(*,*) "shape(faceY_all_d) = ", SHAPE(faceY_all_d)
!    write(*,*) "shape(faceZ_all_d) = ", SHAPE(faceZ_all_d)

!$acc  data create(lo_loc_n, hi_loc_n)                       &
!$acc&      deviceptr(nTiles_d, dt_d, deltas_all_d,          &
!$acc&                lo_all_d, hi_all_d, loGC_all_d,        &
!$acc&                faceX_all_d, faceY_all_d, faceZ_all_d, &
!$acc&                auxC_all_d, U_all_d)

    !----- ADVANCE SOLUTION
    ! Update unk data on interiors only
    !   * It is assumed that the GC are filled already
    !   * No tiling for now means that computing fluxes and updating the
    !     solution can be fused and the full advance run independently on each
    !     block and in place.

    !----- COMPUTE FLUXES
    !$acc  parallel loop gang default(none)               &
    !$acc&                    private(lo_loc_n, hi_loc_n) &
    !$acc&                    async(dataQ_h)
    do n = 1, nTiles_d
        lo_loc_n(:) = lo_all_d(:, n) - loGC_all_d(:, n) + 1
        hi_loc_n(:) = hi_all_d(:, n) - loGC_all_d(:, n) + 1

        CALL hy_computeSoundSpeedHll_oacc(lo_loc_n, hi_loc_n,     &
                                          U_all_d(:, :, :, :, n), &
                                          auxC_all_d(:, :, :, n))
    end do
    !$acc end parallel loop

    ! The X, Y, and Z fluxes each depend on the speed of sound, but can
    ! be computed independently and therefore concurrently.
#if   NDIM == 1
    !$acc  parallel loop gang default(none)               &
    !$acc&                    private(lo_loc_n, hi_loc_n) &
    !$acc&                    async(dataQ_h)
    do n = 1, nTiles_d
        lo_loc_n(:) = lo_all_d(:, n) - loGC_all_d(:, n) + 1
        hi_loc_n(:) = hi_all_d(:, n) - loGC_all_d(:, n) + 1

        CALL hy_computeFluxesHll_X_oacc(dt_d, &
                                        lo_loc_n, hi_loc_n,         &
                                        deltas_all_d(:, n),         &
                                        U_all_d(:, :, :, :, n),     &
                                        auxC_all_d(:, :, :, n),     &
                                        faceX_all_d(:, :, :, :, n))
    end do
    !$acc end parallel loop

    ! No need for barrier since all kernels are launched on the same
    ! queue for 1D case.
#elif NDIM == 2
    !$acc  parallel loop gang default(none)               &
    !$acc&                    private(lo_loc_n, hi_loc_n) &
    !$acc&                    async(dataQ_h)
    do n = 1, nTiles_d
        lo_loc_n(:) = lo_all_d(:, n) - loGC_all_d(:, n) + 1
        hi_loc_n(:) = hi_all_d(:, n) - loGC_all_d(:, n) + 1

        ! It seems like for small 2D blocks, fusing kernels is more
        ! efficient than fusing actions (i.e. running the two kernels
        ! concurrently).  Too much work for the GPU?  Too much overhead
        ! from the stream sync (i.e. OpenACC wait)?
        ! TODO: This was for C++ case.  Confirm for the Fortran case as well.
        CALL hy_computeFluxesHll_X_oacc(dt_d,                       &
                                        lo_loc_n, hi_loc_n,         &
                                        deltas_all_d(:, n),         &
                                        U_all_d(:, :, :, :, n),     &
                                        auxC_all_d(:, :, :, n),     &
                                        faceX_all_d(:, :, :, :, n))
        CALL hy_computeFluxesHll_Y_oacc(dt_d,                       &
                                        lo_loc_n, hi_loc_n,         &
                                        deltas_all_d(:, n),         &
                                        U_all_d(:, :, :, :, n),     &
                                        auxC_all_d(:, :, :, n),     &
                                        faceY_all_d(:, :, :, :, n))
    end do
    !$acc end parallel loop

    ! No need for barrier since all kernels are launched on the same
    ! queue for 2D case.
#elif NDIM == 3
    ! TODO: Launch kernels concurrently as in C++ case

    !$acc  parallel loop gang default(none)               &
    !$acc&                    private(lo_loc_n, hi_loc_n) &
    !$acc&                    async(dataQ_h)
    do n = 1, nTiles_d
        lo_loc_n(:) = lo_all_d(:, n) - loGC_all_d(:, n) + 1
        hi_loc_n(:) = hi_all_d(:, n) - loGC_all_d(:, n) + 1

        CALL hy_computeFluxesHll_X_oacc(dt_d,                       &
                                        lo_loc_n, hi_loc_n,         &
                                        deltas_all_d(:, n),         &
                                        U_all_d(:, :, :, :, n),     &
                                        auxC_all_d(:, :, :, n),     &
                                        faceX_all_d(:, :, :, :, n))
        CALL hy_computeFluxesHll_Y_oacc(dt_d,                       &
                                        lo_loc_n, hi_loc_n,         &
                                        deltas_all_d(:, n),         &
                                        U_all_d(:, :, :, :, n),     &
                                        auxC_all_d(:, :, :, n),     &
                                        faceY_all_d(:, :, :, :, n))
        CALL hy_computeFluxesHll_Z_oacc(dt_d,                       &
                                        lo_loc_n, hi_loc_n,         &
                                        deltas_all_d(:, n),         &
                                        U_all_d(:, :, :, :, n),     &
                                        auxC_all_d(:, :, :, n),     &
                                        faceZ_all_d(:, :, :, :, n))
    end do
    !$acc end parallel loop

    ! No need for barrier since all kernels are launched on the same
    ! queue for 3D case.
#endif

    !----- UPDATE SOLUTIONS IN PLACE

    !$acc  parallel loop gang default(none)               &
    !$acc&                    private(lo_loc_n, hi_loc_n) &
    !$acc&                    async(dataQ_h)
    do n = 1, nTiles_d
        lo_loc_n(:) = lo_all_d(:, n) - loGC_all_d(:, n) + 1
        hi_loc_n(:) = hi_all_d(:, n) - loGC_all_d(:, n) + 1

        ! NOTE: If NDIM < 3, then some of the face[YZ]_all_d will be garbage.
        !       We therefore assume that this routine will not use
        !       those fluxes associated with axes "above" NDIM.
        !       While faceZ is garbage for NDIM <= 2 and, therefore, should
        !       not be used at all by the following routine, we still want
        !       indexing into it by tile to yield a valid array.
        CALL hy_updateSolutionHll_oacc(lo_loc_n, hi_loc_n,         &
                                       faceX_all_d(:, :, :, :, n), &
                                       faceY_all_d(:, :, :, :, n), &
                                       faceZ_all_d(:, :, :, :, n), &
                                       U_all_d(:, :, :, :, n))
    end do
    !$acc end parallel loop

    !$acc  parallel loop gang default(none)               &
    !$acc&                    private(lo_loc_n, hi_loc_n) &
    !$acc&                    async(dataQ_h)
    do n = 1, nTiles_d
        lo_loc_n(:) = lo_all_d(:, n) - loGC_all_d(:, n) + 1
        hi_loc_n(:) = hi_all_d(:, n) - loGC_all_d(:, n) + 1

        CALL eos_idealGammaDensIe_oacc(lo_loc_n, hi_loc_n,     &
                                       U_all_d(:, :, :, :, n))
    end do
    !$acc end parallel loop

    !$acc wait(dataQ_h)

!$acc end data

end subroutine hy_advanceSolutionHll_3_packet_oacc

