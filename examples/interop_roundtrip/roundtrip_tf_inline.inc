            block
                integer :: mh_i, mh_j, mh_k, mh_variable
                real(MILHOJA_REAL) :: mh_checksum
                real(MILHOJA_REAL) :: mh_increment
                real(MILHOJA_REAL) :: mh_metadata

                mh_metadata = real( &
                    tile_lbound_d(1, n) + &
                    tile_lbound_d(2, n) + &
                    tile_lbound_d(3, n), &
                    MILHOJA_REAL)
                mh_checksum = 0.0_MILHOJA_REAL

#ifdef ROUNDTRIP_VARIABLE_FIRST
                do mh_variable = 1, 4
                    do mh_k = 1, 1 + 2 * MILHOJA_K3D
                        do mh_j = 1, 2 + 2 * MILHOJA_K2D
                            do mh_i = 1, 3 + 2 * MILHOJA_K1D
                                mh_increment = real( &
                                    100 * mh_variable + 10 * mh_j + mh_i, &
                                    MILHOJA_REAL) + mh_metadata
                                CC_1_d(mh_variable, mh_i, mh_j, mh_k, n) = &
                                    external_scale_d * &
                                    CC_1_d(mh_variable, mh_i, mh_j, mh_k, n) + &
                                    mh_increment
                                mh_checksum = mh_checksum + &
                                    CC_1_d(mh_variable, mh_i, mh_j, mh_k, n)
                            end do
                        end do
                    end do
                end do
                scratch_checksum_d(1, n) = mh_checksum
                CC_1_d(1, 1, 1, 1, n) = mh_checksum
#else
                do mh_variable = 1, 4
                    do mh_k = 1, 1 + 2 * MILHOJA_K3D
                        do mh_j = 1, 2 + 2 * MILHOJA_K2D
                            do mh_i = 1, 3 + 2 * MILHOJA_K1D
                                mh_increment = real( &
                                    100 * mh_variable + 10 * mh_j + mh_i, &
                                    MILHOJA_REAL) + mh_metadata
                                CC_1_d(mh_i, mh_j, mh_k, mh_variable, n) = &
                                    external_scale_d * &
                                    CC_1_d(mh_i, mh_j, mh_k, mh_variable, n) + &
                                    mh_increment
                                mh_checksum = mh_checksum + &
                                    CC_1_d(mh_i, mh_j, mh_k, mh_variable, n)
                            end do
                        end do
                    end do
                end do
                scratch_checksum_d(1, n) = mh_checksum
                CC_1_d(1, 1, 1, 1, n) = mh_checksum
#endif
            end block
