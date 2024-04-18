module Tile_cpu_tf_hydro_c2f_mod
    implicit none
    private

    public :: instantiate_cpu_tf_hydro_wrapper_c
    public :: delete_cpu_tf_hydro_wrapper_c
    public :: acquire_scratch_cpu_tf_hydro_wrapper_c
    public :: release_scratch_cpu_tf_hydro_wrapper_c

    interface
        function instantiate_cpu_tf_hydro_wrapper_c( &
            C_external_hydro_op1_dt, &
            C_external_hydro_op1_eosMode, &
            C_wrapper &
        ) result(C_ierr) bind(c, name="instantiate_cpu_tf_hydro_wrapper_c")
            use iso_c_binding,     ONLY : C_PTR
            use milhoja_types_mod, ONLY : MILHOJA_INT, MILHOJA_REAL
            real(MILHOJA_REAL),   intent(IN), value :: C_external_hydro_op1_dt
            integer(MILHOJA_INT), intent(IN), value :: C_external_hydro_op1_eosMode
            type(C_PTR),          intent(IN)        :: C_wrapper
            integer(MILHOJA_INT)                    :: C_ierr
        end function instantiate_cpu_tf_hydro_wrapper_c

        function delete_cpu_tf_hydro_wrapper_c(C_wrapper) result(C_ierr) &
        bind(c, name="delete_cpu_tf_hydro_wrapper_c")
            use iso_c_binding,     ONLY : C_PTR
            use milhoja_types_mod, ONLY : MILHOJA_INT
            type(C_PTR),         intent(IN), value :: C_wrapper
            integer(MILHOJA_INT)                   :: C_ierr
        end function delete_cpu_tf_hydro_wrapper_c

        function acquire_scratch_cpu_tf_hydro_wrapper_c() result(C_ierr) &
        bind(c, name="acquire_scratch_cpu_tf_hydro_wrapper_c")
            use milhoja_types_mod, ONLY : MILHOJA_INT
            integer(MILHOJA_INT) :: C_ierr
        end function acquire_scratch_cpu_tf_hydro_wrapper_c

        function release_scratch_cpu_tf_hydro_wrapper_c() result(C_ierr) &
        bind(c, name="release_scratch_cpu_tf_hydro_wrapper_c")
            use milhoja_types_mod, ONLY : MILHOJA_INT
            integer(MILHOJA_INT) :: C_ierr
        end function release_scratch_cpu_tf_hydro_wrapper_c
    end interface

end module Tile_cpu_tf_hydro_c2f_mod
