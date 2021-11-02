module eos_cg_interface

    !!!!!----- ROUTINES WRITTEN BY CODE GENERATOR
    ! Task-specific static Fortran layer
    interface
        subroutine eos_idealGammaDensIe_oacc(lo, hi, U)
            implicit none
            integer, intent(IN)    :: lo(:)
            integer, intent(IN)    :: hi(:)
            real,    intent(INOUT) :: U(:,:,:,:)
        end subroutine eos_idealGammaDensIe_oacc
    end interface

end module eos_cg_interface

