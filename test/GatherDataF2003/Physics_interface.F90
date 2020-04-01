#include "constants.h"
#include "Flash.h"

module Physics_interface
    interface
        subroutine Physics_op1_executeTask1_Tile(tId, tilePtr) bind(c)
            use iso_c_binding, ONLY : C_INT, C_PTR
            implicit none
            integer(C_INT), intent(IN), value :: tId
            type(C_PTR),    intent(IN), value :: tilePtr
        end subroutine Physics_op1_executeTask1_Tile
    end interface

    interface
        subroutine Physics_op1_executeTask2_Tile(tId, tilePtr) bind(c)
            use iso_c_binding, ONLY : C_INT, C_PTR
            implicit none
            integer(C_INT), intent(IN), value :: tId
            type(C_PTR),    intent(IN), value :: tilePtr
        end subroutine Physics_op1_executeTask2_Tile
    end interface

    interface
        subroutine Physics_op1_executeTask3_Tile(tId, tilePtr) bind(c)
            use iso_c_binding, ONLY : C_INT, C_PTR
            implicit none
            integer(C_INT), intent(IN), value :: tId
            type(C_PTR),    intent(IN), value :: tilePtr
        end subroutine Physics_op1_executeTask3_Tile
    end interface
end module Physics_interface

