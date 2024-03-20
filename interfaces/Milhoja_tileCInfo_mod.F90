Module Milhoja_tileCInfo_mod
   use,intrinsic :: iso_c_binding, ONLY: c_ptr, C_INT, C_DOUBLE
   implicit none
#include "Milhoja.h"

   ! The layout of the Fortran derived type for C-compatible tile information
   ! must be the same, whether seen in the Flash-X code (where it is available
   ! in module Orchestration_interfaceTypeDecl) or here in the Milhoja library
   ! adapter code (where we name it Milhoja_tileCInfo_mod and define
   ! it in this file).
   ! We store the actual definition of the layout in the file
   ! "Milhoja_tileCInfo.finc" included below. Note that that Fortran layout
   ! MUST be kept compatible with the C layout in
   ! "Milhoja_FlashxrTileRaw.h" !
#define MDIM MILHOJA_MDIM

#define TYPENAME_PREFIXED(surname) TYPENAME_PREFIXED2(Milhoja,surname)
#include "Milhoja_tileCInfo.finc"

end Module Milhoja_tileCInfo_mod
