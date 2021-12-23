#ifndef MILHOJA_MACROS_H__
#define MILHOJA_MACROS_H__

#include "Milhoja.h"

// LIST_NDIM   makes a comma-separated list of MILHOJA_NDIM elements from a list of 3
// CONCAT_NDIM makes a space-separated list of MILHOJA_NDIM elements from a list of 3
#if   (MILHOJA_NDIM == 1)
#define   LIST_NDIM(x,y,z)    x
#define CONCAT_NDIM(x,y,z)    x
#elif (MILHOJA_NDIM == 2)
#define   LIST_NDIM(x,y,z)    x,y
#define CONCAT_NDIM(x,y,z)    x y
#elif (MILHOJA_NDIM == 3)
#define   LIST_NDIM(x,y,z)    x,y,z
#define CONCAT_NDIM(x,y,z)    x y z
#else
#error "MILHOJA_NDIM not defined or not in {1,2,3}"
#endif

#endif

