#ifndef MILHOJA_MACROS_H__
#define MILHOJA_MACROS_H__

#include "Milhoja.h"

#ifndef MILHOJA_NDIM
#error MILHOJA_NDIM needs to be defined
#endif

#if ((MILHOJA_NDIM!=1)&&(MILHOJA_NDIM!=2)&&(MILHOJA_NDIM!=3))
#error MILHOJA_NDIM needs to be in range 1-3
#endif


#if MILHOJA_NDIM==1
// Make a comma-separated list of MILHOJA_NDIM elements from a list of 3
#define LIST_NDIM(x,y,z) x

// Make a space-separated list of MILHOJA_NDIM elements from a list of 3
#define CONCAT_NDIM(x,y,z) x

#elif MILHOJA_NDIM==2
#define LIST_NDIM(x,y,z) x,y
#define CONCAT_NDIM(x,y,z) x y
#elif MILHOJA_NDIM==3
#define LIST_NDIM(x,y,z) x,y,z
#define CONCAT_NDIM(x,y,z) x y z
#endif

#endif
