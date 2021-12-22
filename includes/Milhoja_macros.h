#ifndef MILHOJA_MACROS_H__
#define MILHOJA_MACROS_H__

#include "Milhoja.h"

#ifndef NDIM
#error NDIM needs to be defined
#endif

#if ((NDIM!=1)&&(NDIM!=2)&&(NDIM!=3))
#error NDIM needs to be in range 1-3
#endif


#if NDIM==1
// Make a comma-separated list of NDIM elements from a list of 3
#define LIST_NDIM(x,y,z) x

// Make a space-separated list of NDIM elements from a list of 3
#define CONCAT_NDIM(x,y,z) x

#elif NDIM==2
#define LIST_NDIM(x,y,z) x,y
#define CONCAT_NDIM(x,y,z) x y
#elif NDIM==3
#define LIST_NDIM(x,y,z) x,y,z
#define CONCAT_NDIM(x,y,z) x y z
#endif

#endif
