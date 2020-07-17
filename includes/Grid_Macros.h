#ifndef GRID_MACROS_H__
#define GRID_MACROS_H__

#include "constants.h"

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

// Macro for iterating over all coordinates in the
// region defined by two IntVects lo and hi.
// Middle three arguments are the iteration variables,
// which can be used in 'function'.
// std::cout << "iterating over " << lo3[0]<<","<<lo3[1]<<","<<lo3[2] << " "<<hi3[0]<<","<<hi3[1]<<","<<hi3[2] <<std::endl;

#define ITERATE_REGION(lo,hi,i,j,k, function) {\
std::vector<int> lo_vec3 = lo.as3D(); \
std::vector<int> hi_vec3 = hi.as3D(); \
for(int i=lo_vec3[0];i<=hi_vec3[0];++i) {\
for(int j=lo_vec3[1];j<=hi_vec3[1];++j) {\
for(int k=lo_vec3[2];k<=hi_vec3[2];++k) {\
    function \
}}}}



#endif
