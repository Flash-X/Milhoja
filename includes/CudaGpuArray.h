#ifndef CUDA_GPU_ARRAY_H__
#define CUDA_GPU_ARRAY_H__

#include <AMReX_Dim3.H>

// TODO: Could we just replace this class with an offline tool that just writes
// the index out?
// TODO: Get rid of Dim3 so that this is generally useful.
// TODO: Update so that they take hi instead of hi+1
struct CudaGpuArray {
    double*      p;
    unsigned int jstride;
    unsigned int kstride;
    unsigned int nstride;
    amrex::Dim3  begin;
    amrex::Dim3  end;  // end is hi + 1
    unsigned int ncomp;

    __host__ __device__
    CudaGpuArray(double* a_p, amrex::Dim3 const& a_begin, amrex::Dim3 const& a_end, unsigned int a_ncomp)
        : p(a_p),
          jstride(a_end.x-a_begin.x),
          kstride(jstride*(a_end.y-a_begin.y)),
          nstride(kstride*(a_end.z-a_begin.z)),
          begin(a_begin),
          end(a_end),
          ncomp(a_ncomp)
        {}

    __host__ __device__
    double& operator()(int i, int j, int k, int n) const {
        return p[(i-begin.x) + (j-begin.y)*jstride + (k-begin.z)*kstride + n*nstride];
    }
};

#endif

