from libc.stdint cimport int64_t

# Caller owns/validates all buffers. coeff/exponents: [center,3,N+1];
# scaled: [center], u/indices: [Q], out: [Q,3]. Return 1 on range failure.
cdef int series_horner_batch(const double* coeff,const int64_t* exponents,
                            const unsigned char* scaled,const double* u,
                            const Py_ssize_t* indices,Py_ssize_t Q,int N,
                            double* out) noexcept nogil

from ._basis_kernels cimport BasisTable

cdef struct SeriesWorkspace:
    int order
    int nf
    double* fast
    double* weights
    unsigned char* active
    void* extended

# One workspace per concurrent caller. Allocate once, reuse across centers.
cdef SeriesWorkspace* series_workspace_new(int order,int nf) noexcept nogil
cdef void series_workspace_free(SeriesWorkspace* workspace) noexcept nogil
# maps: exact-precombined sign maps [4,3*nf,d], theta [d].
# out/exponents: [3,order+1]. Return 0 fast success, 1 scaled success,
# -1 invalid geometry/workspace, -2 unsupported numerical range.
cdef int series_construct_center(const BasisTable* basis,const double* maps,
                                 const double* theta,double center,double scale,
                                 SeriesWorkspace* workspace,double* out,
                                 int64_t* exponents) noexcept nogil
