# cython: language_level=3
"""Small optional Cython accelerator for fixed-rate conditional tail quadrature.

All 96 Gauss-Laguerre nodes are consumed with stable shifted exponentials.
"""
from libc.math cimport exp, log, isfinite, INFINITY

def objective_gradient(double lb, double la, double ym, double mh, double rate,
                       double[::1] ys, double[::1] h, double[::1] logwr):
    cdef Py_ssize_t n=ys.shape[0],i
    cdef double b=exp(lb),a=exp(la),mx=-INFINITY,score,v
    cdef double sump=0.,sumy=0.,sumh=0.,p,nll
    if n == 0 or h.shape[0] != n or logwr.shape[0] != n:
        raise ValueError("tail quadrature arrays must have equal nonzero lengths")
    if not isfinite(rate) or rate <= 0:
        raise ValueError("tail quadrature rate must be positive and finite")
    for i in range(n):
        score=logwr[i]-b*ys[i]-a*h[i]
        if score>mx:mx=score
    if not isfinite(mx):
        return 1e30,(0.,0.)
    for i in range(n):
        score=logwr[i]-b*ys[i]-a*h[i]
        if not isfinite(score):continue
        p=exp(score-mx)
        sump+=p
        sumy+=p*ys[i]
        sumh+=p*h[i]
    if sump<=0.:
        return 1e30,(0.,0.)
    nll=b*ym+a*mh+mx+log(sump)-log(rate)
    if not isfinite(nll):
        return 1e30,(0.,0.)
    return nll,(b*(ym-sumy/sump),a*(mh-sumh/sump))
