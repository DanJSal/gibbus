# distutils: language = c++
"""Independent derivative jets and batched Horner kernels. All arithmetic is nogil.

Python entry points are private: basis_series validates dimensions and geometry.
The fused kernels compile to ordinary double and extended-exponent versions.
"""
import numpy as np
cimport numpy as cnp
from libc.math cimport exp, expm1, pow, sqrt, hypot, fabs, isfinite, frexp, ldexp
from libc.stdint cimport int64_t
from libc.stdlib cimport malloc, free
from ._basis_kernels cimport BasisKernel, BasisTable, FN_MONOMIAL, FN_LOG, FN_SOFT, FN_EXP, SIDE_LOWER

cdef extern from "_series_number.hpp" namespace "gibbus_series" nogil:
    cppclass Wide:
        Wide() noexcept
        Wide(double) noexcept
        Wide(double, int64_t) noexcept
        double m
        int64_t e
        Wide operator+(Wide) noexcept
        Wide operator-(Wide) noexcept
        Wide operator-() noexcept
        Wide operator*(Wide) noexcept
        Wide operator/(Wide) noexcept
        bint operator==(Wide) noexcept
        bint operator!=(Wide) noexcept
    double value(Wide) noexcept
    Wide wide_exp(Wide) noexcept
    Wide wide_sqrt(Wide) noexcept
    Wide wide_pow(Wide, double) noexcept
    Wide* allocate_wide(size_t) noexcept
    void free_wide(Wide*) noexcept
    cppclass RangeGuard:
        RangeGuard() noexcept
        int failed() noexcept

ctypedef fused Real:
    double
    Wide

cdef inline Real rexp(Real x) noexcept nogil:
    if Real is double:
        return exp(x)
    else:
        return wide_exp(x)

cdef inline Real rpow(Real x, double p) noexcept nogil:
    if Real is double:
        return pow(x,p)
    else:
        return wide_pow(x,p)

cdef inline Real rhypot(Real x, Real y) noexcept nogil:
    if Real is double:
        return hypot(x,y)
    else:
        return wide_sqrt(x*x+y*y)

cdef inline void add(Real x, Real* s, Real* correction) noexcept nogil:
    # Kahan summation; setup disables contraction for this extension.
    cdef Real y=x-correction[0]
    cdef Real t=s[0]+y
    correction[0]=(t-s[0])-y
    s[0]=t

cdef void atom(const BasisTable* t, int j, double x, double ell,
               int N, Real* f) noexcept nogil:
    cdef int n,r,k,m=t.degree[j], stride=N+1
    cdef Real z=(<Real>x-<Real>t.c)/<Real>t.h
    cdef Real s=<Real>ell/<Real>t.h
    cdef Real v,e,w,R,seed,fac,dist,lam,power
    cdef Real H[6]
    cdef double p=t.power[j], eps=1.0
    for k in range(5*stride): f[k]=<Real>0.0
    if t.kind[j]==FN_MONOMIAL:
        fac=<Real>1.0
        for r in range(5):
            if r>m: break
            # Shift each analytic derivative with polynomial Horner; no z division.
            f[r*stride]=fac
            for k in range(m-r):
                for n in range(min(k+1,N),0,-1):
                    f[r*stride+n]=z*f[r*stride+n]+s*f[r*stride+n-1]
                f[r*stride]=z*f[r*stride]
            fac=fac*<Real>(m-r)/<Real>t.h
    elif t.kind[j]==FN_LOG:
        if t.side[j]!=SIDE_LOWER: eps=-1.0
        if eps>0: dist=<Real>x-<Real>t.lower
        else: dist=<Real>t.upper-<Real>x
        fac=<Real>(-eps)/dist
        seed=fac*fac
        for r in range(2,5):
            f[r*stride]=seed
            for n in range(N):
                f[r*stride+n+1]=f[r*stride+n]*fac*<Real>ell*<Real>(n+r)/<Real>(n+1)
            seed=seed*fac*<Real>r
    elif t.kind[j]==FN_EXP:
        lam=<Real>t.rate[j]
        if t.side[j]==SIDE_LOWER: lam=-lam
        seed=rexp(lam*(<Real>x-<Real>t.location[j]))
        for r in range(5):
            f[r*stride]=seed
            for n in range(N):
                f[r*stride+n+1]=f[r*stride+n]*lam*<Real>ell/<Real>(n+1)
            seed=seed*lam
    else:
        R=rhypot(<Real>x-<Real>t.c,<Real>sqrt(t.delta[j]))
        v=(<Real>x-<Real>t.c)/R
        e=<Real>sqrt(t.delta[j])/R
        e=e*e
        w=<Real>ell/R
        z=v*v
        H[0]=<Real>1.0
        H[1]=<Real>p*v
        H[2]=<Real>p*(e+<Real>(p-1)*z)
        H[3]=<Real>(p*(p-2))*v*(<Real>3.0*e+<Real>(p-1)*z)
        H[4]=<Real>(p*(p-2))*(<Real>3.0*e*e+<Real>(6*(p-3))*e*z+<Real>((p-1)*(p-3))*z*z)
        H[5]=<Real>(p*(p-2)*(p-4))*v*(<Real>15.0*e*e+<Real>(10*(p-3))*e*z+<Real>((p-1)*(p-3))*z*z)
        seed=rpow(R/<Real>t.h,p)
        for r in range(5):
            f[r*stride]=seed*H[r]
            if N: f[r*stride+1]=seed*H[r+1]*w
            for n in range(N-1):
                f[r*stride+n+2]=-(<Real>((n+1)*(2*n+2*r+2-p))*v*w*f[r*stride+n+1]+<Real>((n+r+1)*(n+r-p))*w*w*f[r*stride+n])/<Real>((n+2.0)*(n+1.0))
            seed=seed/R

cdef void gate(double a, double b, double x, double ell, int N,
               int sigma, Real* g) noexcept nogil:
    cdef int n,j,r,S=N+1
    cdef Real k=rexp(<Real>(-2.0*sigma)*<Real>a*(<Real>x-<Real>b))
    cdef Real kap=<Real>(-2.0*sigma)*<Real>a
    cdef Real den=<Real>1.0+k
    cdef Real mag=(<Real>1.0-k)/den
    cdef Real total,corr,power
    cdef double v0=a*(x-b), em
    if fabs(v0)<=0.5:
        em=expm1(-2*fabs(v0))
        mag=<Real>(-em/(2+em))
    g[0]=k/den
    g[S]=k/(den*den)
    for n in range(N+1):
        if n:
            total=mag*g[n]; corr=<Real>0.0
            for j in range(1,n): add(-g[j]*g[n-j],&total,&corr)
            g[S+n]=total
        if n<N: g[n+1]=kap*<Real>ell*g[S+n]/<Real>(n+1)
    for n in range(N+1):
        total=mag*g[S+n]; corr=<Real>0.0
        for j in range(1,n+1): add(-<Real>2.0*g[j]*g[S+n-j],&total,&corr)
        g[2*S+n]=total
    for n in range(N+1):
        total=g[S+n]; corr=<Real>0.0
        for j in range(n+1): add(-<Real>6.0*g[S+j]*g[S+n-j],&total,&corr)
        g[3*S+n]=total
        total=g[2*S+n]; corr=<Real>0.0
        for j in range(n+1): add(-<Real>12.0*g[S+j]*g[2*S+n-j],&total,&corr)
        g[4*S+n]=total
    power=<Real>1.0
    for r in range(5):
        for n in range(N+1):
            g[r*S+n]=g[r*S+n]*power
            if x==b and n+r>0 and (n+r)%2==0: g[r*S+n]=<Real>0.0
        power=power*kap

cdef void fused_exp(double lam, double loc, double a, double b,
                    double x, double ell, int N, int sigma,
                    Real* gatejet, Real* out, Real* scratch) noexcept nogil:
    # Y=exp(lambda*(x-loc))*W; derivatives Y*P_r(W), constructed independently.
    cdef int n,j,r,k,S=N+1
    cdef double kap=-2.0*sigma*a
    cdef Real rate=<Real>lam+<Real>kap
    cdef Real exponent=rate*(<Real>x-<Real>b)+<Real>lam*(<Real>b-<Real>loc)
    cdef Real total,corr
    cdef Real p[25]
    cdef Real* y=scratch
    cdef Real* powers=scratch+S
    cdef Real* poly=scratch+6*S
    for j in range(25): p[j]=<Real>0.0
    p[0]=<Real>1.0
    y[0]=rexp(exponent)*(<Real>1.0-gatejet[0])
    for n in range(N):
        total=(rate-<Real>kap*gatejet[0])*y[n]; corr=<Real>0.0
        for j in range(1,n+1): add(-<Real>kap*gatejet[j]*y[n-j],&total,&corr)
        y[n+1]=<Real>ell*total/<Real>(n+1)
    for r in range(4):
        for j in range(r+2):
            total=<Real>0.0
            if j<=r: total=(rate+<Real>(j*kap))*p[5*r+j]
            if j: total=total-<Real>(j*kap)*p[5*r+j-1]
            p[5*(r+1)+j]=total
    for n in range(5*S): powers[n]=<Real>0.0
    powers[0]=<Real>1.0
    for n in range(S): powers[S+n]=gatejet[n]
    for r in range(2,5):
        for n in range(S):
            total=<Real>0.0; corr=<Real>0.0
            for j in range(n+1): add(powers[(r-1)*S+j]*gatejet[n-j],&total,&corr)
            powers[r*S+n]=total
    for r in range(2,5):
        for n in range(S):
            total=<Real>0.0; corr=<Real>0.0
            for j in range(r+1): add(p[5*r+j]*powers[j*S+n],&total,&corr)
            poly[n]=total
        for n in range(S):
            total=<Real>0.0; corr=<Real>0.0
            for j in range(n+1): add(y[j]*poly[n-j],&total,&corr)
            out[(r-2)*S+n]=total

cdef inline int choose(int r,int k) noexcept nogil:
    if k==0 or k==r: return 1
    if r==4 and k==2: return 6
    return r

cdef void product(Real* f, Real* g, int degree, int N, Real* out) noexcept nogil:
    cdef int n,r,k,j,end,S=N+1
    cdef Real total,corr
    for r in range(2,5):
        for n in range(S):
            total=<Real>0.0; corr=<Real>0.0
            for k in range(r+1):
                end=n
                if degree>=0: end=min(n,degree-k)
                for j in range(end+1):
                    add(<Real>choose(r,k)*f[k*S+j]*g[(r-k)*S+n-j],&total,&corr)
            out[(r-2)*S+n]=total

cdef int center_core(const BasisTable* t, double x, double ell, int N,
                     int sl,int su, const Real* weights, const unsigned char* active, Real* out,
                     Real* cache, Real* work) noexcept nogil:
    # weights[nf,3]: exact sign-map contraction, or NULL for primitive cache.
    cdef int j,ch,n,r,S=N+1,count=3*S,degree
    cdef Real* f=work
    cdef Real* gl=f+5*S
    cdef Real* gu=gl+5*S
    cdef Real* prod=gu+5*S
    cdef Real* tmp=prod+3*S
    cdef Real* comp=tmp+7*S
    cdef Real* dest
    cdef double a,b,lam
    cdef bint need_l=False,need_u=False
    cdef Real w
    for n in range(count):
        out[n]=<Real>0.0
        comp[n]=<Real>0.0
    for j in range(t.nf):
        if active[3*j+1] and (weights==NULL or weights[3*j+1]!=<Real>0.0): need_l=True
        if active[3*j+2] and (weights==NULL or weights[3*j+2]!=<Real>0.0): need_u=True
    if t.has_l and need_l: gate(t.al,t.xl,x,ell,N,sl,gl)
    if t.has_u and need_u: gate(-t.au,t.xu,x,ell,N,su,gu)
    for j in range(t.nf):
        if not t.fn_used[j] or not (active[3*j] or active[3*j+1] or active[3*j+2]): continue
        if weights!=NULL and weights[3*j]==<Real>0.0 and weights[3*j+1]==<Real>0.0 and weights[3*j+2]==<Real>0.0: continue
        # Exponentials with only gated-complement contributions need no raw seed.
        if (weights==NULL and active[3*j]) or (weights!=NULL and weights[3*j]!=<Real>0.0) or t.kind[j]!=FN_EXP:
            atom(t,j,x,ell,N,f)
        for ch in range(3):
            if not active[3*j+ch]: continue
            if ch==1 and not t.has_l: continue
            if ch==2 and not t.has_u: continue
            if ch and t.kind[j]==FN_LOG: continue
            if weights!=NULL:
                w=weights[3*j+ch]
                if w==<Real>0.0: continue
            else:
                w=<Real>1.0
            dest=prod
            if ch==0:
                for n in range(count): prod[n]=f[2*S+n]
            elif t.kind[j]==FN_EXP:
                lam=t.rate[j]
                if t.side[j]==SIDE_LOWER: lam=-lam
                if ch==1: fused_exp(lam,t.location[j],t.al,t.xl,x,ell,N,sl,gl,prod,tmp)
                else: fused_exp(lam,t.location[j],-t.au,t.xu,x,ell,N,su,gu,prod,tmp)
            else:
                degree=-1
                if t.kind[j]==FN_MONOMIAL: degree=t.degree[j]
                if ch==1: product(f,gl,degree,N,prod)
                else: product(f,gu,degree,N,prod)
            if weights==NULL:
                for n in range(count): cache[(3*j+ch)*count+n]=prod[n]
            else:
                for n in range(count): add(<Real>w*prod[n],out+n,comp+n)
    return 0

cdef int normal_center(const BasisTable* t, double x, double ell,int N,int sl,int su,
                       const double* weights,const unsigned char* active,double* out,double* cache,double* work) noexcept nogil:
    cdef RangeGuard guard
    cdef int n,count=3*(N+1)
    center_core(t,x,ell,N,sl,su,weights,active,out,cache,work)
    if guard.failed(): return 1
    if weights==NULL:
        for n in range(t.nf*3*count):
            if not isfinite(cache[n]): return 1
    else:
        for n in range(count):
            if not isfinite(out[n]): return 1
    return 0

cdef void contract_weights(const double* maps,const double* theta,int nf,int d,
                           Real* weights) noexcept nogil:
    cdef int s,j,k
    cdef Real total,corr
    for s in range(4):
        for j in range(3*nf):
            total=<Real>0.0;corr=<Real>0.0
            for k in range(d): add(<Real>maps[(s*3*nf+j)*d+k]*<Real>theta[k],&total,&corr)
            weights[s*3*nf+j]=total

cdef int normal_weights(const double* maps,const double* theta,int nf,int d,
                        double* weights) noexcept nogil:
    cdef RangeGuard guard
    contract_weights(maps,theta,nf,d,weights)
    return guard.failed()

# The wrapper owns all buffers. There is no allocation inside a center loop.
def _construct(BasisKernel kernel, const double[:,::1] centers,
               const int[:,::1] signs, const double[:,:,::1] maps,
               const double[::1] theta, bint cache=False):
    if centers.shape[0]<1 or centers.shape[1]!=3:
        raise ValueError("invalid center dimensions")
    cdef int N=int(centers[0,2])
    if N<0 or N>4096 or np.any(np.asarray(centers)[:,2]!=N):
        raise ValueError("invalid series order")
    if signs.shape[0]!=centers.shape[0] or signs.shape[1]!=2:
        raise ValueError("invalid sign dimensions")
    if maps.shape[0]!=4 or maps.shape[1]!=3*kernel.t.nf or maps.shape[2]!=kernel.t.d:
        raise ValueError("invalid map dimensions")
    if not cache and theta.shape[0]!=kernel.t.d:
        raise ValueError("invalid coefficient dimensions")
    cdef int C=centers.shape[0], nf=kernel.t.nf,d=kernel.t.d,S=N+1
    cdef int width=3*S, outwidth=width*3*nf if cache else width
    cdef object mo=np.zeros((C,outwidth),dtype=np.float64)
    cdef object eo=np.zeros((C,outwidth),dtype=np.int64)
    cdef double[:,::1] mv=mo
    cdef int64_t[:,::1] ev=eo
    cdef object wo=np.empty((4*3*nf,),dtype=np.float64)
    cdef double[::1] wv=wo
    cdef double* work=NULL
    cdef Wide* widework=NULL
    cdef Wide* wideweights=NULL
    cdef const Wide* wp
    cdef object active_obj=np.ascontiguousarray(np.any(np.asarray(maps)!=0,axis=2),dtype=np.uint8)
    cdef const unsigned char[:,::1] active=active_obj
    cdef bint weight_fallback=False
    cdef double* weights
    cdef Wide* wideout
    cdef Wide* widecache
    cdef int i,k,idx,status=0,fallbacks=0
    work=<double*>malloc((28*S+width)*sizeof(double))
    widework=allocate_wide(28*S+width+outwidth+4*3*nf)
    if work==NULL or widework==NULL:
        free(work);free_wide(widework)
        raise MemoryError()
    wideweights=widework+28*S+width+outwidth
    try:
        with nogil:
            if not cache:
                weight_fallback=normal_weights(&maps[0,0,0],&theta[0],nf,d,&wv[0])!=0
                if weight_fallback: contract_weights(&maps[0,0,0],&theta[0],nf,d,wideweights)
            for i in range(C):
                idx=(2 if signs[i,0]>0 else 0)+(1 if signs[i,1]>0 else 0)
                weights=<double*>NULL if cache else &wv[idx*3*nf]
                if weight_fallback or normal_center(&kernel.t,centers[i,0],centers[i,1],N,
                                 signs[i,0],signs[i,1],weights,&active[idx,0],work+28*S,
                                 &mv[i,0] if cache else NULL,work):
                    fallbacks+=1
                    wideout=widework+28*S
                    widecache=wideout+width
                    if not cache and not weight_fallback:
                        contract_weights(&maps[0,0,0],&theta[0],nf,d,wideweights)
                    wp=<const Wide*>NULL if cache else wideweights+idx*3*nf
                    for k in range(outwidth): widecache[k]=Wide(0.0)
                    center_core(&kernel.t,centers[i,0],centers[i,1],N,
                                signs[i,0],signs[i,1],wp,&active[idx,0],wideout,widecache,widework)
                    if not cache: widecache=wideout
                    for k in range(outwidth):
                        mv[i,k]=widecache[k].m;ev[i,k]=widecache[k].e
                        if not isfinite(mv[i,k]): status=1
                elif not cache:
                    for k in range(width): mv[i,k]=work[28*S+k]

    finally:
        free(work);free_wide(widework)
    if status: raise OverflowError("series construction exceeded supported arithmetic range")
    return mo,eo,fallbacks


cdef int combine_normal(const double* maps,const double* theta,const int* signs,
                        const double* primitive,const int64_t* exponents,
                        int C,int nf,int d,int width,double* weights,
                        double* out) noexcept nogil:
    cdef RangeGuard guard
    cdef int c,n,j,idx,base
    cdef double total,corr,w
    contract_weights(maps,theta,nf,d,weights)
    for c in range(C):
        idx=(2 if signs[2*c]>0 else 0)+(1 if signs[2*c+1]>0 else 0)
        for n in range(width):
            total=0;corr=0
            for j in range(3*nf):
                w=weights[idx*3*nf+j]
                if w==0: continue
                base=(c*3*nf+j)*width+n
                if exponents[base]!=0: return 1
                add(w*primitive[base],&total,&corr)
            out[c*width+n]=total
            if not isfinite(total): return 1
    return guard.failed()


def _combine(const double[:,:,::1] maps,const double[::1] theta,
             const int[:,::1] signs,const double[:,::1] primitive,
             const int64_t[:,::1] exponents,int nf,int N):
    cdef int C=primitive.shape[0],d=theta.shape[0],width=3*(N+1)
    if C<1 or nf<1 or N<0 or N>4096 or d<1:
        raise ValueError("invalid series dimensions")
    if maps.shape[0]!=4 or maps.shape[1]!=nf*3 or maps.shape[2]!=d:
        raise ValueError("invalid map dimensions")
    if signs.shape[0]!=C or signs.shape[1]!=2 or primitive.shape[1]!=nf*3*width:
        raise ValueError("invalid cache dimensions")
    if exponents.shape[0]!=C or exponents.shape[1]!=primitive.shape[1]:
        raise ValueError("invalid exponent dimensions")
    cdef object mo=np.empty((C,width)), eo=np.zeros((C,width),dtype=np.int64)
    cdef double[:,::1] out=mo
    cdef int64_t[:,::1] expout=eo
    cdef object wo=np.empty(4*3*nf)
    cdef double[::1] weights=wo
    cdef Wide* wideweights=allocate_wide(4*3*nf)
    cdef int c,n,j,idx,base,status=0
    cdef Wide wt,wc,w
    if wideweights==NULL: raise MemoryError()
    try:
        with nogil:
            if combine_normal(&maps[0,0,0],&theta[0],&signs[0,0],&primitive[0,0],
                              &exponents[0,0],C,nf,d,width,&weights[0],&out[0,0]):
                contract_weights(&maps[0,0,0],&theta[0],nf,d,wideweights)
                for c in range(C):
                    idx=(2 if signs[c,0]>0 else 0)+(1 if signs[c,1]>0 else 0)
                    for n in range(width):
                        wt=Wide(0.0);wc=Wide(0.0)
                        for j in range(3*nf):
                            w=wideweights[idx*3*nf+j]
                            if w!=Wide(0.0):
                                base=j*width+n
                                add(w*Wide(primitive[c,base],exponents[c,base]),&wt,&wc)
                        out[c,n]=wt.m;expout[c,n]=wt.e
                        if not isfinite(wt.m): status=1
    finally:
        free_wide(wideweights)
    if status: raise OverflowError("series contraction exceeded supported arithmetic range")
    return mo,eo

cdef inline double horner(const double* coeff,int N,double u) noexcept nogil:
    cdef int n
    cdef double v=coeff[N]
    for n in range(N-1,-1,-1): v=v*u+coeff[n]
    return v

cdef double guarded_horner(const double* m,const int64_t* e,int N,double u,
                          bint scaled) noexcept nogil:
    cdef RangeGuard guard
    cdef double v
    cdef int n
    cdef Wide w
    if not scaled:
        v=horner(m,N,u)
        if isfinite(v) and not guard.failed(): return v
    w=Wide(m[N],e[N])
    for n in range(N-1,-1,-1): w=w*Wide(u)+Wide(m[n],e[n])
    return value(w)

cdef inline void horner_four(const double* coeff,int N,const double* u,
                             double* out) noexcept nogil:
    # Four independent points expose SIMD/ILP without changing Horner ordering.
    cdef double v[12]
    cdef int r,k,n,S=N+1
    for r in range(3):
        for k in range(4): v[4*r+k]=coeff[r*S+N]
    for n in range(N-1,-1,-1):
        for r in range(3):
            for k in range(4): v[4*r+k]=v[4*r+k]*u[k]+coeff[r*S+n]
    for k in range(4):
        for r in range(3): out[3*k+r]=v[4*r+k]

cdef int normal_evaluate(const double* coeff,const int64_t* exponents,
                         const unsigned char* scaled,const double* u,
                         const Py_ssize_t* indices,Py_ssize_t Q,int N,
                         double* out) noexcept nogil:
    cdef RangeGuard guard
    cdef Py_ssize_t i,base
    cdef int r,c,S=N+1
    cdef int n
    cdef double v0,v1,v2,z
    i=0
    while i<Q:
        c=indices[i]
        if not scaled[c] and i+3<Q and indices[i+1]==c and indices[i+2]==c and indices[i+3]==c:
            horner_four(coeff+c*3*S,N,u+i,out+3*i)
            i+=4
            continue
        if scaled[c]:
            for r in range(3):
                base=(c*3+r)*S
                out[3*i+r]=guarded_horner(coeff+base,exponents+base,N,u[i],True)
        else:
            base=c*3*S;z=u[i]
            v0=coeff[base+N];v1=coeff[base+S+N];v2=coeff[base+2*S+N]
            for n in range(N-1,-1,-1):
                v0=v0*z+coeff[base+n]
                v1=v1*z+coeff[base+S+n]
                v2=v2*z+coeff[base+2*S+n]
            out[3*i]=v0;out[3*i+1]=v1;out[3*i+2]=v2
        i+=1
    return guard.failed()

cdef int series_horner_batch(const double* coeff,const int64_t* exponents,
                            const unsigned char* scaled,const double* u,
                            const Py_ssize_t* indices,Py_ssize_t Q,int N,
                            double* out) noexcept nogil:
    """Allocation-free C API; caller validates buffers, indices and radius.

    Return 0 on success, 1 if a result exceeds float64 range. Output layout Q*3.
    This evaluates polynomials, with no finite-order truncation certificate.
    """
    cdef Py_ssize_t i,base
    cdef int r,c,S=N+1,status=0
    cdef bint retry=normal_evaluate(coeff,exponents,scaled,u,indices,Q,N,out)!=0
    for i in range(Q):
        for r in range(3):
            if retry or not isfinite(out[3*i+r]):
                base=(indices[i]*3+r)*S
                out[3*i+r]=guarded_horner(coeff+base,exponents+base,N,u[i],True)
            if not isfinite(out[3*i+r]): status=1
    return status


def _evaluate(const double[:,:,::1] coeff,const int64_t[:,:,::1] exponents,
              const unsigned char[::1] scaled,const double[::1] u,
              const cnp.intp_t[::1] indices):
    cdef int N=coeff.shape[2]-1,status=0
    cdef object result=np.empty((u.shape[0],3))
    cdef double[:,::1] out=result
    if coeff.shape[1]!=3 or N<0 or indices.shape[0]!=u.shape[0] or scaled.shape[0]!=coeff.shape[0]:
        raise ValueError("invalid series buffer dimensions")
    if exponents.shape[0]!=coeff.shape[0] or exponents.shape[1]!=3 or exponents.shape[2]!=N+1:
        raise ValueError("invalid exponent buffer dimensions")
    if u.shape[0]==0: return result
    if np.any(np.asarray(indices)<0) or np.any(np.asarray(indices)>=coeff.shape[0]):
        raise IndexError("invalid center index")
    with nogil:
        status=series_horner_batch(&coeff[0,0,0],&exponents[0,0,0],&scaled[0],
                                  &u[0],<const Py_ssize_t*>&indices[0],u.shape[0],N,&out[0,0])
    if status: raise OverflowError("series value is not representable as float64")
    return result


cdef SeriesWorkspace* series_workspace_new(int order,int nf) noexcept nogil:
    cdef SeriesWorkspace* w
    cdef size_t S
    if order<0 or order>4096 or nf<1: return NULL
    S=order+1
    w=<SeriesWorkspace*>malloc(sizeof(SeriesWorkspace))
    if w==NULL: return NULL
    w.order=order;w.nf=nf
    w.fast=<double*>malloc(31*S*sizeof(double))
    w.weights=<double*>malloc(<size_t>12*nf*sizeof(double))
    w.active=<unsigned char*>malloc(<size_t>3*nf)
    w.extended=<void*>allocate_wide(31*S+<size_t>12*nf)
    if w.fast==NULL or w.weights==NULL or w.active==NULL or w.extended==NULL:
        series_workspace_free(w)
        return NULL
    return w


cdef void series_workspace_free(SeriesWorkspace* w) noexcept nogil:
    if w!=NULL:
        free(w.fast);free(w.weights);free(w.active)
        free_wide(<Wide*>w.extended)
        free(w)


cdef int series_construct_center(const BasisTable* t,const double* maps,
                                 const double* theta,double x,double ell,
                                 SeriesWorkspace* w,double* out,
                                 int64_t* exponents) noexcept nogil:
    cdef int sl=1,su=1,idx,j,k,n,S,N
    cdef bint fallback
    cdef double radius,dist
    cdef Wide* wide
    cdef Wide* wideout
    cdef Wide* wideweights
    if w==NULL or t==NULL or w.nf!=t.nf: return -1
    if not isfinite(x) or not isfinite(ell) or ell<=0 or x<=t.lower or x>=t.upper: return -1
    N=w.order;S=N+1
    # Conservative structural radii, independent of current theta.
    if t.has_l:
        if x<t.xl: sl=-1
        if ell>=hypot(x-t.xl,1.5707963267948966/t.al): return -1
    if t.has_u:
        if x>t.xu: su=-1
        if ell>=hypot(x-t.xu,1.5707963267948966/t.au): return -1
    for j in range(t.nf):
        if not t.fn_used[j]: continue
        if t.kind[j]==FN_LOG:
            dist=x-t.lower if t.side[j]==SIDE_LOWER else t.upper-x
            if ell>=dist: return -1
        elif t.kind[j]==FN_SOFT:
            if ell>=hypot(x-t.c,sqrt(t.delta[j])): return -1
    idx=(2 if sl>0 else 0)+(1 if su>0 else 0)
    for j in range(3*t.nf):
        w.active[j]=0
        for k in range(t.d):
            if maps[(idx*3*t.nf+j)*t.d+k]!=0: w.active[j]=1;break
    fallback=normal_weights(maps,theta,t.nf,t.d,w.weights)!=0
    if not fallback:
        fallback=normal_center(t,x,ell,N,sl,su,w.weights+idx*3*t.nf,
                               w.active,w.fast+28*S,NULL,w.fast)!=0
    if not fallback:
        for n in range(3*S): out[n]=w.fast[28*S+n];exponents[n]=0
        return 0
    wide=<Wide*>w.extended
    wideout=wide+28*S
    wideweights=wide+31*S
    contract_weights(maps,theta,t.nf,t.d,wideweights)
    center_core(t,x,ell,N,sl,su,wideweights+idx*3*t.nf,w.active,wideout,<Wide*>NULL,wide)
    for n in range(3*S):
        if not isfinite(wideout[n].m): return -2
        out[n]=wideout[n].m;exponents[n]=wideout[n].e
    return 1


def _construct_capi(BasisKernel kernel,const double[:,:,::1] maps,
                    const double[::1] theta,double x,double ell,int order):
    """Exercise the public nogil C interface; used by integration tests."""
    cdef SeriesWorkspace* w
    cdef int status
    if order<0 or order>4096 or maps.shape[0]!=4 or maps.shape[1]!=3*kernel.t.nf or maps.shape[2]!=kernel.t.d or theta.shape[0]!=kernel.t.d:
        raise ValueError("invalid C API input dimensions")
    cdef object mo=np.empty((3,order+1)),eo=np.empty((3,order+1),dtype=np.int64)
    cdef double[:,::1] m=mo
    cdef int64_t[:,::1] e=eo
    with nogil:
        w=series_workspace_new(order,kernel.t.nf)
        if w!=NULL:
            status=series_construct_center(&kernel.t,&maps[0,0,0],&theta[0],x,ell,w,&m[0,0],&e[0,0])
            series_workspace_free(w)
    if w==NULL: raise MemoryError()
    if status<0: raise ValueError("C API construction failed with status %d"%status)
    return mo,eo,status
