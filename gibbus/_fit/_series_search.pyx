"""Cached Taylor candidate search. Heuristic only; never certifies feasibility."""
import numpy as np
cimport numpy as cnp
from libc.math cimport fabs,isfinite,NAN

cdef inline double polynomial(const double* c,int N,double u) noexcept nogil:
    cdef int n
    cdef double v=c[N]
    for n in range(N-1,-1,-1): v=v*u+c[n]
    return v

cdef inline int dot(const double* row,const double* theta,int d,double* result) noexcept nogil:
    cdef int j
    cdef double total=0.,corr=0.,y,tmp
    for j in range(d):
        y=row[j]*theta[j]-corr;tmp=total+y
        corr=(tmp-total)-y;total=tmp
    result[0]=total
    return 0 if isfinite(total) else -1

cdef int combine(SeriesSearchTable* t,const double* theta,int panel,int derivative,
                 int* ready) noexcept nogil:
    cdef int n,S=t.order+1,mask=1<<derivative
    cdef const double* row
    if ready[0]&mask: return 0
    for n in range(S):
        row=t.tensor+((panel*3+derivative)*S+n)*t.d
        if dot(row,theta,t.d,t.coefficients+derivative*S+n): return -1
    ready[0]|=mask
    if derivative==0: t.q2_builds+=1
    elif derivative==1: t.q3_builds+=1
    else: t.q4_builds+=1
    return 0

cdef void record(SeriesSearchTable* t,double x,double score,double* out,
                 int cap,int* count) noexcept nogil:
    cdef int j,k,worst=0
    if not isfinite(x) or not isfinite(score) or score>t.threshold: return
    for j in range(count[0]):
        if fabs(out[j]-x)<=t.tolerance:
            if score<t.scores[j]: out[j]=x;t.scores[j]=score
            return
    if count[0]<cap:
        j=count[0];count[0]+=1
    else:
        for k in range(1,cap):
            if t.scores[k]>t.scores[worst]: worst=k
        if score>=t.scores[worst]: return
        j=worst
    out[j]=x;t.scores[j]=score

cdef void remember(SeriesSearchTable* t,int p,double u,double score,
                   double tolerance) noexcept nogil:
    cdef int i,j,worst=0,base=8*p,count=t.warm_counts[p]
    if not isfinite(u) or not isfinite(score) or u<-1. or u>1.: return
    for i in range(count):
        j=base+i
        if fabs(t.warm[j]-u)<=tolerance:
            if score<t.warm_scores[j]: t.warm[j]=u;t.warm_scores[j]=score
            return
    if count<8:
        j=base+count;t.warm_counts[p]=count+1
    else:
        for i in range(1,8):
            if t.warm_scores[base+i]>t.warm_scores[base+worst]: worst=i
        j=base+worst
        if score>=t.warm_scores[j]: return
    t.warm[j]=u;t.warm_scores[j]=score

cdef int series_candidates_c(SeriesSearchTable* t,const double* theta,
                             double* out,int capacity) noexcept nogil:
    cdef int p,i,j,n,k,count=0,S=t.order+1,K=t.samples+1,cap=capacity
    cdef int brackets,ready,nwarm,w
    cdef double total,lo,hi,u,nextu,best,bestu,bestx,score,q3,q4
    cdef double panelbest,panelx,rootbest,margin
    cdef double center,scale,utol,x,seed_distance,distance
    cdef double seeds[8]
    cdef const double* row
    cdef double* c=t.coefficients
    if cap>t.capacity: cap=t.capacity
    if cap<1: return 0
    t.calls+=1
    if t.has_last and cap>=t.last_count:
        for j in range(t.d):
            if theta[j]!=t.last_theta[j]: break
        else:
            for j in range(t.last_count): out[j]=t.last_points[j]
            t.replays+=1
            return t.last_count
    for p in range(t.panels):
        center=t.geometry[2*p];scale=t.geometry[2*p+1]
        utol=t.tolerance/scale
        brackets=0;ready=0
        # Fixed exact-basis scan rows use only dot products with this theta.
        # No Horner work or q'''' values are needed to scan the panel.
        for i in range(K):
            for j in range(2):
                row=t.scan_rows+((p*2+j)*K+i)*t.d
                if dot(row,theta,t.d,t.scan_values+j*K+i):
                    t.failures+=1;return -1
            if i and t.scan_values[K+i-1]<=0. and t.scan_values[K+i]>=0. and (t.scan_values[K+i-1]<0. or t.scan_values[K+i]>0.):
                brackets+=1
        t.scan_nodes+=K
        best=t.scan_values[0];bestx=t.scan_points[p*K]
        for i in range(1,K):
            if t.scan_values[i]<best:
                best=t.scan_values[i];bestx=t.scan_points[p*K+i]
        panelbest=best;panelx=bestx;rootbest=NAN
        nwarm=t.warm_counts[p]
        if brackets==0 and nwarm==0:
            record(t,panelx,panelbest,out,cap,&count)
            continue
        # Only candidate panels require any Taylor coefficient contraction.
        if combine(t,theta,p,0,&ready): t.failures+=1;return -1
        total=c[0]
        for n in range(1,S): total-=fabs(c[n])
        if total>t.threshold:
            record(t,panelx,panelbest,out,cap,&count)
            continue
        if combine(t,theta,p,1,&ready): t.failures+=1;return -1
        for w in range(nwarm): seeds[w]=t.warm[8*p+w]
        t.warm_counts[p]=0
        for w in range(nwarm):
            u=seeds[w]
            if u<-1. or u>1.: continue
            best=polynomial(c,t.order,u);bestu=u
            if not isfinite(best): t.failures+=1;return -1
            for k in range(8):
                q3=polynomial(c+S,t.order,u)
                if combine(t,theta,p,2,&ready): t.failures+=1;return -1
                q4=polynomial(c+2*S,t.order,u)
                t.warm_probes+=1
                if not isfinite(q3) or not isfinite(q4): t.failures+=1;return -1
                if q3==0. or q4<=0.: break
                nextu=u-q3/(q4*scale)
                if not isfinite(nextu) or nextu<-1. or nextu>1.: break
                distance=fabs(nextu-u);u=nextu
                score=polynomial(c,t.order,u)
                if not isfinite(score): t.failures+=1;return -1
                if score<best: best=score;bestu=u
                if distance<=utol: break
            if best<panelbest: panelbest=best;panelx=center+scale*bestu
            q3=polynomial(c+S,t.order,u)
            q4=polynomial(c+2*S,t.order,u)
            score=polynomial(c,t.order,u)
            if not isfinite(q3) or not isfinite(q4) or not isfinite(score):
                t.failures+=1;return -1
            if q4>0. or (q4==0. and q3==0.):
                remember(t,p,u,score,utol)
                record(t,center+scale*u,score,out,cap,&count)
                if score<=t.threshold and (not isfinite(rootbest) or score<rootbest): rootbest=score
        for i in range(1,K):
            if not (t.scan_values[K+i-1]<=0. and t.scan_values[K+i]>=0. and (t.scan_values[K+i-1]<0. or t.scan_values[K+i]>0.)):
                continue
            lo=(t.scan_points[p*K+i-1]-center)/scale
            hi=(t.scan_points[p*K+i]-center)/scale
            u=.5*(lo+hi);seed_distance=hi-lo
            for w in range(t.warm_counts[p]):
                x=t.warm[8*p+w]
                distance=fabs(x-.5*(lo+hi))
                if lo<x<hi and distance<seed_distance:
                    seed_distance=distance;u=x
            for k in range(t.max_steps):
                q3=polynomial(c+S,t.order,u)
                t.refinements+=1
                if not isfinite(q3): t.failures+=1;return -1
                if q3==0. or hi-lo<=utol: break
                if q3<0.: lo=u
                else: hi=u
                if combine(t,theta,p,2,&ready): t.failures+=1;return -1
                q4=polynomial(c+2*S,t.order,u)
                if not isfinite(q4): t.failures+=1;return -1
                nextu=NAN
                if q4>0.: nextu=u-q3/(q4*scale)
                if not isfinite(nextu) or nextu<=lo+.05*(hi-lo) or nextu>=hi-.05*(hi-lo):
                    nextu=.5*(lo+hi)
                u=nextu
            score=polynomial(c,t.order,u)
            if not isfinite(score): t.failures+=1;return -1
            record(t,center+scale*u,score,out,cap,&count)
            if score<=t.threshold and (not isfinite(rootbest) or score<rootbest): rootbest=score
            if combine(t,theta,p,2,&ready): t.failures+=1;return -1
            q4=polynomial(c+2*S,t.order,u)
            if not isfinite(q4): t.failures+=1;return -1
            if q4>=0.: remember(t,p,u,score,utol)
        # Favor stationary cuts; redundant sampled rows can slow exchange fits.
        # Keep a lower sampled/probed point if refinement did not account for it.
        if not isfinite(rootbest):
            record(t,panelx,panelbest,out,cap,&count)
        else:
            margin=fabs(rootbest)+fabs(panelbest)+fabs(t.threshold)
            if panelbest<rootbest-1e-10*margin:
                record(t,panelx,panelbest,out,cap,&count)
    for i in range(1,count):
        x=out[i];j=i-1
        while j>=0 and out[j]>x:
            out[j+1]=out[j];j-=1
        out[j+1]=x
    # Repeated violating theta must not manufacture rows from refinement history.
    if count:
        for j in range(t.d): t.last_theta[j]=theta[j]
        for j in range(count): t.last_points[j]=out[j]
        t.last_count=count;t.has_last=1
    return count

cdef class SeriesSearchData:
    """Immutable geometry/tensor snapshot, shareable across search workspaces."""
    def __cinit__(self,geometry,tensor,int samples=8,scan_points=None,scan_rows=None):
        g=np.array(geometry,dtype=np.float64,order='C',copy=True)
        a=np.array(tensor,dtype=np.float64,order='C',copy=True)
        if g.ndim!=2 or g.shape[1]!=2 or a.ndim!=4 or a.shape[0]!=g.shape[0] or a.shape[1]!=3 or a.shape[2]<1 or a.shape[3]<1 or g.shape[0]<1:
            raise ValueError('invalid panel/tensor dimensions')
        if not np.isfinite(g).all() or not np.isfinite(a).all() or np.any(g[:,1]<=0):
            raise ValueError('series search requires finite ordinary coefficients')
        if samples<2: raise ValueError('at least two scan subintervals are required')
        # Structural zero high powers cost nothing at evaluation time. In
        # particular polynomial curvature does not need the requested tail order.
        active=np.flatnonzero(np.any(a!=0,axis=(0,1,3)))
        effective=int(active[active.size-1]) if active.size else 0
        a=np.ascontiguousarray(a[:,:,:effective+1,:])
        provided=scan_rows is not None
        u=np.linspace(-1.,1.,samples+1)
        x=g[:,0,None]+g[:,1,None]*u
        if scan_points is not None:
            x=np.array(scan_points,dtype=np.float64,order='C',copy=True)
        if x.shape!=(g.shape[0],samples+1) or not np.isfinite(x).all():
            raise ValueError('invalid scan points')
        if np.any(np.diff(x,axis=1)<0): raise ValueError('scan points must be ordered')
        if provided:
            rows=np.array(scan_rows,dtype=np.float64,order='C',copy=True)
        else:
            # Synthetic tensor callers can cache their polynomial scan rows.
            # Model-backed construction supplies actual basis rows instead.
            rows=np.zeros((g.shape[0],2,samples+1,a.shape[3]))
            u=(x-g[:,0,None])/g[:,1,None]
            for n in range(effective,-1,-1):
                rows=rows*u[:,None,:,None]+a[:,:2,n,None,:]
        if rows.shape!=(g.shape[0],2,samples+1,a.shape[3]) or not np.isfinite(rows).all():
            raise ValueError('invalid finite scan rows')
        # A bytes owner prevents writeability from being re-enabled by callers.
        g=np.frombuffer(g.tobytes(),dtype=np.float64).reshape(g.shape)
        a=np.frombuffer(a.tobytes(),dtype=np.float64).reshape(a.shape)
        x=np.frombuffer(x.tobytes(),dtype=np.float64).reshape(x.shape)
        rows=np.frombuffer(rows.tobytes(),dtype=np.float64).reshape(rows.shape)
        self._keep=(g,a,x,rows,provided)

    @property
    def geometry(self):
        return self._keep[0].view()

    @property
    def tensor(self):
        return self._keep[1].view()

    @property
    def scan_points(self):
        return self._keep[2].view()

    @property
    def scan_rows(self):
        return self._keep[3].view()

    @property
    def exact_scan(self):
        return self._keep[4]

    @property
    def samples(self):
        return self._keep[2].shape[1]-1

    @property
    def nbytes(self):
        return sum(a.nbytes for a in self._keep[:4])


cdef class SeriesSearchKernel:
    """Per-Newton-objective mutable workspace; do not share concurrent calls."""
    def __cinit__(self,geometry,tensor=None,int samples=8,int max_steps=16,
                  double tolerance=1e-5,double threshold=0.,int capacity=68):
        cdef SeriesSearchData data
        if isinstance(geometry,SeriesSearchData):
            if tensor is not None:
                raise ValueError('a prepared tensor does not take a second tensor')
            data=geometry
        else:
            data=SeriesSearchData(geometry,tensor,samples=samples)
        if samples<2 or max_steps<1 or capacity<1 or not isfinite(tolerance) or tolerance<=0 or not isfinite(threshold):
            raise ValueError('invalid series search options')
        if samples!=data.samples: raise ValueError('scan count does not match the cached grid')
        g,a,x,rows,provided=data._keep
        coefficients=np.empty(a.shape[1:3]);warm=np.full((g.shape[0],8),np.nan)
        warm_scores=np.empty_like(warm);warm_counts=np.zeros(g.shape[0],dtype=np.int32)
        scores=np.empty(capacity);scan_values=np.empty((2,samples+1))
        last_theta=np.empty(a.shape[3]);last_points=np.empty(capacity)
        self._keep=(data,coefficients,warm,warm_scores,warm_counts,scores,last_theta,last_points,scan_values)
        self.t.last_theta=<double*>cnp.PyArray_DATA(last_theta)
        self.t.last_points=<double*>cnp.PyArray_DATA(last_points)
        self.t.last_count=0;self.t.has_last=0;self.t.replays=0
        self.t.panels=g.shape[0];self.t.order=a.shape[2]-1;self.t.d=a.shape[3]
        self.t.geometry=<const double*>cnp.PyArray_DATA(g)
        self.t.tensor=<const double*>cnp.PyArray_DATA(a)
        self.t.scan_points=<const double*>cnp.PyArray_DATA(x)
        self.t.scan_rows=<const double*>cnp.PyArray_DATA(rows)
        self.t.scan_values=<double*>cnp.PyArray_DATA(scan_values)
        self.t.coefficients=<double*>cnp.PyArray_DATA(coefficients)
        self.t.warm=<double*>cnp.PyArray_DATA(warm)
        self.t.warm_scores=<double*>cnp.PyArray_DATA(warm_scores)
        self.t.warm_counts=<int*>cnp.PyArray_DATA(warm_counts)
        self.t.scores=<double*>cnp.PyArray_DATA(scores)
        self.t.samples=samples;self.t.max_steps=max_steps;self.t.capacity=capacity
        self.t.tolerance=tolerance;self.t.threshold=threshold
        self.t.calls=0;self.t.failures=0;self.t.refinements=0
        self.t.scan_nodes=0;self.t.q2_builds=0;self.t.q3_builds=0;self.t.q4_builds=0;self.t.warm_probes=0

    @property
    def data(self):
        return self._keep[0]

    def candidates(self,const double[::1] theta):
        cdef int n
        cdef object points=np.empty(self.t.capacity)
        cdef double[::1] v=points
        if theta.shape[0]!=self.t.d or not np.isfinite(theta).all():
            raise ValueError('invalid theta')
        with nogil: n=series_candidates_c(&self.t,&theta[0],&v[0],self.t.capacity)
        if n<0: raise FloatingPointError('series candidate arithmetic failed')
        return points[:n].copy()

    def statistics(self):
        return dict(panels=self.t.panels,calls=self.t.calls,failures=self.t.failures,
                    refinements=self.t.refinements,order=self.t.order,replays=self.t.replays,
                    scan_nodes=self.t.scan_nodes,q2_builds=self.t.q2_builds,
                    q3_builds=self.t.q3_builds,q4_builds=self.t.q4_builds,
                    warm_probes=self.t.warm_probes,tracked_roots=int(np.sum(self._keep[4])))
