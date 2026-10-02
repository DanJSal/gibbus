# cython: language_level=3
"""Compiled stable reductions for mixture log-sum-exp and derivative jets."""

import numpy as np
cimport numpy as cnp

from libc.math cimport exp, log, isfinite, INFINITY, NAN
from libc.stdlib cimport malloc, free

cnp.import_array()


def neg_logsumexp_batch(double[:, ::1] ell0):
    """Return ``-log(sum_j exp(ell0[j, r]))`` for every trailing point."""
    cdef Py_ssize_t K = ell0.shape[0]
    cdef Py_ssize_t R = ell0.shape[1]
    cdef cnp.ndarray[cnp.float64_t, ndim=1] out = np.empty(R, dtype=np.float64)
    cdef Py_ssize_t j, r
    cdef double c, v, total
    if K < 1:
        raise ValueError("ell0 must contain at least one component")
    for r in range(R):
        c = -INFINITY
        for j in range(K):
            v = ell0[j, r]
            if v > c:
                c = v
        if c == -INFINITY:
            out[r] = INFINITY
            continue
        if not isfinite(c):
            out[r] = -c
            continue
        total = 0.0
        for j in range(K):
            total += exp(ell0[j, r] - c)
        out[r] = -(c + log(total))
    return out


def neg_log_mix_derivs_batch(double[:, :, ::1] ell_jets, int max_order):
    """Compute mixture negative-log derivative jets with C-level recurrences."""
    cdef Py_ssize_t K = ell_jets.shape[0]
    cdef Py_ssize_t M = ell_jets.shape[1]
    cdef Py_ssize_t R = ell_jets.shape[2]
    cdef int N = int(max_order)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] out
    cdef double* fact = NULL
    cdef double* a = NULL
    cdef double* S = NULL
    cdef double* h = NULL
    cdef Py_ssize_t j, r
    cdef int n, m
    cdef double c, v, acc
    if N < 0:
        raise ValueError("max_order must be >= 0")
    if K < 1:
        raise ValueError("ell_jets must contain at least one component")
    if M < N + 1:
        raise ValueError("ell_jets does not contain the requested derivative order")
    out = np.empty((N + 1, R), dtype=np.float64)
    fact = <double*>malloc((N + 1) * sizeof(double))
    a = <double*>malloc(K * (N + 1) * sizeof(double))
    S = <double*>malloc((N + 1) * sizeof(double))
    h = <double*>malloc((N + 1) * sizeof(double))
    if fact == NULL or a == NULL or S == NULL or h == NULL:
        if fact != NULL:
            free(fact)
        if a != NULL:
            free(a)
        if S != NULL:
            free(S)
        if h != NULL:
            free(h)
        raise MemoryError()
    try:
        fact[0] = 1.0
        for n in range(1, N + 1):
            fact[n] = fact[n - 1] * n
        for r in range(R):
            c = -INFINITY
            for j in range(K):
                v = ell_jets[j, 0, r]
                if v > c:
                    c = v
            if c == -INFINITY:
                out[0, r] = INFINITY
                for n in range(1, N + 1):
                    out[n, r] = NAN
                continue
            if not isfinite(c):
                out[0, r] = -c
                for n in range(1, N + 1):
                    out[n, r] = NAN
                continue

            for n in range(N + 1):
                S[n] = 0.0
                h[n] = 0.0
            for j in range(K):
                a[j * (N + 1)] = exp(ell_jets[j, 0, r] - c)
                S[0] += a[j * (N + 1)]
                for n in range(1, N + 1):
                    acc = 0.0
                    for m in range(1, n + 1):
                        acc += (
                            m * (ell_jets[j, m, r] / fact[m])
                            * a[j * (N + 1) + (n - m)]
                        )
                    a[j * (N + 1) + n] = acc / n
                    S[n] += a[j * (N + 1) + n]

            h[0] = log(S[0]) + c
            out[0, r] = -h[0]
            for n in range(1, N + 1):
                acc = 0.0
                for m in range(1, n):
                    acc += m * h[m] * S[n - m]
                h[n] = (S[n] - acc / n) / S[0]
                out[n, r] = -h[n] * fact[n]
    finally:
        free(fact)
        free(a)
        free(S)
        free(h)
    return out
