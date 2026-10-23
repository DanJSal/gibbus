# cython: language_level=3

cimport numpy as cnp

# Function kinds and sides of the compiled column table.
cdef enum:
    FN_MONOMIAL = 0
    FN_LOG = 1
    FN_SOFT = 2
    FN_EXP = 3
    SIDE_LOWER = 0
    SIDE_UPPER = 1

# Tail verdict statuses.
cdef enum:
    TAIL_NONNEGATIVE = 0
    TAIL_VIOLATED = 1
    TAIL_INCONCLUSIVE = 2

# Growth (properness) statuses.
cdef enum:
    GROWTH_PROPER = 0
    GROWTH_IMPROPER = 1
    GROWTH_INCONCLUSIVE = 2


cdef struct Iv:
    double lo
    double hi


cdef struct BasisTable:
    int nf
    int d
    const int* kind
    const int* side
    const int* degree
    const double* power
    const double* delta
    const double* rate
    const double* location
    const int* col_start
    const int* term_fn
    const double* term_g0
    const double* term_g1
    const double* term_g2
    const double* term_glow
    const double* term_gup
    const double* cmap
    const double* cmap_low
    int orth_degree
    int polynomial_point
    const int* orth_column_degree
    const int* orth_column_fn
    const double* orth_norm
    const double* orth_reference
    double orth_scale
    double orth_alpha
    double orth_origin
    const double* orth_recurrence
    const double* orth_map
    const double* lim_low
    const double* lim_up
    const double* lim_low_abs
    const double* lim_up_abs
    const int* nonneg
    const int* fn_used
    double c
    double h
    double lower
    double upper
    int has_l
    int has_u
    double xl
    double al
    double xu
    double au
    double mid
    int log_col_lower
    int log_col_upper


cdef class BasisKernel:
    cdef BasisTable t
    cdef object _keep


cdef struct TailTable:
    int na
    int ne
    int d
    double sign
    double zeta_min
    double c
    double h
    const int* akind
    const double* apower
    const double* arate
    const double* aoffset
    const double* arow
    const double* acut
    const int* ecol
    const double* ecoef
    const double* epower
    const double* erate


cdef class TailKernel:
    cdef TailTable t
    cdef object _keep


cdef struct TailResult:
    int status
    double cutoff
    double zeta
    double lead_value
    double lead_power
    double lead_rate
    double margin
    bint has_lead
    bint has_margin
    bint has_cutoff
    int cut_atom


# Per-theta data of the potential enclosure (slots of ``full = M theta``).
cdef struct PotentialPrep:
    const double* theta
    int all_zero
    Iv* slot
    int* slot_zero
    Iv* k_low
    Iv* k_up
    int* skip
    int* gated


cdef Py_ssize_t prep_size(const BasisTable* t) noexcept nogil
cdef void prepare_potential(
    const BasisTable* t, const double* theta, double* work, PotentialPrep* p
) noexcept nogil
cdef double scaled_z(const BasisTable* t, double x) noexcept nogil
cdef void eval_point(
    const BasisTable* t, double x, int order, double* fwork, double* out
) noexcept nogil
cdef void potential_point(
    const BasisTable* t,
    const double* theta,
    double x,
    int order,
    double* fwork,
    double* colwork,
    double* q,
) noexcept nogil
cdef void potential_enclosure_cell(
    const BasisTable* t,
    const PotentialPrep* p,
    double lo,
    double hi,
    int top,
    Iv* fwork,
    Iv* out,
) noexcept nogil
cdef void basis_enclosure_cell(
    const BasisTable* t,
    double lo,
    double hi,
    int top,
    const int* want,
    Iv* fwork,
    Iv* out,
) noexcept nogil
cdef double endpoint_exclusion(
    const BasisTable* t,
    const double* theta,
    int upper,
    double threshold,
    double* rest,
    double* prepwork,
    Iv* ivwork,
) noexcept nogil
cdef Py_ssize_t tail_work_size(const TailTable* tt) noexcept nogil
cdef TailResult tail_verdict(
    const TailTable* tt,
    const double* theta,
    double zeta_cap,
    double dominance,
    bint growth,
    double* work,
) noexcept nogil
cdef int growth_status(const TailResult* res) noexcept nogil
cdef double tail_bound(const TailResult* res, double zeta) noexcept nogil
cdef double expit(double x) noexcept nogil
cdef double logaddexp(double x, double y) noexcept nogil
cdef double npmax(double a, double b) noexcept nogil
cdef double npmin(double a, double b) noexcept nogil
