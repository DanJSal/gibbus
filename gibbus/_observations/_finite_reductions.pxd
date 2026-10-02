cdef api int finite_natural_objective_c(
    Py_ssize_t R, Py_ssize_t P, Py_ssize_t G, Py_ssize_t W, Py_ssize_t nq,
    const double* intervals, const double* row_weights,
    const double* point_lower_distance, const double* point_upper_distance,
    const double* q_poly, double a_lower, double a_upper,
    double q_shift, double shifted_log_Z, double mode, double log_coordinate_scale,
    const int* partial_kinds, const int* partial_lengths,
    const double* partial_coeffs, double support_lower, double support_upper,
    const double* gl_nodes, const double* gl_log_weights, double width_eps_mult,
    double* log_probability, double* obs_h, double* obs_cov,
    double* sum_h, double* sum_second, double* hbuf,
    double* nll_out,
) noexcept nogil

cdef api int finite_natural_real_line_objective_c(
    Py_ssize_t R, Py_ssize_t P, Py_ssize_t G, Py_ssize_t nq,
    const double* intervals, const double* row_weights,
    const double* q_poly, double q_shift, double shifted_log_Z,
    double mode, double log_coordinate_scale,
    const double* natural_scale,
    const double* gl_nodes, const double* gl_log_weights,
    double width_eps_mult,
    double* log_probability, double* obs_h, double* obs_cov,
    double* sum_h, double* sum_second, double* hbuf,
    double* nll_out,
) noexcept nogil
