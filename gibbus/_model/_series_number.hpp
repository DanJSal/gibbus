#ifndef GIBBUS_SERIES_NUMBER_HPP
#define GIBBUS_SERIES_NUMBER_HPP
#include <cmath>
#include <cstdint>
#include <limits>
#include <cfenv>
#include <new>
#include <cstddef>
// A binary64 significand with an extended exponent, used only on range fallback.
// This extends range, not precision. No allocation, exceptions, or global state.
namespace gibbus_series {
struct Wide {
    double m;
    int64_t e;
    Wide() noexcept : m(0), e(0) {}
    Wide(double x) noexcept : m(x), e(0) { normalize(); }
    Wide(double x, int64_t n) noexcept : m(x), e(n) { normalize(); }
    void normalize() noexcept {
        if (m == 0) { e=0; return; }
        if (!std::isfinite(m)) { e=0; return; }
        int k; m=std::frexp(m,&k); e+=k;
        if (e > (INT64_C(1)<<50) || e < -(INT64_C(1)<<50)) {
            m=std::numeric_limits<double>::quiet_NaN(); e=0;
        }
    }
    Wide operator-() const noexcept { return Wide(-m,e); }
    Wide operator+(Wide b) const noexcept {
        if (!std::isfinite(m) || !std::isfinite(b.m)) return Wide(NAN);
        if (m==0) return b;
        if (b.m==0) return *this;
        if (e < b.e) return b+*this;
        int64_t d=e-b.e;
        if (d>1074) return *this;
        return Wide(m+std::ldexp(b.m,-static_cast<int>(d)),e);
    }
    Wide operator-(Wide b) const noexcept { return *this+(-b); }
    Wide operator*(Wide b) const noexcept { return Wide(m*b.m,e+b.e); }
    Wide operator/(Wide b) const noexcept { return Wide(m/b.m,e-b.e); }
    bool operator==(Wide b) const noexcept { return m==b.m && e==b.e; }
    bool operator!=(Wide b) const noexcept { return !(*this==b); }
};
inline Wide* allocate_wide(size_t n) noexcept { return new(std::nothrow) Wide[n]; }
inline void free_wide(Wide* p) noexcept { delete[] p; }
inline double value(Wide x) noexcept {
    if (!std::isfinite(x.m) || x.m==0) return x.m;
    if (x.e>1024) return std::copysign(INFINITY,x.m);
    if (x.e < -1074) return std::copysign(0.0,x.m);
    return std::ldexp(x.m,static_cast<int>(x.e));
}
inline Wide wide_exp(Wide x) noexcept {
    double a=value(x);
    if (!std::isfinite(a) || std::abs(a)>281474976710656.0) return Wide(NAN);
    const double hi=0.69314718036912381649, lo=1.9082149292705877000e-10;
    int64_t e=static_cast<int64_t>(std::floor(a*1.4426950408889634074));
    // fma preserves the reduction residual even for large (representable) a.
    double r=std::fma(-static_cast<double>(e),hi,a);
    r=std::fma(-static_cast<double>(e),lo,r);
    return Wide(std::exp(r),e);
}
inline Wide wide_sqrt(Wide x) noexcept {
    if (x.e%2) { x.m*=2; --x.e; }
    return Wide(std::sqrt(x.m),x.e/2);
}
inline Wide wide_pow(Wide x,double p) noexcept {
    if (x.m<=0 || !std::isfinite(x.m)) return Wide(NAN);
    return wide_exp(Wide(p*(std::log(x.m)+static_cast<double>(x.e)*0.6931471805599453094)));
}
// fenv is thread-local. Restore the caller's flags/mode after the guarded pass.
struct RangeGuard {
    fenv_t saved;
    RangeGuard() noexcept { std::feholdexcept(&saved); }
    int failed() const noexcept { return std::fetestexcept(FE_OVERFLOW|FE_UNDERFLOW|FE_INVALID|FE_DIVBYZERO); }
    ~RangeGuard() noexcept { std::fesetenv(&saved); }
};
}
#endif
