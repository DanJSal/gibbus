"""Build script for the gibbus Cython extensions.

Compiler flags are selected per-platform: MSVC does not understand the
GCC/Clang ``-O3`` family. Spectral SIMD hot loops provide their own portable
no-alias/dependency hints for GCC, Clang, and MSVC; generic wheel builds leave
the actual ISA choice to the compiler target. ``-ffast-math`` is deliberately *not*
used anywhere because the kernels rely on IEEE NaN/inf semantics
(``_finite`` checks, deliberate ``NAN``/``INFINITY`` results) that fast-math is free to
optimize away.
"""

import os
import sys

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

# ---------------------------------------------------------------------
# Platform-appropriate compiler flags
# ---------------------------------------------------------------------


def _env_flag(name):
    """Return whether a build environment variable contains a truthy token."""
    value = os.environ.get(name)
    return value is not None and value.strip().lower() in {"1", "true", "yes", "on"}


if sys.platform == "win32":
    EXTRA_COMPILE_ARGS = ["/O2"]
    EXTRA_LINK_ARGS = []
else:
    EXTRA_COMPILE_ARGS = ["-O3"]
    EXTRA_LINK_ARGS = []

# Flags used only when the compiler accepts them (probed at build time).
# -fopenmp-simd honors ``omp simd`` loop hints (vectorized reductions in the
# solver kernels) without linking an OpenMP runtime; without it the hints are
# ignored and the loops stay correct, just scalar.
OPTIONAL_COMPILE_ARGS = [] if sys.platform == "win32" else ["-fopenmp-simd"]

# Aggressive loop unrolling is opt-in for the same portability/reproducibility
# reason as native-architecture tuning.
if _env_flag("GIBBUS_UNROLL_LOOPS") and sys.platform != "win32":
    EXTRA_COMPILE_ARGS.append("-funroll-loops")

# Opt-in native tuning: not on by default because it produces binaries
# that will not run on other machines (a problem for wheels).
if _env_flag("GIBBUS_NATIVE_ARCH") and sys.platform != "win32":
    EXTRA_COMPILE_ARGS.append("-march=native")

DEFINE_MACROS = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]

# Dotted module path -> source file, both relative to the package root.
MODULES = {
    "_fit._conic_kernels": "_fit/_conic_kernels.pyx",
    "_fit._curvature_certificate": "_fit/_curvature_certificate.pyx",
    "_fit._mixture_kernels": "_fit/_mixture_kernels.pyx",
    "_model._quad_integrals": "_model/_quad_integrals.pyx",
    "_model._moment_kernels": "_model/_moment_kernels.pyx",
    "_model._state_kernels": "_model/_state_kernels.pyx",
    "_observations._interval_integrals": "_observations/_interval_integrals.pyx",
    "_observations._finite_reductions": "_observations/_finite_reductions.pyx",
    "_postfit._mix_kernels": "_postfit/_mix_kernels.pyx",
    "_spectral._cdf_eval": "_spectral/_cdf_eval.pyx",
    "_spectral._builders": "_spectral/_builders.pyx",
    "_spectral._certify": "_spectral/_certify.pyx",
    "_spectral._ppf_eval": "_spectral/_ppf_eval.pyx",
    "_spectral._panel_kernels": "_spectral/_panel_kernels.pyx",
    "_spectral._tail_integrals": "_spectral/_tail_integrals.pyx",
}

# ---------------------------------------------------------------------
# Cythonize from .pyx.  The .c files are build artifacts: they are not
# committed and not shipped in the sdist, so there is nothing to fall
# back to.  Cython is a declared build requirement in pyproject.toml.
# ---------------------------------------------------------------------

missing = [
    src for src in MODULES.values()
    if not os.path.exists(os.path.join("gibbus", src))
]
if missing:
    raise RuntimeError(
        f"Cannot build gibbus: missing source for {missing}. "
        "The .pyx sources are required; obtain a complete checkout or "
        "source distribution."
    )

extensions = [
    Extension(
        name=f"gibbus.{mod}",
        sources=[os.path.join("gibbus", src)],
        include_dirs=[np.get_include(), os.path.dirname(os.path.join("gibbus", src))],
        define_macros=DEFINE_MACROS,
        extra_compile_args=EXTRA_COMPILE_ARGS,
        extra_link_args=EXTRA_LINK_ARGS,
    )
    for mod, src in MODULES.items()
]

DEBUG_BUILD = _env_flag("GIBBUS_DEBUG_BUILD")

# These safety directives must not be repeated in per-file ``# cython:``
# headers: a header directive overrides the value passed to cythonize(),
# which would silently turn GIBBUS_DEBUG_BUILD=1 into an unchecked build.

compiler_directives = {
    "language_level": "3",
    "boundscheck": DEBUG_BUILD,
    "wraparound": DEBUG_BUILD,
    # Always C division: the kernels rely on IEEE results such as 1.0/0.0 = inf
    # (for example ppf(1) at an infinite endpoint), which Python-style
    # division would turn into a ZeroDivisionError swallowed inside nogil code.
    "cdivision": True,
    "nonecheck": DEBUG_BUILD,
    "initializedcheck": DEBUG_BUILD,
}

extensions = cythonize(
    extensions,
    compiler_directives=compiler_directives,
    # cythonize() only compares timestamps, so a .c generated for the other
    # GIBBUS_DEBUG_BUILD mode would otherwise be reused silently.
    force=True,
)



def _accepts_flag(compiler, flag):
    """Return whether ``compiler`` compiles a trivial file with ``flag``."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        source = os.path.join(tmp, "probe.c")
        with open(source, "w", encoding="utf-8") as handle:
            handle.write("int main(void) { return 0; }\n")
        try:
            compiler.compile([source], output_dir=tmp, extra_postargs=[flag, "-Werror"])
        except Exception:  # noqa: BLE001 - any failure means "not supported"
            return False
    return True


class BuildExt(build_ext):
    """``build_ext`` that appends the optional flags the compiler accepts."""

    def build_extensions(self):
        """Probe ``OPTIONAL_COMPILE_ARGS`` once, then build every extension."""
        accepted = [flag for flag in OPTIONAL_COMPILE_ARGS if _accepts_flag(self.compiler, flag)]
        for extension in self.extensions:
            extension.extra_compile_args = [*extension.extra_compile_args, *accepted]
        super().build_extensions()


setup(ext_modules=extensions, cmdclass={"build_ext": BuildExt})
