"""Smoke checks executed against each installed release wheel."""

from importlib import import_module
from importlib.metadata import version
from pathlib import Path

import gibbus
import numpy
import scipy


EXTENSIONS = (
    "gibbus._fit._conic_kernels",
    "gibbus._fit._curvature_certificate",
    "gibbus._fit._mixture_kernels",
    "gibbus._fit._shared_mixture_kernels",
    "gibbus._model._moment_kernels",
    "gibbus._model._quad_integrals",
    "gibbus._model._state_kernels",
    "gibbus._observations._finite_reductions",
    "gibbus._observations._interval_integrals",
    "gibbus._postfit._mix_kernels",
    "gibbus._spectral._builders",
    "gibbus._spectral._cdf_eval",
    "gibbus._spectral._certify",
    "gibbus._spectral._panel_kernels",
    "gibbus._spectral._ppf_eval",
    "gibbus._spectral._tail_integrals",
)


def main():
    package_root = Path(gibbus.__file__).resolve().parent
    print(f"gibbus {version('gibbus')} from {package_root}")
    print(f"numpy {numpy.__version__}")
    print(f"scipy {scipy.__version__}")

    for module_name in EXTENSIONS:
        module = import_module(module_name)
        origin = Path(module.__file__).resolve()
        if package_root not in origin.parents:
            raise RuntimeError(
                f"{module_name} imported from outside the installed package: {origin}"
            )
        if origin.suffix.lower() not in {".so", ".pyd"}:
            raise RuntimeError(
                f"{module_name} is not a compiled extension module: {origin}"
            )
        print(f"OK {module_name}: {origin.name}")


if __name__ == "__main__":
    main()
