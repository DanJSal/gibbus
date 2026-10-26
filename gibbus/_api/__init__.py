"""The public API layer: :class:`~gibbus._api.distribution.Distribution` and its internals.

Split by responsibility rather than by class:

* :mod:`.views` — the ``base`` and ``exp`` coordinate-space views.
* :mod:`.component` — a single fitted log-concave component over its packed
  basis state (``_postfit.basis_state``).
* :mod:`.fitting` — public-fit request preparation and state-free controller
  orchestration.
* :mod:`.mixture_stats` — mixture-level moments, potentials, and the
  rebuilt-on-demand mixture spectral cache.
* :mod:`.diagnostics` — fit, selection, and spectral diagnostic records.
* :mod:`.validation` — canonical probability and region inputs at query
  boundaries; :mod:`.errors` — typed control-flow errors.
* :mod:`.distribution` — the public class that composes all of the above.
"""

from .distribution import Distribution

__all__ = ["Distribution"]
