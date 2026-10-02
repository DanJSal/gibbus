"""SciPy-compatible frozen-distribution adapter for :class:`Distribution`."""

import numpy as np


class FrozenDistribution:
    """Live frozen-distribution view over a fitted Distribution instance.

    Parameters
    ----------
    parent : Distribution
        Fitted parent. The adapter intentionally remains live: refitting or
        transforming the parent is reflected by subsequent calls.
    """

    def __init__(self, parent):
        """Bind the adapter to a fitted parent distribution.

        Parameters
        ----------
        parent : Distribution
            Fitted parent distribution.
        """
        self._parent = parent

    def pdf(self, x):
        """Evaluate the probability density.

        Parameters
        ----------
        x : float or array_like
            Evaluation coordinate(s).
        """
        return self._parent.pdf(x)

    def logpdf(self, x):
        """Evaluate the log probability density.

        Parameters
        ----------
        x : float or array_like
            Evaluation coordinate(s).
        """
        return self._parent.logpdf(x)

    def cdf(self, x):
        """Evaluate the cumulative distribution function.

        Parameters
        ----------
        x : float or array_like
            Evaluation coordinate(s).
        """
        return self._parent.cdf(x)

    def logcdf(self, x):
        """Evaluate the log cumulative distribution function.

        Parameters
        ----------
        x : float or array_like
            Evaluation coordinate(s).
        """
        return self._parent.logcdf(x)

    def sf(self, x):
        """Evaluate the survival function.

        Parameters
        ----------
        x : float or array_like
            Evaluation coordinate(s).
        """
        return self._parent.sf(x)

    def logsf(self, x):
        """Evaluate the log survival function.

        Parameters
        ----------
        x : float or array_like
            Evaluation coordinate(s).
        """
        return self._parent.logsf(x)

    def ppf(self, p):
        """Evaluate the lower-tail quantile function.

        Parameters
        ----------
        p : float or array_like
            CDF probabilities.
        """
        return self._parent.ppf(p)

    def isf(self, p):
        """Evaluate the inverse survival function.

        Parameters
        ----------
        p : float or array_like
            Survival probabilities.
        """
        return self._parent.isf(p)

    def rvs(self, size=None, random_state=None):
        """Draw random variates using SciPy-style argument names.

        Parameters
        ----------
        size : int, tuple of int, or None, optional
            Output shape.  A tuple is drawn flat and reshaped, matching
            SciPy's frozen distributions.
        random_state : optional
            Random-number source accepted by :meth:`Distribution.sample`.
        """
        if size is None or isinstance(size, (int, np.integer)):
            return self._parent.sample(size=size, rng=random_state)
        shape = tuple(int(dim) for dim in size)
        count = 1
        for dim in shape:
            count *= dim
        draws = np.asarray(
            self._parent.sample(size=count, rng=random_state), dtype=float
        )
        return draws.reshape(shape)

    def mean(self):
        """Return the distribution mean."""
        return float(self._parent.mean)

    def var(self):
        """Return the distribution variance."""
        return float(self._parent.var)

    def std(self):
        """Return the distribution standard deviation."""
        return float(self._parent.std)

    def median(self):
        """Return the distribution median."""
        return float(self._parent.median)

    def entropy(self):
        """Return the differential entropy."""
        return float(self._parent.entropy())

    def moment(self, order):
        """Return the raw moment of the requested non-negative integer order.

        Parameters
        ----------
        order : int
            Moment order.
        """
        return float(self._parent.moment(order))

    def cumulant(self, order):
        """Return the cumulant of the requested positive integer order.

        Parameters
        ----------
        order : int
            Positive cumulant order.

        Returns
        -------
        float
            Requested cumulant.

        Raises
        ------
        ValueError
            If *order* is not a positive integer.
        """
        return float(self._parent.cumulant(order))

    def interval(self, confidence):
        """Return an equal-tailed confidence interval.

        Parameters
        ----------
        confidence : float or array_like
            Probability mass in ``(0, 1]``.  An array of levels returns a
            pair of arrays, as SciPy's frozen distributions do.
        """
        levels = np.asarray(confidence, dtype=float)
        if levels.ndim == 0:
            return self._parent.interval(float(levels))
        bounds = [self._parent.interval(float(level)) for level in levels.ravel()]
        lower = np.array([pair[0] for pair in bounds], dtype=float)
        upper = np.array([pair[1] for pair in bounds], dtype=float)
        return lower.reshape(levels.shape), upper.reshape(levels.shape)

    def support(self):
        """Return the lower and upper support endpoints."""
        lo, hi = map(float, self._parent.support)
        return lo, hi

    def stats(self, moments="mv"):
        """Return selected conventional distribution statistics.

        Parameters
        ----------
        moments : str, optional
            Combination of ``m``, ``v``, ``s``, and ``k``.
        """
        values = []
        for token in str(moments):
            if token == "m":
                values.append(float(self._parent.mean))
            elif token == "v":
                values.append(float(self._parent.var))
            elif token == "s":
                values.append(float(self._parent.skew))
            elif token == "k":
                values.append(float(self._parent.kurt - 3.0))
            else:
                raise ValueError("moments may contain only 'm', 'v', 's', and 'k'")
        if len(values) == 1:
            return values[0]
        return tuple(values)

    def expect(self, func=None, args=(), lb=None, ub=None, conditional=False, **_kwargs):
        """Compute an expectation with SciPy-compatible options.

        Parameters
        ----------
        func : callable or None, optional
            Function of the random variable; ``None`` uses the identity.
        args : tuple, optional
            Extra positional arguments passed to *func*.
        lb, ub : float or None, optional
            Optional integration bounds.
        conditional : bool, optional
            Divide by the probability mass between *lb* and *ub*.
        **_kwargs : dict
            Ignored compatibility keywords.
        """
        if args:
            if func is None:
                raise ValueError("args require a callable func")
            base_func = func

            def wrapped_func(x):
                return base_func(x, *args)

            func = wrapped_func
        if func is None:
            def identity(x):
                return x

            func = identity
        if lb is None and ub is None:
            return self._parent.expect(func)
        lo, hi = map(float, self._parent.support)
        a = lo if lb is None else max(lo, float(lb))
        b = hi if ub is None else min(hi, float(ub))
        if not a < b:
            return np.nan if conditional else 0.0
        num = self._parent._expect_between(func, a, b)
        if not conditional:
            return num
        mass = float(np.exp(self._parent._log_mass(a, b)))
        return num / mass
