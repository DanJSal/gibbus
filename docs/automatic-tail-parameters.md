# Automatic tail parameters

For an infinite support end, users can select the leading-potential tail family
without supplying its parameter. During `Distribution.fit`, the estimator
chooses an integer polynomial power, real soft-power exponent, or exponential
rate from observed points. Explicitly supplied parameters remain fixed.

```python
from gibbus import Distribution, BasisSpec, Polynomial, SoftPower, Exponential

# Empirically selected powers and rates
model = Distribution().fit(samples, basis=BasisSpec(
    bulk_degree=2,
    lower_tail=Exponential(),
    upper_tail=Polynomial(),
))
print(model.tail_parameter_diagnostics)

# Fully specified (no empirical tail estimation)
fixed = Distribution().fit(samples, basis=BasisSpec(
    bulk_degree=2,
    lower_tail=Exponential(0.75),
    upper_tail=SoftPower(2.5),
))
```

`upper_tail=None` and `lower_tail=None` retain the bulk-derived tail policy.
Each end is estimated independently when its parameter is omitted. Integer
polynomial requests screen powers 1 through 6. Soft-power requests screen the
same integers and optimize the continuous exponent within one of the winner,
keeping `p > 1`; an estimate that reaches `p = 1` means the tail is heavier
than any soft power, and the fit stops with an error that says so. Exponential
requests optimize the positive rate in standardized coordinates from 0.12
upward (the bracket widens until the optimum is interior), then return the
physical inverse-unit rate. Automatic estimation is empirical: it does **not**
run a full outer profile of the fit.

These statistical estimators use the outer part of the oriented sample and
are intended to find usable parameters, not determine the unknowable ultimate
asymptotic tail class. The fit's shape verification remains mandatory, and
there is no second proposal: a fit that fails with the estimated parameter
fails. Explicit requests are never modified.

The fitting result provides a session diagnostic under
`model.tail_parameter_diagnostics` with each end's chosen family, parameter
and effective tail sample size.
This diagnostic is retained by copying and pickling but is not part of the
portable numeric model format.

**Data coverage:** Automatic estimation currently requires at least 20 finite
point observations. Weighted point fitting uses a deterministic approximation
to the weighted empirical quantile distribution. In a mixture fit, parameter
screening is performed on pooled observations and is shared by the components;
it is not a per-component tail estimate. Interval-censored observations must
specify explicit parameters until a censoring-aware empirical procedure is
implemented. Fixed parameters and omitted tail families continue to support
the existing observation types.

An optional Cython quadrature/gradient kernel is built along with Gibbus's
other extensions. Its absence falls back to equivalent NumPy/SciPy calculations.
The kernel is purely an accelerator, not a change in the statistical objective.

## Numerical screening

Tail data are standardized before variance calculations so changing physical
units does not cause avoidable squared-scale overflow or underflow. A rate
that cannot be represented in physical inverse units fails with a rescaling
message. Weighted quantile interpolation is deterministic.

The conditional likelihood accepts only converged finite optimizer results;
a failed coefficient start tries the second configured start. Unsuccessful
scalar parameter searches fail explicitly. Shared fitting deadlines are checked
between candidates, tail sides and optimizer callbacks. This remains empirical
screening with finite tail data, rather than an asymptotic-family guarantee.
