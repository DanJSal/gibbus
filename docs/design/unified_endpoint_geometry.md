# Gibbus: Endpoint-Geometry Potential Family Design

**Status:** design proposal / planning note  
**Scope:** mathematical and architectural specification; no implementation or benchmark claims are implied by this document.

---

## 1. Summary

Gibbus should represent a component through a **potential family** rather than requiring every model to be a polynomial in one affine coordinate.

A potential family supplies:

1. a monotone coordinate map between the fitting coordinate and an internal coordinate;
2. a finite linear energy-feature space;
3. an exact linear map from fitted coefficients to a univariate polynomial whose nonnegativity is equivalent to log-concavity in the user's coordinate;
4. the sufficient statistics required for the likelihood, gradient and Fisher/Hessian;
5. runtime evaluation rules and endpoint asymptotics.

Two families are primary:

- **Polynomial-\(z\) family:** today's model, retained as a separate exact and fast path for polynomial potentials.
- **Compact endpoint-geometry family:** a new family on \(t\in(0,1)\) designed for independently specifiable finite boundaries and tail behavior.

The compact family uses a monotone map

\[
z=Z(t),\qquad 0<t<1,
\]

and a direct energy expansion

\[
Q(t)=\theta^\top T(t).
\]

The coordinate geometry at \(t=0\) and \(t=1\) determines the endpoint type. A small vocabulary of endpoint energy features then determines how the density behaves there.

The preferred endpoint layouts are deliberately **canonical**, rather than forcing every asymptotic class into one continuously parameterized representation:

- ordinary finite boundary: \(r=-1\);
- algebraic vanishing at a finite boundary: \(r=-1\) plus a logarithmic feature;
- essential/super-algebraic finite-boundary zero: \(r<0\) plus a first-order pole;
- logistic-type exponential tail: \(r=0\) plus a logarithmic feature;
- real power-exponential tail: \(0<r<1\) plus a first-order pole, with \(\lambda=1/r\);
- algebraic-approach exponential tail: \(r=1\) plus a first-order pole, optionally with a logarithmic correction controlling the power prefactor;
- doubly-exponential tail: \(r=0\) plus a first-order pole.

The leading left and right endpoint classes and exponents are independently specifiable. Subleading asymptotic terms can couple through the global coordinate map and must be treated as such in diagnostics.

For fixed structural parameters, the fitted energy remains linear in its coefficients. Therefore the likelihood is still a finite-dimensional exponential-family likelihood:

\[
\nabla L
=n\bigl(E_\theta[T]-\bar T_{\rm data}\bigr),
\qquad
\nabla^2L
=n\,\operatorname{Cov}_\theta(T).
\]

The high-dimensional inner fit remains convex.

Exact log-concavity is certified in the original fitting coordinate \(z\), hence in \(x\) under the public affine transform. The transformed curvature condition reduces, after exact structural simplification, to nonnegativity of one univariate polynomial on \([0,1]\). The existing Markov–Lukács / sum-of-squares cone machinery can therefore remain the global certificate.

Continuous structural quantities are handled by a small, low-dimensional outer profile around that convex inner fit. For compact layouts, structural pivot and scale are the default unless the **entire fitted feature span** is closed under the corresponding affine action. Data-derived location/scale used for numerical preconditioning must never silently define the statistical family. The full end-to-end procedure is not globally convex when structural parameters are profiled, and the public guarantees should say so explicitly.

---

## 2. Potential-family architecture

The implementation should not assume that a single representation subsumes every useful case.

A potential family should conceptually expose:

\[
\boxed{
\text{coordinate}
+\text{energy feature space}
+\text{curvature certificate map}
+\text{state-statistic layout}
}
\]

It must also declare how affine changes of the physical fitting coordinate act on its feature span. In particular, every family/layout should state whether translation and scale changes are absorbed exactly by its linear coefficient space or instead change the statistical family.

The common optimizer operates on the fitted linear coefficients once a family and its structural parameters are fixed.

### 2.1 Polynomial-\(z\) family

The existing family remains first-class:

\[
Q(z)=\sum_{k=0}^{K}\beta_k z^k
+\text{current finite-boundary terms},
\]

with today's exact polynomial-curvature positivity machinery.

This family remains the preferred fast path for distributions whose potentials are genuinely polynomial in \(z\), including current exact polynomial cases. It should not be represented indirectly through the compact family merely for architectural uniformity.

The polynomial feature span is affine-invariant: translating or rescaling \(z\) only changes the polynomial coefficients. Therefore the data-derived location/scale used by current Gibbus can remain pure numerical preconditioning for this family.

In particular, the compact first-order-pole family does **not** contain arbitrary polynomial potentials. Gaussian energy is representable compactly, but odd powers and higher even powers such as \(z^4\) generally require additional nontrivial features. The polynomial family therefore remains necessary.

### 2.2 Compact endpoint-geometry family

The new family uses

\[
t\in(0,1),\qquad z=Z(t),
\]

with direct linear energy features in \(t\).

Its purpose is not to replace the polynomial family. Its purpose is to represent endpoint and tail geometries that polynomial curvature handles poorly, while preserving:

- one global analytic potential;
- exact log-concavity certification in \(x\);
- a convex coefficient fit for fixed structure;
- sufficient-statistic moment gradients;
- covariance/Fisher Hessians;
- tractable CDF, PPF, moments and interval probabilities.

Many compact layouts are **not** affine-invariant as finite feature spaces. A logistic-type layout, for example, has a physical transition scale and pivot; changing either can change which exact densities lie in the span. Such quantities are statistical structural parameters, not merely numerical coordinate choices.

### 2.3 Family metadata for affine actions and identifiability

Affine invariance is a property of the **whole fitted feature span**, not of a leading asymptotic term or a convenient subspace. A layout is translation- or scale-invariant only if every fitted feature is carried into the same finite span under the corresponding affine action.

Each potential family/layout should therefore expose metadata describing:

- whether the full fitted span is translation-invariant;
- whether the full fitted span is scale-invariant;
- whether the affine \(z\) direction is already contained in the elementary basis;
- which structural pivot/scale parameters must be profiled when invariance fails;
- known coefficient subspaces or active faces on which a structural direction becomes redundant or unidentified.

The last item is important numerically. A non-invariant layout can contain an invariant subspace—for example the symmetric body-free Gaussian subspace inside the \(r=1/2\) compact layout. If a fit lands on or very near such a subspace, the structural profile can become flat even though the general layout requires pivot/scale profiling.

This metadata determines both the statistical profile and its diagnostics. Expected flat directions should be recognized as structural non-identifiability rather than misdiagnosed as optimizer failure.

## 3. Base compact coordinate geometry

A useful base coordinate is specified through

\[
\boxed{
Z'(t)
=C\,t^{-r_L-1}(1-t)^{-r_R-1},
\qquad 0<t<1,
}
\]

with \(C>0\).

An additive constant fixes an anchor such as

\[
Z(t_0)=0.
\]

Near \(t=0\), the factor involving \(1-t\) tends to a finite positive constant, so the leading left geometry is determined by \(r_L\). Near \(t=1\), the leading right geometry is determined by \(r_R\).

The global product form gives independent **leading** endpoint class and exponent choices, but subleading terms on one side can contain the opposite-side parameter. The design should therefore say that the sides are **asymptotically independent at leading order**, not independent to all asymptotic orders.

### 3.1 One-sided local model

Locally, suppressing the opposite endpoint factor,

\[
Z_r'(t)=C t^{-r-1}.
\]

For \(r\neq0\),

\[
Z_r(t)
=Z_0-\frac{C}{r}t^{-r},
\]

up to an additive constant. For \(r=0\),

\[
Z_0(t)=Z_0+C\log t.
\]

The limit is continuous in \(r\).

### 3.2 \(r<0\): finite endpoint

Write \(r=-q\), \(q>0\). Then

\[
z-L\asymp t^q=t^{|r|}.
\]

The physical endpoint is finite.

For ordinary finite-boundary behavior, the canonical choice should be

\[
\boxed{r=-1.}
\]

Then \(z-L\) is locally linear in \(t\), the coordinate Jacobian is regular at the endpoint, and a fitted log coefficient alone controls algebraic boundary order. Profiling arbitrary negative \(r\) in this common case would introduce a redundant direction.

Values \(r<0\) other than \(-1\) are reserved for essential/super-algebraic zeros, where \(|r|\) controls a genuine boundary exponent.

### 3.3 \(r=0\): logarithmic infinite endpoint

\[
z\sim C\log t\to-\infty,
\qquad
t\asymp e^{z/C}.
\]

This geometry supports two distinct important tail mechanisms:

- a log energy feature gives a **logistic-type ordinary exponential tail**;
- a pole energy feature gives a **doubly-exponential tail**.

These are different model structures even though they share the same coordinate geometry.

### 3.4 \(0<r<1\): real power-exponential endpoint

\[
|z|\sim \frac{C}{r}t^{-r},
\qquad
t\asymp\left(\frac{r|z|}{C}\right)^{-1/r}.
\]

A first-order endpoint pole \(B/t\) gives

\[
Q(z)\asymp C_1|z|^{1/r}.
\]

Thus

\[
\boxed{
\lambda=\frac1r>1
}
\]

is an arbitrary real power-exponential exponent.

Examples:

\[
r=\tfrac12\Rightarrow\lambda=2,
\qquad
r=\tfrac13\Rightarrow\lambda=3,
\qquad
r=1/1.15\Rightarrow\lambda=1.15.
\]

The structural exponent is continuous; awkward rational or irrational \(\lambda\) does not require higher Laurent degree.

### 3.5 \(r=1\): algebraic-approach exponential endpoint

At \(r=1\), a first-order pole still produces a leading ordinary exponential energy:

\[
Q(z)\sim c|z|.
\]

However, this is **not** the same asymptotic class as the logistic-type \(r=0\)+log construction. The slope approaches its limiting value algebraically, and the global two-sided coordinate generates logarithmic corrections and hence power prefactors.

This endpoint is therefore its own first-class exponential subclass.

### 3.6 \(r>1\)

With a first-order pole, \(r>1\) would imply

\[
\lambda=1/r<1,
\]

which is incompatible with a standalone log-concave power-exponential tail. It is not an admissible canonical infinite-tail layout.

---

## 4. Canonical endpoint energy features

For the compact family, use a direct-energy basis of the form

\[
\boxed{
Q(t)
=
R_K(t)
+\gamma Z(t)
-A_L\log t
-A_R\log(1-t)
+B_L t^{-1}
+B_R(1-t)^{-1}
}
\]

with terms activated according to the structural endpoint layout.

Here:

- \(R_K(t)\) is the smooth body basis;
- \(\gamma Z(t)=\gamma z\) is the zero-curvature affine direction in the fitting coordinate;
- endpoint logs represent algebraic finite-boundary behavior and logistic-type exponential structure;
- first-order poles represent essential finite zeros, real power-exponential tails, algebraic-approach exponential tails, and doubly-exponential tails depending on \(r\).

All fitted coefficients enter linearly.

### 4.1 The affine \(z\) direction is required

The feature space must contain the affine-in-\(z\) energy direction.

If

\[
Q(t)\supset\gamma Z(t),
\]

then

\[
Q_t=\gamma Z',
\qquad
Q_{tt}=\gamma Z'',
\]

and therefore

\[
Q_{tt}-\frac{Z''}{Z'}Q_t=0.
\]

Thus \(\gamma\) has exactly zero curvature contribution and drops out of the convexity certificate, just as the linear coefficient does in the current polynomial model.

Keeping this direction has several benefits:

- the fitted exponential family retains the first-moment sufficient statistic in \(z\);
- when unconstrained, the corresponding score equation matches the model mean of \(z\) to the empirical mean;
- shifted Normal structure remains in the convex coefficient span rather than being forced into a nonlinear pivot search;
- skew/asymmetric linear tilts do not need to be represented solely through structural profiling.

The implementation must construct a **canonical rank-independent feature basis**, not blindly append a literal \(Z(t)\) column. Whenever \(Z\) is already an exact linear combination of elementary features, the affine direction is represented through that combination.

Examples include:

- \(r_L=r_R=0\): \(Z(t)=C[\log t-\log(1-t)]\);
- \(r_L=r_R=1\) in the unwarped product coordinate: partial fractions put \(Z\) in the span of \(1/t\), \(1/(1-t)\), \(\log t\), \(\log(1-t)\), and a constant.

More generally, every layout should provide an analytic span map when one is known. Rank reduction should be structural/algebraic rather than discovered from a numerically near-singular design matrix.

### 4.2 Higher-order poles

Higher pole orders are not part of the minimal compact family. They can be added later when they provide a genuine exact structure or useful correction.

They should not be used merely to reparameterize a leading exponent already controlled by \(r\), because only ratios such as \(m/r\) may determine the leading power and can create structural redundancy.

The first implementation should therefore prefer first-order poles and use separate potential families when an exact polynomial-in-\(z\) representation is already available.

---

## 5. Canonical endpoint layouts

The public structural vocabulary should distinguish asymptotic classes that have the same leading decay rate but different second-order behavior.

### 5.1 Ordinary finite boundary

Use

\[
\boxed{r=-1.}
\]

If the energy remains finite at the endpoint,

\[
Q(t)\to Q_0,
\]

then the density can approach a finite nonzero boundary value.

No singular endpoint energy feature is required.

### 5.2 Algebraically vanishing finite boundary

Use

\[
r=-1,
\qquad
Q(t)\supset-A\log t,
\qquad A>0.
\]

Since \(z-L\asymp t\),

\[
Q(z)\sim-A\log(z-L),
\]

and

\[
\boxed{
f(z)\asymp(z-L)^A.
}
\]

This is the natural Gamma/Beta-style finite-boundary mechanism.

There is no need to profile \(r<0\) here: with general negative \(r\), only the ratio \(A/|r|\) determines the leading algebraic exponent, creating a redundant structural direction.

### 5.3 Essential / super-algebraic finite-boundary zero

Use

\[
r<0,
\qquad
Q(t)\supset \frac{B}{t},
\qquad B>0.
\]

Because

\[
z-L\asymp t^{|r|},
\]

we obtain

\[
Q(z)\asymp
\frac{C_1}{(z-L)^{1/|r|}},
\]

hence

\[
\boxed{
f(z)\asymp
\exp\!\left[-\frac{C_1}{(z-L)^p}\right],
\qquad
p=\frac1{|r|}.
}
\]

Here \(r\) controls a genuine boundary shape and is not redundant.

### 5.4 Logistic-type exponential tail

Use

\[
\boxed{r=0}
\]

with a logarithmic energy feature

\[
Q(t)\supset-A\log t,
\qquad A>0.
\]

Since

\[
t\asymp e^{z/C},
\]

we have

\[
Q(z)\sim-\frac{A}{C}z
\]

on the left, giving

\[
\boxed{
f(z)\asymp e^{-(A/C)|z|}.}
\]

The rate \(A/C\) is in canonical \(z\)-units. Under

\[
x=c_0+h_0[\mu+s z],
\]

the physical \(x\)-rate is

\[
\boxed{a_x=\frac{A}{h_0sC}.}
\]

The defining second-order behavior is that the limiting slope is approached **exponentially fast** in \(|z|\). For the base two-sided coordinate,

\[
Z(t)
=C\log t+O(t),
\]

so corrections are powers of \(t\), i.e. exponentially small in \(|z|\).

This should be the default public meaning of an ordinary `exponential` tail because it naturally contains the motivating logistic-type cases.

Natural exact or structurally aligned examples include:

- logistic;
- log-Beta-prime;
- log-GB2;
- the exponential side of log-Gamma;
- the light exponential side of Gumbel-type models;
- the exponential side of Ex-Gaussian-type tails.

### 5.5 Real power-exponential tail

Use

\[
0<r<1,
\qquad
Q(t)\supset\frac{B}{t},
\qquad B>0.
\]

Then

\[
\boxed{
f(z)\asymp
\exp(-C_1|z|^\lambda),
\qquad
\lambda=1/r>1.
}
\]

This includes arbitrary real exponents continuously.

The endpoint parameter should normally be determined by a requested \(\lambda\),

\[
r=1/\lambda,
\]

rather than freely profiled unless automatic tail-shape estimation is explicitly requested.

### 5.6 Algebraic-approach exponential tail

Use

\[
\boxed{r=1}
\]

with a first-order pole

\[
Q(t)\supset\frac{B}{t}.
\]

For the base two-sided coordinate with opposite-side parameter \(r_R\),

\[
Z'(t)
=Ct^{-2}(1-t)^{-r_R-1}
=Ct^{-2}\left[1+(r_R+1)t+O(t^2)\right].
\]

Integrating,

\[
Z(t)
=-\frac{C}{t}
+C(r_R+1)\log t
+O(1).
\]

Therefore

\[
\frac{B}{t}
=
\frac{B}{C}|z|
-B(r_R+1)\log|z|
+O(1).
\]

If an additional log feature \(-A\log t\) is present,

\[
Q(z)
=
\frac{B}{C}|z|
+\left[A-B(r_R+1)\right]\log|z|
+O(1),
\]

and the density behaves as

\[
\boxed{
f(z)
\asymp
|z|^{B(r_R+1)-A}
\exp[-(B/C)|z|].
}
\]

The leading exponential rate \(B/C\) is in canonical \(z\)-units. In physical \(x\)-units it is

\[
\boxed{b_x=\frac{B}{h_0sC}.}
\]

For the **unwarped base product coordinate**, far-tail log-concavity requires the algebraic curvature coefficient to be nonnegative:

\[
\boxed{B(r_R+1)-A\ge0.}
\]

This closed-form coefficient is layout-specific. A positive interior warp changes the subleading expansion of \(Z\), and therefore can change the logarithmic prefactor coefficient. The exact global certificate is always authoritative; asymptotic diagnostics must be derived from the full coordinate actually used by the layout.

This class has **algebraically decaying curvature**, unlike logistic-type exponential tails, whose curvature decays exponentially.

Natural homes include exponential tails with power prefactors and hyperbolic-type structures when the coordinate includes an appropriate interior warp.

### 5.7 Doubly-exponential tail

Use

\[
\boxed{r=0}
\]

with a first-order pole

\[
Q(t)\supset\frac{B}{t},
\qquad B>0.
\]

Because

\[
t^{-1}\asymp e^{|z|/C},
\]

we obtain

\[
\boxed{
f(z)
\asymp
\exp[-B e^{|z|/C}].
}
\]

In physical \(x\)-units the distance scale inside the outer exponential is \(h_0sC\), so the corresponding inverse scale is \(1/(h_0sC)\).

A log term can coexist as a lower-order exponential slope component when an exact named structure requires it.

---

## 6. Endpoint summary table

For one side, using the left endpoint notation \(t\to0\), the following leading behaviors are written in canonical \(z=Z(t)\) units; public diagnostics convert rates/scales to physical \(x\)-units:

| Physical class | Coordinate layout | Leading energy feature | Leading behavior |
|---|---|---|---|
| finite, nonzero density allowed | \(r=-1\) | none | \(f(L+)\) finite/nonzero possible |
| finite, algebraic zero | \(r=-1\) | \(-A\log t\) | \(f\sim(z-L)^A\) |
| finite, essential zero | \(r<0\) | \(B/t\) | \(f\sim\exp[-C/(z-L)^{1/|r|}]\) |
| exponential, logistic-type | \(r=0\) | \(-A\log t\) | \(e^{-c|z|}\), exponentially fast slope settling |
| power-exponential | \(0<r<1\) | \(B/t\) | \(\exp[-C|z|^{1/r}]\) |
| exponential, algebraic approach | \(r=1\) | \(B/t\), optional log | \(|z|^a e^{-c|z|}\), algebraic slope settling |
| doubly exponential | \(r=0\) | \(B/t\) | \(\exp[-B e^{|z|/C}]\) |

The right endpoint uses the mirror features \((1-t)^{-1}\) and \(-\log(1-t)\).

---

## 7. Leading side independence and subleading coupling

The compact coordinate allows the **leading endpoint geometry** on the two sides to be specified independently:

\[
(r_L,\text{left features})
\qquad\text{and}\qquad
(r_R,\text{right features}).
\]

This determines independently:

- finite versus infinite support;
- leading tail class;
- leading power-exponential exponent;
- leading essential-boundary exponent.

However, the global coordinate is one analytic function. In the base product form, the opposite-side factor contributes to the subleading expansion. The \(r=1\) calculation above is the clearest example: the left exponential power prefactor depends on \(r_R\).

Therefore:

- public claims should say **independently specifiable leading endpoint behavior**;
- diagnostics for subleading rates or prefactors must use the full coordinate, not a one-sided formula;
- the body and optional log terms can absorb useful subleading corrections during fitting;
- exact named-family representations may require a positive interior warp \(h(t)\), discussed later.

---

## 8. Exact log-concavity in \(z\) and \(x\)

The transformed coordinate is an internal representation only. The guarantee remains log-concavity in the fitting coordinate \(z\), hence in \(x\) under an affine presentation transform.

Since

\[
z=Z(t),
\]

we have

\[
\frac{dQ}{dz}
=\frac{Q_t}{Z'},
\]

and

\[
\boxed{
\frac{d^2Q}{dz^2}
=\frac{Q_{tt}-(Z''/Z')Q_t}{(Z')^2}.
}
\]

Because \((Z')^2>0\),

\[
Q''(z)\ge0
\iff
Q_{tt}-\frac{Z''}{Z'}Q_t\ge0.
\]

For the base coordinate,

\[
\frac{Z''}{Z'}
=-\frac{r_L+1}{t}
+\frac{r_R+1}{1-t}.
\]

Thus

\[
\boxed{
\mathcal L_{r_L,r_R}[Q]
:=
t(1-t)Q_{tt}
+\left[(r_L+1)(1-t)-(r_R+1)t\right]Q_t
\ge0.
}
\]

The operator \(\mathcal L\) is linear in \(Q\). Since \(Q\) is linear in fitted coefficients, the transformed curvature is linear in those coefficients.

For a basis consisting of:

- a polynomial body;
- fixed-order endpoint poles;
- endpoint logs;
- the zero-curvature affine \(Z(t)\) direction,

the expression is rational with only known fixed endpoint denominators. Clearing the necessary positive denominators produces a univariate polynomial

\[
C_\theta(t)
\]

whose coefficients are linear in \(\theta\).

Therefore the exact constraint is

\[
\boxed{
C_\theta(t)\ge0
\qquad\forall t\in[0,1].
}
\]

This is exactly the kind of condition handled by the existing Markov–Lukács / SOS cone machinery and rigorous univariate certificates.

---

## 9. Structural endpoint factors in the certificate

A generic “multiply by a worst-case denominator and certify the resulting polynomial” implementation is not sufficient numerically.

Canonical endpoint layouts can create **forced zeros** in the cleared certificate polynomial.

### 9.1 Pole at \(r=1\)

For a left pole \(B/t\), the most singular endpoint coefficient in \(\mathcal L[Q]\) is proportional to

\[
B(1-r_L).
\]

At the canonical algebraic-approach exponential layout \(r_L=1\), this term vanishes identically. Positivity is carried by the next asymptotic term.

### 9.2 Log at \(r=0\)

For a left log feature \(-A\log t\), the leading singular coefficient contains a factor proportional to \(r_L\). At the canonical logistic-type layout \(r_L=0\), it likewise vanishes structurally.

### 9.3 Required implementation rule

Each canonical endpoint configuration should provide an **analytically reduced certificate map**:

1. form the transformed curvature symbolically for that structural layout;
2. cancel known positive denominators;
3. factor out known endpoint powers that are forced by structure;
4. build the cone on the residual minimal-degree polynomial;
5. run the numerical/rational certificate on that residual polynomial.

This avoids:

- false borderline failures from certificate tolerances applied to structural zeros;
- unnecessary degree inflation;
- poor conditioning near canonical structural values;
- confusing a forced zero with a small fitted curvature margin.

The endpoint layouts are few enough that these factorizations can be encoded explicitly rather than delegated to a fragile generic simplifier.

For continuously profiled power exponents \(0<r<1\), conditioning as \(r\to1\) should be benchmarked carefully. The canonical \(r=1\) exponential class remains a separate structural layout rather than merely the endpoint of a numerical profile.

---

## 10. Exponential-family likelihood

The model should separate **pure numerical preconditioning** from the **statistical structural map**.

Let

\[
y=\frac{x-c_0}{h_0}
\]

be a fixed data-derived numerical coordinate, with \(c_0,h_0\) chosen by robust equivariant rules such as weighted median/MAD. These values are not statistical parameters and must not determine the model family.

For a compact layout that requires structural location/scale, write

\[
\boxed{
y=\mu+s Z_\rho(t),\qquad s>0,\quad 0<t<1,}
\]

where:

- \(\mu\) is a structural pivot in preconditioned units;
- \(s\) is a structural scale in preconditioned units;
- \(\rho\) denotes endpoint exponents and any interior-warp parameters;
- \(Z_\rho\) is gauge-normalized internally so its remaining multiplicative constant is not redundant with \(s\).

For affine-invariant families, \(\mu\) and/or \(s\) can be omitted because the linear coefficient space absorbs their effect exactly.

Fix the potential family and all structural parameters \((\mu,s,\rho)\). Let

\[
Q_\theta(t)=\theta^\top T(t).
\]

The density with respect to \(x\) is proportional to \(\exp[-Q_\theta(t(x))]\), and

\[
dx=h_0 s Z_\rho'(t)\,dt.
\]

Therefore the physical normalizer is

\[
\boxed{
\mathcal Z_x(\theta;\mu,s,\rho)
=h_0 s
\int_0^1
\exp[-Q_\theta(t)]Z_\rho'(t)\,dt.
}
\]

For fixed structural parameters, \(h_0s\) is constant with respect to \(\theta\). The coefficient negative log-likelihood is therefore, up to structural constants,

\[
L(\theta\mid\mu,s,\rho)
=\sum_i Q_\theta(t_i)
+n\log\int_0^1 e^{-Q_\theta(t)}Z_\rho'(t)\,dt.
\]

Hence

\[
\boxed{
\nabla_\theta L
=n\left(E_\theta[T]-\bar T_{\rm data}\right),
}
\]

and

\[
\boxed{
\nabla_\theta^2L
=n\operatorname{Cov}_\theta(T).
}
\]

Nothing fundamental changes in the high-dimensional optimizer:

- the objective is convex;
- the Hessian is positive semidefinite;
- Newton/conic Newton remains appropriate;
- Fisher assembly remains covariance assembly;
- warm starts and line searches retain their interpretation;
- the cone constraints remain linear in the fitted coefficient representation.

### 10.1 Structural profile likelihood

When \(s\) is a statistical structural parameter, its Jacobian is not a discardable constant for **point observations**.

For weighted point rows, the explicit scale contribution to the profile objective is

\[
\boxed{
\left(\sum_{i\in\mathcal P} w_i\right)\log s,
}
\]

where \(\mathcal P\) is the set of point-observation rows. If the implementation uses normalized observation weights or a per-unit-weight NLL, the same formula is interpreted in that normalization. The fixed preconditioning contribution involving \(h_0\) can be dropped because \(h_0\) is not estimated.

Interval-censored rows contribute probability masses rather than point densities, so they have **no explicit \(\log s\) Jacobian term**. They still depend on \(s\) and \(\mu\) through their transformed bounds:

\[
t_{\rm lo}=Z_\rho^{-1}\!\left(\frac{y_{\rm lo}-\mu}{s}\right),
\qquad
t_{\rm hi}=Z_\rho^{-1}\!\left(\frac{y_{\rm hi}-\mu}{s}\right).
\]

Thus “no Jacobian term” must not be read as “no structural-scale dependence.”

The pivot \(\mu\) changes the data map but contributes no multiplicative Jacobian factor.

A structural profile evaluation must therefore:

1. map point and interval observations using the candidate \((\mu,s,\rho)\);
2. solve the inner convex coefficient problem;
3. evaluate the full physical profile NLL, adding the weighted point-mass \(\log s\) term and the interval probabilities in their transformed bounds;
4. use equivariant seeds and bounds so fit-then-transform and transform-then-fit remain consistent.

For point-only unit-weight data this reduces to the familiar \(n\log s\) contribution.

### 10.2 Mean matching through the affine feature

When the feature span contains \(Z(t)=z\) as a free sufficient statistic, its score equation is

\[
E_\theta[z]=\bar z_{\rm data}
\]

at an unconstrained optimum, where \(z=Z_\rho(t)\) is the family coordinate before the structural \(\mu+s\cdot\) map.

This preserves the useful zero-curvature linear-tilt direction of the current polynomial family. It does **not** replace a structural pivot for layouts whose feature span is not translation-invariant; a linear tilt and a shift of the coordinate bend are generally different operations.

## 11. State statistics and Fisher compression

The compact family no longer uses only raw power moments, but the same compression strategy remains available.

Typical elementary features include:

\[
1,t,t^2,\ldots,
\]

\[
\log t,\qquad\log(1-t),
\]

\[
t^{-1},\qquad(1-t)^{-1},
\]

plus the affine \(Z(t)\) direction when it is not already in their span.

The Hessian requires expectations of products \(T_jT_k\). The elementary part lives in a structured lattice containing:

- integer powers and Laurent powers;
- power \(\times\) log terms;
- log squares and cross-logs;
- pole \(\times\) log terms;
- left-pole \(\times\) right-pole terms.

These can be deduplicated and integrated once per state, then used to assemble the covariance matrix algebraically.

### 11.1 Cost of a non-elementary affine feature

For general real \(r_L,r_R\), \(Z(t)\) may not belong to the elementary product lattice. That does not destroy compression.

If there are \(m\) other fitted features, adding one independent \(Z\) feature requires only:

- \(E[Z]\);
- \(E[Z^2]\);
- \(E[ZT_j]\) for each other feature \(T_j\).

That is \(O(m)\) additional state integrals, not a new \(O(m^2)\) numerical traversal. The remaining covariance entries still assemble from the compressed elementary lattice.

When \(Z\) is already in the elementary feature span, no extra integrands are needed.

---

## 12. Body representation and degree sieve

The compact body should be represented as

\[
R_K(t)=\sum_{k=0}^{K}\beta_k\phi_k(t)
\]

using a numerically stable compact-interval basis.

Reasonable internal choices include:

- Bernstein polynomials;
- shifted Chebyshev polynomials;
- a family-specific orthogonalized basis.

Raw monomials are acceptable for a prototype but should not be assumed to be the production basis.

### 12.1 Degree growth

The information-based degree sieve should survive conceptually but operate on the **actual omitted sufficient statistic** for the family.

It should distinguish:

1. body-degree growth;
2. activation of a left structural feature;
3. activation of a right structural feature;
4. optional higher-order endpoint corrections if they are ever introduced.

The nested-model implementation must use explicit degree-elevation / embedding maps when the internal stable basis is not simple coefficient truncation.

### 12.2 Interaction with structural profiling

When a layout carries profiled structural parameters \(\eta\) (at minimum \((\mu,s)\) for non-affine-invariant spans), the sieve and the structural profile must be nested in a fixed order, and the sieve's score test must treat the profiled structural directions as nuisance parameters.

**Ordering.** The omitted-statistic test at a given body degree depends on \(\eta\): the same omitted feature can look informative at a mismatched pivot/scale and uninformative at the profiled optimum. Each sieve decision is therefore made at the **profiled optimum of the current candidate model**:

1. at the current degree \(K\) and endpoint-feature set, profile \(\eta\) to \(\hat\eta_K\) (warm-started from the previous rung's \(\hat\eta_{K-1}\));
2. run the omitted-statistic test at \((\hat\theta_K,\hat\eta_K)\);
3. if the test recommends growth, lift \(\hat\theta_K\) into the larger model by the exact embedding map, re-profile \(\eta\) at \(K+1\), and repeat.

Comparing degrees at a shared, fixed \(\eta\) is not acceptable, because it attributes structural mismatch to missing body flexibility.

**Nuisance projection.** The current score test projects the omitted statistic's residual onto the complement of the fitted statistics using the model covariance. Once \(\eta\) is estimated, the structural score directions are additional nuisance directions. For point data, with \(z_i=(y_i-\mu)/s\) and \(t_i=Z_\rho^{-1}(z_i)\), the per-observation structural scores are

\[
\frac{\partial}{\partial\mu}\bigl[Q_\theta(t_i)\bigr]
=-\frac{1}{s}\,\frac{Q_t(t_i)}{Z_\rho'(t_i)},
\qquad
\frac{\partial}{\partial s}\bigl[Q_\theta(t_i)+\log s\bigr]
=-\frac{z_i}{s}\,\frac{Q_t(t_i)}{Z_\rho'(t_i)}+\frac1s,
\]

together with the corresponding derivatives of the log normalizer. These directions must be included in the nuisance block of the efficient-score projection (alongside the fitted energy statistics), so that the test measures information in the omitted statistic **beyond** what re-profiling \(\eta\) could absorb. Omitting them overstates the omitted statistic's information and biases the sieve toward growing body degree to absorb misfit that pivot/scale should absorb. For interval rows, the structural scores come from differentiating the row log-masses through the transformed bounds, with no explicit \(\log s\) term.

When \(\eta\) is flagged non-identified at the current fit (Section 18.4), drop the corresponding direction from the nuisance block rather than projecting onto a degenerate direction; the rank-aware pseudo-inverse in the existing score test already supports this.

**Endpoint-feature activation.** Decisions to activate a left or right structural feature (items 2–3 above) follow the same rule: compare candidate layouts each at its own profiled \(\hat\eta\), and treat \(\eta\) as nuisance in any score-type activation test.

**Cost.** Each rung costs one structural profile (warm-started, typically a few profile evaluations with analytic \(\mu/s\) gradients) rather than one inner solve. Since the sieve usually visits only a few rungs, this is the same order as a single profiled fit at the final degree.

### 12.3 Polynomial-\(z\) family remains separate

The compact family should not be used to recover today's polynomial degree sieve indirectly.

Examples:

- a Normal is exactly representable in the compact family for a suitable symmetric \(r=1/2\) coordinate and first-order poles;
- a quartic potential \(Q=z^4\), exact in the polynomial family, generally requires second-order and mixed pole structure in that compact coordinate;
- odd polynomial powers likewise do not generally belong to the minimal compact feature span.

Therefore the polynomial family retains its own exact sieve and fast numerical path.

---

## 13. Exact and instructive special structures

Exactness claims for non-affine-invariant compact layouts are always conditional on the structural scale and pivot being matched or profiled. Data-derived preconditioning alone must not be treated as estimating those quantities.

### 13.1 Normal under a symmetric power coordinate

For

\[
r_L=r_R=\tfrac12,
\qquad
Z'(t)=C[t(1-t)]^{-3/2},
\]

one antiderivative is

\[
Z(t)=2C\frac{2t-1}{\sqrt{t(1-t)}}
\]

up to an additive constant.

Using

\[
(2t-1)^2=1-4t(1-t),
\]

we obtain

\[
z^2
=4C^2\left(\frac1t+\frac1{1-t}-4\right).
\]

Thus a Gaussian quadratic energy is exactly representable by first-order left/right poles plus a constant, and a shifted Gaussian uses the affine \(\gamma Z(t)\) direction.

For this particular feature span, affine shifts/scales of \(z\) are absorbed by the linear coefficients, so the structural scale/pivot need not be profiled merely to retain Gaussian exactness.

### 13.2 Logistic-type families

For the symmetric logarithmic coordinate \(r_L=r_R=0\),

\[
Z'(t)=\frac{C}{t(1-t)},
\]

so

\[
Z(t)=C[\log t-\log(1-t)]
\]

up to an additive constant.

The affine \(z\) direction is already in the span of the two log features.

This layout naturally represents logistic/Beta-prime/GB2-type structures and mixed log-plus-pole energies such as log-Gamma/Gumbel-type potentials. However, its finite feature span is generally not invariant to arbitrary shifts of the logistic bend or changes of its transition scale. Exactness therefore requires the structural pivot and scale to be matched or profiled.

For example, after gauge-normalizing the base coordinate and matching the structural scale so that

\[
z=\log\frac{t}{1-t},
\]

we have

\[
e^z=\frac{t}{1-t}=\frac1{1-t}-1.
\]

Therefore a log-Gamma energy

\[
Q(z)=-\alpha z+e^z
\]

is a linear combination of

\[
\log t,\quad \log(1-t),\quad (1-t)^{-1},\quad 1.
\]

### 13.3 Hyperbolic through a positive interior warp

A broader coordinate class allows

\[
Z'(t)
=Ct^{-r_L-1}(1-t)^{-r_R-1}h(t),
\]

with fixed positive rational \(h(t)\).

For

\[
r_L=r_R=1,
\qquad
h(t)=1-2t(1-t),
\]

choose the gauge-normalized coordinate so that

\[
Z'(t)=\frac{h(t)}{2t^2(1-t)^2}.
\]

Then

\[
z=\frac{2t-1}{2t(1-t)},
\]

and

\[
\sqrt{1+z^2}
=\frac{t^2+(1-t)^2}{2t(1-t)}
=\frac12\left(\frac1t+\frac1{1-t}\right)-1.
\]

Hence the symmetric hyperbolic energy is exactly a first-order-pole compact energy. The general energy

\[
Q(x)=\alpha\sqrt{\delta^2+(x-\mu)^2}-\beta(x-\mu)
\]

is exact when the structural pivot/scale are matched so that the normalized coordinate has the required \(\delta\), together with the affine zero-curvature tilt. This is a non-affine-invariant layout; the relevant physical scale and pivot are statistical structural parameters.

### 13.4 Gamma as an algebraic-approach oracle

Take

\[
(r_L,r_R)=(-1,1).
\]

With gauge-normalization chosen so that

\[
Z'(t)=(1-t)^{-2},
\]

and anchor \(Z(0)=0\),

\[
x=Z(t)=\frac{1}{1-t}-1=\frac{t}{1-t},
\qquad 0<t<1.
\]

For a unit-rate Gamma density with shape \(\alpha\),

\[
Q(x)=x-(\alpha-1)\log x+\text{const}.
\]

Substituting \(x=t/(1-t)\),

\[
\boxed{
Q(t)
=\frac1{1-t}-1
-(\alpha-1)\log t
+(\alpha-1)\log(1-t)
+\text{const}.
}
\]

This is exactly in the compact feature span: one right pole, both endpoint logs, and a constant. A non-unit rate is represented through the structural physical scale. For Gibbus's certified log-concave family, this oracle is relevant for \(\alpha\ge1\).

This example demonstrates why algebraic-approach exponential tails are a distinct first-class subclass rather than merely a technical alternative to logistic-type exponential tails.

## 14. Properness and endpoint feasibility

Properness should be enforced structurally rather than discovered accidentally by failed quadrature.

### 14.1 Finite endpoints

Finite support is geometrically bounded. The energy may:

- approach a finite value;
- diverge logarithmically;
- diverge through an essential pole.

The resulting density remains locally integrable whenever the activated feature coefficients satisfy the family-specific admissibility conditions.

### 14.2 Logistic-type exponential tails

At \(r=0\) with \(-A\log t\), the canonical-coordinate rate is \(A/C\). Under \(x=c_0+h_0(\mu+sZ(t))\), the physical exponential rate is \(A/(h_0sC)\). The outward rate must be positive.

The feasible coefficient condition is linear once the structural coordinate is fixed.

### 14.3 Power-exponential and doubly-exponential tails

For a first-order pole, a positive leading pole coefficient forces

\[
Q(z)\to+\infty
\]

at the corresponding infinite endpoint for the admissible layouts.

### 14.4 Algebraic-approach exponential tails

The leading pole coefficient must give the correct outward linear energy slope. Optional logarithmic corrections must remain compatible with global convexity.

For the **unwarped base product coordinate**, the far-tail condition reduces asymptotically to relations such as

\[
B(r_{\rm opposite}+1)-A\ge0.
\]

This formula must not be reused blindly for warped layouts. A positive interior warp changes the subleading expansion of \(Z\) and therefore can change the coefficient of \(\log|z|\). Every layout that reports a subleading prefactor should derive it from its full coordinate definition.

The exact reduced certificate polynomial remains the authoritative global feasibility condition in all cases.

### 14.5 Objective behavior outside the proper region

A non-normalizable trial point should produce an objective value of \(+\infty\) or an equivalent barrier result, not an exception or a fallback density.

---

## 15. Quadrature and Jacobian handling

All normalizing and expectation integrals can be performed directly on the compact interval:

\[
\int e^{-Q(z)}\,dz
=
\int_0^1 e^{-Q(t)}Z'(t)\,dt.
\]

The domain is fixed, but the Jacobian and energy can produce endpoint singularities or rapid damping. These asymptotics are known from the structural layout and should be used explicitly.

### 15.1 Ordinary finite endpoint, \(r=-1\)

The coordinate Jacobian is regular at that endpoint.

With a log feature \(-A\log t\),

\[
e^{-Q(t)}Z'(t)\asymp t^A
\]

up to a smooth positive factor.

### 15.2 Logistic-type exponential endpoint, \(r=0\)+log

Near \(t=0\),

\[
Z'(t)\asymp C/t.
\]

If

\[
Q(t)\sim-A\log t,
\]

then

\[
e^{-Q(t)}Z'(t)
\asymp C t^{A-1}
\]

when \(A\) denotes the coefficient of \(-\log t\). Equivalently, if \(a_x=A/(h_0sC)\) is the physical exponential rate, the exponent is \(a_x h_0 s C-1=A-1\). The quadrature exponent is governed by the dimensionless feature coefficient \(A\), while public rate diagnostics should use physical \(x\)-units.

Slow exponential tails therefore produce a Jacobi-type integrable singularity. Endpoint-weighted quadrature is required here just as at finite algebraic boundaries.

### 15.3 Power-exponential endpoint, \(0<r<1\)+pole

The Jacobian grows algebraically, while

\[
e^{-B/t}
\]

suppresses the endpoint faster than any power. The transformed integrand is strongly damped.

### 15.4 Algebraic-approach exponential endpoint, \(r=1\)+pole

The transformed integrand has an essential \(e^{-B/t}\)-type damping in \(t\), even though the physical tail in \(z\) is only exponential. This is numerically favorable near the compact endpoint, but the coordinate itself is highly stretched and must be evaluated stably.

### 15.5 Doubly-exponential endpoint, \(r=0\)+pole

Again the compact integrand is super-algebraically damped by \(e^{-B/t}\).

### 15.6 Numerical quadrature strategy

The quadrature layer should receive the endpoint layout and select an appropriate treatment:

- Gauss–Jacobi-style weighted rules for algebraic endpoint weights;
- explicit singularity-removing substitutions;
- spectral panels after factoring known endpoint behavior;
- asymptotic tail panels where that is cheaper;
- generic interior quadrature on the central region.

The endpoint asymptotics are part of the potential family specification and should not be rediscovered numerically at every fit.

---

## 16. Coordinate evaluation and inversion

For common canonical values, \(Z\) is elementary or reducible to elementary partial fractions. For general real \(r\), the base integral is incomplete-beta / hypergeometric in form and is awkward near divergent endpoints if evaluated naively.

The production coordinate layer should therefore use a **piecewise numerical representation of \(Z\)** rather than depend on a generic special-function call at every evaluation.

### 16.1 Endpoint asymptotic expansions

Near \(t=0\), build a structural asymptotic series such as

\[
Z(t)
=-\frac{C}{r}t^{-r}
\left(1+c_1 t+c_2 t^2+\cdots\right)
+\text{possible log terms}
\]

for \(r\ne0\), with the corresponding logarithmic series at \(r=0\).

Mirror the construction at \(t=1\).

These expansions avoid catastrophic cancellation and overflow in extreme tails.

### 16.2 Central Chebyshev patch

On a fixed central interval \([\varepsilon,1-\varepsilon]\), build a high-accuracy Chebyshev representation of \(Z(t)\) and, if useful, its derivatives.

The structural coordinate changes only during an outer structural evaluation, so this representation is built once and reused throughout the inner fit.

### 16.3 Inversion

Map observations \(z_i\to t_i\) by a safeguarded monotone Newton/Halley method using:

- asymptotic inverse seeds in the tails;
- the central Chebyshev patch in the body;
- bracketing to guarantee monotonic convergence.

A monotone inverse interpolation table is another viable optimization if benchmarks justify it.

### 16.4 Reuse

The same coordinate object is reused for:

- data mapping;
- runtime PDF/log-PDF mapping;
- PPF forward transformation;
- moments involving \(Z(t)\);
- the affine sufficient statistic;
- structural-profile evaluations.

---

## 17. Numerical preconditioning, affine invariance and structural scale

The implementation must distinguish two different affine operations that are easy to conflate.

### 17.1 Pure numerical preconditioning

Choose fixed robust values \(c_0,h_0>0\) from the data and define

\[
y=\frac{x-c_0}{h_0}.
\]

This coordinate exists only for numerical conditioning. It is not fitted and should not change the statistical family.

The polynomial-\(z\) family satisfies this automatically because its feature span is affine-invariant.

### 17.2 Statistical structural affine map

For a compact layout whose feature span is not affine-invariant, use a separate structural map

\[
\boxed{y=\mu+s Z_\rho(t),\qquad s>0.}
\]

Only the combined physical scale \(h_0s\) matters in \(x\)-units. The internal coordinate \(Z_\rho\) should therefore be gauge-normalized once, while \(s\) carries the estimable statistical scale.

Likewise, \(\mu\) is the estimable pivot/bend location in preconditioned units when translating the layout changes its feature span.

This resolves the apparent \(C/h\) redundancy cleanly:

- the coordinate implementation fixes its own arbitrary multiplicative gauge;
- the data-preconditioning scale \(h_0\) remains fixed;
- the structural scale \(s\) is profiled only when the family is not scale-invariant.

### 17.3 Affine-invariant spans and invariant subspaces

The criterion is strict: the **entire fitted feature span** must be closed under the affine action. It is not enough for the leading tail feature, or one named distribution inside the layout, to be invariant.

The polynomial-\(z\) family satisfies this criterion.

Compact layouts generally do not once they include body terms, asymmetric endpoint features, or other finite-span corrections. The symmetric body-free Gaussian subspace inside the \(r=1/2\) layout is an important exception: there, scale can be absorbed by the symmetric pole coefficient and shift by the affine direction. But the full \(r=1/2\) layout with arbitrary body terms or asymmetric poles is not affine-invariant.

Therefore a compact layout should omit structural \(\mu,s\) only when its **actual active fitted span** is known to be invariant. In all other cases, pivot/scale profiling is part of the statistical model.

### 17.4 Non-affine-invariant layouts are the compact-family default

For most compact layouts, shifting or rescaling the physical coordinate changes the finite feature span. The corresponding pivot and scale are genuine model parameters.

Important examples include:

- logistic-type \(r=0\) layouts, where structural scale controls how quickly the slope settles and the pivot sets the bend location;
- power-exponential compact layouts with finite body corrections;
- asymmetric \(r=1/2\) layouts or \(r=1/2\) layouts with body terms;
- warped hyperbolic layouts, where the physical scale controls the \(\delta\)-like shape parameter;
- finite-boundary compact layouts with body terms;
- other interior-warp families whose exactness depends on a characteristic coordinate scale or center.

Gauge-fixing the coordinate and then silently inheriting the data-derived MAD scale would make the represented density family depend on a numerical preconditioning choice. That is not acceptable. Structural \((\mu,s)\) profiling should therefore be treated as baseline compact-family work, not as an optional late refinement.

### 17.5 Equivariance

Structural scale and pivot should be seeded and bounded by equivariant statistics such as median and MAD, but they are estimated by the profile likelihood rather than fixed to those seeds.

Under an outer affine transformation of the data, the preconditioning coordinate, profile seeds and structural optimum must transform consistently. This preserves fit-then-transform versus transform-then-fit equivariance while still optimizing over the intended statistical family.

## 18. Structural parameters and outer profiling

For fixed structural parameters \(\eta\), define

\[
\hat\theta(\eta)
=\arg\min_\theta L(\theta;\eta)
\]

subject to the exact conic log-concavity constraints, and profile

\[
L_{\rm prof}(\eta)=L(\hat\theta(\eta);\eta).
\]

The architecture should distinguish three parameter categories.

### 18.1 Discrete topology / class choices

These determine which endpoint layout is active and are selected discretely rather than optimized as ordinary smooth parameters. Examples include:

- polynomial-\(z\) versus compact family;
- finite ordinary/algebraic/essential endpoint;
- logistic-type exponential;
- algebraic-approach exponential;
- doubly exponential;
- selected fixed interior-warp layouts.

Canonical values such as \(r=-1\), \(r=0\), and \(r=1\) belong here when they identify qualitatively distinct structures.

### 18.2 Continuous structural parameters

These define the function space but do not enter linearly in the energy coefficients. They may include:

- a requested/free real power exponent through \(r=1/\lambda\);
- an essential-boundary exponent through negative \(r\);
- the structural pivot \(\mu\) for non-translation-invariant layouts;
- the structural scale \(s\) for non-scale-invariant layouts;
- parameters of a fixed positive rational or exponential interior warp.

The coordinate object's arbitrary multiplicative gauge is **not** a structural parameter; it is fixed internally. Only the physical structural scale after that gauge fixing is profiled.

### 18.3 Linear energy parameters

Keep in the convex inner fit whenever possible:

- body coefficients;
- endpoint log amplitudes;
- endpoint pole amplitudes;
- the affine \(z\) tilt;
- any other coefficient multiplying a fixed energy feature.

Do not move these into the nonlinear outer loop merely because the coordinate is nonlinear.

### 18.4 Numerical profile strategy

For most compact layouts, the baseline outer problem already includes the two-dimensional structural pair \((\mu,s)\). Initial implementation should therefore prioritize efficient profile evaluation rather than rely on a dense grid.

Recommended strategy:

- deterministic equivariant seeds and bounds;
- warm-started inner convex fits;
- analytic envelope-theorem profile gradients for \(\mu\) and \(s\) whenever the inner optimum is locally regular;
- safeguarded quasi-Newton or trust-region steps in regular regions;
- bounded/bracketed or derivative-free fallback when the active certificate face changes or the profile is visibly nonsmooth;
- one-dimensional Brent searches for isolated exponent/warp parameters where appropriate;
- explicit structural bounds;
- full physical profile NLL, including the weighted point-observation \(\log s\) term whenever structural scale is estimated.

For logistic-type layouts, a natural scale seed can be derived from the MAD, but the seed must not be mistaken for the fitted structural scale.

The profile value function should not be assumed globally smooth through active-set or structural-class transitions.

The profiler must also detect nearly flat structural directions. When the fit lies on or near an affine-invariant coefficient subspace, it should keep the equivariant seed as the representative value rather than drift arbitrarily along the flat valley, and report the corresponding structural direction as non-identified.

### 18.5 Nonregular and singular class selection

Comparisons among finite, logistic-exponential, power-exponential, algebraic-exponential and double-exponential layouts can occur at nonregular boundaries where parameters disappear or change interpretation. Ordinary chi-square likelihood-ratio references should not be assumed automatically.

Selection between logistic-type and algebraic-approach exponential subclasses is scientifically narrower than selecting among completely different leading tail classes, but the layouts are still generally nonnested.

For regular, well-identified candidate fits, information criteria may be reported only if their parameter counts include all identifiable profiled structural parameters as well as the effective coefficient dimension. If a structural direction is non-identified, the model is singular and ordinary BIC is not formally justified merely by deleting that parameter from the count. In such cases prefer held-out likelihood, simulation-calibrated criteria, or a method explicitly designed for singular models; report the identifiable local rank only as a diagnostic.

## 19. Broader admissible coordinate class

The base beta/Jacobi derivative is only the simplest useful family.

The transformed curvature identity requires only that

\[
\frac{Z''}{Z'}
=\frac{d}{dt}\log Z'
\]

be rational, with known positive denominators.

A broad class is therefore

\[
\boxed{
Z'(t)
=
C\,t^{-r_L-1}(1-t)^{-r_R-1}
 e^{p(t)}h(t),
}
\]

where:

- \(p(t)\) is a fixed low-degree polynomial;
- \(h(t)>0\) is a fixed positive rational function on \((0,1)\).

Then

\[
\frac{Z''}{Z'}
=
-\frac{r_L+1}{t}
+\frac{r_R+1}{1-t}
+p'(t)
+\frac{h'(t)}{h(t)},
\]

which remains rational.

After clearing fixed positive denominators, exact log-concavity is still one polynomial nonnegativity condition linear in the fitted energy coefficients.

Interior warps allow the body geometry and exact named-family structure to improve without changing the leading endpoint classes.

The hyperbolic example in Section 13 is the canonical worked example for why this extension matters.

---

## 20. Runtime distribution methods

No core distribution method becomes mathematically intractable.

### 20.1 PDF and log-PDF

Given \(x\):

1. apply fixed numerical preconditioning to obtain \(y=(x-c_0)/h_0\);
2. invert the structural affine map \(z=(y-\mu)/s\) when the layout uses one;
3. map \(z\to t\);
4. evaluate \(Q(t)\);
5. subtract the stored physical log normalizer, including the fixed/preprofiled affine Jacobian terms.

The coordinate Jacobian \(Z'(t)\) is **not** multiplied into the \(x\)-density evaluation; it appears only when changing the integration variable.

### 20.2 Potential derivatives

\[
Q_z=\frac{Q_t}{Z'},
\]

and

\[
Q_{zz}
=\frac{Q_{tt}-(Z''/Z')Q_t}{(Z')^2}.
\]

Higher derivatives can be derived family-wise if required by diagnostics.

### 20.3 CDF

\[
F(z)
=
\frac{
\int_0^{t(z)}e^{-Q(u)}Z'(u)\,du
}{
\int_0^1e^{-Q(u)}Z'(u)\,du
}.
\]

A spectral/panel representation can be built directly in \(t\).

### 20.4 PPF

Solve the compact-coordinate CDF for \(t\), then evaluate

\[
z=Z(t),\qquad x=c_0+h_0[\mu+s z],
\]

with \(\mu=0,s=1\) for layouts that do not require a structural affine map.

The final PPF step uses the forward coordinate, not \(Z^{-1}\).

### 20.5 Survival and hazard

Use the compact CDF/survival representation and the ordinary density:

\[
S=1-F,
\qquad
h=f/S.
\]

### 20.6 Moments

\[
E[X^k]
=
\frac{
\int_0^1\{c_0+h_0[\mu+sZ(t)]\}^k e^{-Q(t)}Z'(t)\,dt
}{\int_0^1 e^{-Q(t)}Z'(t)\,dt}.
\]

Structural tail information can determine moment existence before numerical integration.

### 20.7 Entropy

With normalized density

\[
\log f(Z(t))=-Q(t)-\log\mathcal Z,
\]

entropy is

\[
H=E[Q]+\log\mathcal Z
\]

plus the affine contribution \(\log(h_0s)\) when a structural scale \(s\) is present (with \(s=1\) otherwise).

### 20.8 Mode

Since \(Z\) is monotone, minimizing \(Q(z)\) is equivalent to minimizing \(Q(t)\). The mode can therefore be found in compact coordinates.

### 20.9 Sampling

Inverse-CDF sampling is natural:

1. draw \(u\sim U(0,1)\);
2. solve \(F_t(t)=u\);
3. return the forward map to \(x\).

### 20.10 Interval censoring and truncation

A monotone coordinate maps intervals to intervals. Interval likelihoods remain CDF differences and truncation remains probability renormalization over a compact-coordinate interval. Structural \(\mu,s\) change the transformed interval endpoints, but interval rows carry no explicit point-density \(\log s\) Jacobian term.

### 20.11 Tail diagnostics

The fitted structure should report:

- endpoint class;
- leading physical rate or exponent;
- whether the endpoint is finite;
- essential-boundary exponent if present;
- subleading power prefactor where structurally meaningful;
- structural coordinate parameters;
- exact certificate margin after structural factors are removed.

Subleading diagnostics must use the full two-sided coordinate because of cross-end coupling.

---

## 21. Mixtures

Each component can have its own potential family and coordinate structure. Once compact layouts carry structural pivot and scale, those parameters should be updated component-wise rather than through one high-dimensional outer profile.

For a mixture with component \(k\), write its compact structural parameters as \((\mu_k,s_k)\). A practical v1 strategy is ECM/ECME-like:

1. perform the usual responsibility update;
2. for each component independently, seed \((\mu_k,s_k)\) from responsibility-weighted robust statistics;
3. take a small number of warm-started profile steps in \((\mu_k,s_k)\) using that component's responsibility-weighted data;
4. at the accepted structure, update the component's linear energy coefficients by the exact convex inner solve;
5. accept structural/linear conditional updates only when they do not decrease the relevant EM auxiliary objective, or use an ECME step that directly increases observed likelihood.

For point rows, the component scale term is weighted by the responsibility-weighted point mass,

\[
\left(\sum_{i\in\mathcal P} r_{ik}w_i\right)\log s_k.
\]

Interval rows contribute no explicit \(\log s_k\) term, although their transformed bounds still depend on \((\mu_k,s_k)\).

This replaces an infeasible joint \(2K\)-dimensional structural grid with \(K\) small conditional two-dimensional problems. A later joint polish can remain restricted to linear coefficients initially; bringing structural variables into a joint nonlinear polish is optional.

To control v1 complexity:

- use one requested endpoint-class specification across components unless there is strong evidence otherwise;
- allow component-private linear body/tail coefficients and affine \(z\) directions;
- profile component-private \((\mu_k,s_k)\) for non-invariant compact layouts;
- keep additional exponent and interior-warp parameters fixed or shared across components initially.

Auto-\(K\) screening should use an endpoint family compatible with the requested tails. Screening heavy/exponential-tailed data with an overly light polynomial family can manufacture extra mixture components to imitate tail mass.

## 22. Numerical conditioning

Primary risks include:

- \(t\) extremely close to 0 or 1;
- endpoint log and pole evaluation;
- correlation between body and endpoint features;
- forced endpoint zeros in the raw certificate polynomial;
- flat structural profiles;
- coordinate inversion in extreme tails;
- flat or redundant structural scale/pivot directions on invariant coefficient subspaces;
- near-boundary power exponents such as \(r\to1^-\);
- redundant feature columns when \(Z(t)\) already lies in the elementary span.

Mitigations:

- compute \(\log t\) and \(\log(1-t)\) with endpoint-safe primitives;
- evaluate pole features with scaled/log-domain formulas when necessary;
- use stable compact polynomial bases;
- analytically factor structural endpoint zeros before cone construction;
- rank-reduce each family feature space;
- fix only the coordinate object's internal multiplicative gauge; profile physical structural scale for non-invariant layouts;
- use asymptotic coordinate expansions in the tails;
- use a central Chebyshev coordinate patch;
- use safeguarded monotone inversion;
- constrain structural searches to identifiable parameterizations;
- detect near-flat profile directions, retain equivariant seeds as representatives, and report non-identification explicitly.

---

## 23. Scope and out-of-scope behavior

The compact endpoint family is broad but not universal.

It is designed for smooth log-concave densities whose endpoints can be represented by combinations of:

- ordinary finite boundaries;
- algebraic finite-boundary zeros;
- essential finite-boundary zeros;
- logistic-type exponential tails;
- power-exponential tails with arbitrary real exponent \(\lambda>1\);
- algebraic-approach exponential tails;
- doubly-exponential tails;
- smooth lower-order body corrections.

The polynomial family separately handles exact polynomial potentials efficiently.

Distributions whose far tails are not log-concave remain outside Gibbus regardless of coordinate choice. For example, a tail of the form

\[
f(z)\asymp |z|^{-3/2}e^{-\alpha|z|}
\]

has

\[
Q(z)=\alpha|z|+\tfrac32\log|z|+O(1),
\]

so

\[
Q''(z)\sim-\frac{3}{2z^2}<0.
\]

Such a tail is eventually not log-concave and should not be listed as an exact or approximate target of a certified log-concave family.

Non-smooth cusp models that violate the global-analytic design goal likewise remain outside the exact family even if their leading tail exponent is admissible.

---

## 24. Validation plan

The design should be validated in layers.

### 24.1 Mathematical identities

For every family/layout:

- verify the symbolic curvature operator;
- verify the coefficient-to-certificate linear map;
- verify structural endpoint factor cancellation;
- compare the certified polynomial condition against direct global minimization of \(Q''(z)\);
- verify affine-feature zero curvature;
- finite-difference-check likelihood gradients;
- compare analytic covariance Hessians with finite differences.

### 24.2 Exact-structure oracles

Use known exact cases to test the representation:

- current polynomial family: Normal, polynomial-potential controls, applicable generalized-normal even powers;
- compact \(r=1/2\) family: Normal;
- \(r=0\) logistic family: logistic, log-Beta-prime/log-GB2-type cases;
- mixed \(r=0\) log/pole layouts: log-Gamma and Gumbel-type cases;
- warped \(r=1\) family: hyperbolic example with structural scale/pivot matched or profiled;
- mixed \((r_L,r_R)=(-1,1)\) compact layout: Gamma with \(\alpha\ge1\), including the algebraic-approach right tail;
- finite \(r=-1\)+log layouts: Beta/Gamma-style boundary powers where appropriate.

For generalized-normal shapes that are not exactly in the chosen compact body span, report **tail-class exactness**, not density exactness.

### 24.3 Statistical benchmarks

Use multiple sample sizes and seeds. Primary metrics:

- numerical KL divergence to truth;
- held-out log-likelihood on a large independent sample;
- CDF sup error;
- quantile error;
- explicit tail-probability error;
- recovery of physical tail rate/exponent;
- certificate margin.

Secondary metrics:

- training likelihood gap;
- runtime;
- coordinate-build cost;
- coordinate-inversion cost;
- number of inner Newton steps;
- number of outer structural evaluations;
- conditioning of the conic map and Fisher matrix.

### 24.4 Robustness tests

Include:

- deliberately misspecified endpoint class;
- \(r\) near structural boundaries;
- slow logistic-type exponential tails;
- strongly asymmetric endpoint classes;
- sparse-tail samples;
- finite endpoints with weak and strong algebraic zeros;
- essential-zero synthetic cases;
- high-degree body fits;
- mixture screening under correct and incorrect tail families.

### 24.5 Equivariance and structural identifiability

Check fit-then-transform against transform-then-fit under the public affine transform. The fixed preconditioning statistics must transform equivariantly; structural pivot/scale parameters for non-affine-invariant layouts must transform in physical units; gauge-fixed internal coordinate parameters should remain invariant by construction.

Changing only numerical preconditioning should produce fitted densities that agree within numerical tolerance whenever the structural profile has a unique optimum. If the profile is flat, nearly flat, or multimodal, report that condition explicitly rather than requiring arbitrary structural coordinates to match. The density-level comparison remains authoritative.

Validation should include deliberately near-invariant cases, such as nearly Gaussian data under the \(r=1/2\) compact layout, to verify that the profiler retains stable equivariant representatives and flags unidentified directions rather than wandering along a flat valley.

---

## 25. Implementation phases

### Phase 0 — stand-alone mathematical prototype

Implement outside the production package:

- compact coordinate layouts;
- direct energy features;
- affine \(z\) direction;
- normalizer and expectations;
- gradient and covariance Hessian;
- exact curvature-polynomial construction;
- structural endpoint factoring;
- endpoint-aware quadrature;
- coordinate forward/inverse evaluation.

Start with single-component point data. First verify fixed structural parameters; then, still within the prototype, profile scale/pivot for at least one non-affine-invariant layout so the physical likelihood and its \(n\log s\) term are exercised before production architecture is committed.

Priority layouts:

1. \(r=0\) logistic-type exponential;
2. \(r=0\) mixed log/pole for log-Gamma/Gumbel-type models;
3. \(r=1/2\) Gaussian power coordinate;
4. one general \(0<r<1\) real power-exponential case;
5. \(r=-1\) finite algebraic boundary;
6. \(r=1\) algebraic-approach exponential;
7. hyperbolic positive-warp example;
8. Gamma at \((r_L,r_R)=(-1,1)\) as an algebraic-approach oracle.

### Phase 1 — potential-family abstraction, no behavior change

Introduce internal abstractions for:

- coordinate evaluation;
- feature evaluation;
- state-statistic layout;
- curvature certificate map;
- endpoint metadata.

Route today's polynomial-\(z\) model through the abstraction while preserving its dedicated fast kernels.

Gate:

- full existing test suite passes;
- polynomial outputs unchanged;
- polynomial performance unchanged within benchmark tolerance.

### Phase 2 — first complete compact-family fit

Add production support for:

- canonical compact endpoint layouts;
- rank-independent feature construction;
- affine \(z\) feature;
- exact structurally factored certificate maps;
- endpoint-aware state quadrature;
- single-component point-data fitting;
- mandatory structural \((\mu,s)\) profiling for non-affine-invariant compact spans;
- analytic \(\mu/s\) profile gradients in regular regions, with safeguarded fallback at active-face changes;
- flat-direction detection and structural-identifiability diagnostics.

A fixed-seed \((\mu,s)\) mode may exist as a development scaffold, but it must not be treated as the statistical compact-family fit or used for exactness/quality claims.

### Phase 3 — runtime methods

Implement:

- PDF/log-PDF;
- derivatives;
- CDF/PPF;
- moments;
- entropy;
- survival/hazard;
- sampling;
- truncation;
- structural tail diagnostics.

### Phase 4 — additional structural-shape profiling

Extend the Phase 2 \((\mu,s)\) profile to additional genuinely identifiable continuous shape parameters:

- requested/free real power exponent;
- essential-zero exponent;
- positive interior-warp parameters.

Keep canonical endpoint topologies discrete. Gauge-fix only the coordinate object's arbitrary internal scale; do not freeze a statistically meaningful physical scale to the data-preconditioning value.

### Phase 5 — observation types and support integration

Add:

- interval censoring, with structural scale affecting transformed bounds but no explicit interval-row \(\log s\) Jacobian;
- weighted observations, with the explicit scale term weighted only by point-observation mass;
- existing support/observation variants;
- any family-specific endpoint transforms needed by spectral integration.

### Phase 6 — mixtures

Add compact-family components using per-component ECM/ECME-style structural updates for \((\mu_k,s_k)\), responsibility-weighted seeds and point-mass Jacobian terms. Keep extra exponent/warp parameters fixed or shared initially, and retain a conservative linear-coefficient joint polish.

### Phase 7 — automatic endpoint-family selection

Only after fixed-family benchmarks are strong, add model selection. Prioritize selection between the two exponential subclasses because they share the same leading rate but differ materially in second-order behavior. Then extend selection among:

- polynomial-\(z\);
- finite ordinary/algebraic/essential layouts;
- logistic-type exponential;
- power-exponential;
- algebraic-approach exponential;
- doubly-exponential;
- selected fixed interior-warps.

Selection criteria must account for nonregular class boundaries, identifiable profiled structural parameters and active coefficient dimension. If a candidate fit is structurally singular, ordinary BIC is not treated as formally valid; use held-out or simulation-calibrated comparison instead.

---

## 26. Public modeling semantics

The public API should expose **physical endpoint behavior**, not internal coordinate jargon.

Useful concepts include:

- `finite`;
- `finite_power` / algebraic boundary;
- `finite_essential`;
- `exponential` for an ordinary exponential leading rate, with subclass selection either explicit or automatic;
- `exponential_logistic` for exponentially fast slope settling (the default explicit subclass for logistic/log-Beta-prime/log-Gamma-like behavior);
- `exponential_algebraic` for power-prefactor / algebraically settling exponential tails (Gamma-, variance-gamma-, hyperbolic-type behavior where log-concave);
- `power` with real \(\lambda>1\);
- `double_exponential`;
- `polynomial` for today's family.

For an early implementation, explicit exponential subclasses are safer. Once both are validated, `exponential` can reasonably mean: **fit the leading exponential class and choose the second-order subclass by the validated structural-selection criterion**, while retaining explicit overrides.

The API need not expose \(r\) unless the user explicitly requests low-level structural control. A request such as

\[
\lambda=1.15
\]

maps internally to

\[
r=1/1.15.
\]

Finite ordinary/algebraic boundaries map to \(r=-1\). Doubly exponential and logistic-type exponential both use \(r=0\) but activate different energy features, so they remain distinct model structures.

All public rate diagnostics should be expressed in physical \(x\)-units. Internal \(z\)- or \(t\)-unit quantities may be exposed only when explicitly labeled.

Diagnostics should report, in this order:

1. physical endpoint class;
2. leading physical rate/exponent;
3. second-order exponential subclass where relevant;
4. fitted structural pivot/scale for non-affine-invariant layouts;
5. internal structural layout and certificate information.

Named guidance should make the subclass distinction concrete:

- logistic, log-Beta-prime, log-GB2, and the exponential side of log-Gamma: logistic-type;
- Gamma right tails, variance-gamma-type tails, and hyperbolic-type tails where log-concave: algebraic-approach.

## 27. Core guarantees

For a fixed potential family and fixed structural coordinate parameters, the design preserves the important Gibbus properties.

### One global analytic potential

No knots, splices, KDE or piecewise density definition is required.

### Exact log-concavity in \(x\)

The transformed curvature condition is mapped to one univariate polynomial positivity problem and certified exactly by the existing cone/certificate machinery after structural factors are removed.

### Convex high-dimensional fit

The fitted energy coefficients enter linearly. The coefficient MLE is an exponential-family convex optimization problem.

### Moment gradient and covariance Hessian

The likelihood retains the standard sufficient-statistic form.

### Independently specifiable leading endpoint behavior

Left and right endpoint class/exponent choices are independent at leading order, with known subleading coupling from the global analytic coordinate.

### Tractable evaluation

Normalization, CDF, PPF, moments, entropy, hazard, interval probabilities and sampling all have direct compact-coordinate formulations.

### Extensible family architecture

The polynomial model remains exact where it is strongest, while compact endpoint families handle asymptotic structures the polynomial family cannot represent efficiently.

---

## 28. Main caveats

### 28.1 The full profiled fit is not globally convex

Convexity applies to the inner coefficient fit conditional on structural choices. For most compact layouts, profiling structural pivot/scale is already part of the baseline fit; profiling an exponent or warp adds further low-dimensional nonlinear structure.

### 28.2 Exponential tails are not one second-order class

Logistic-type \(r=0\)+log and algebraic-approach \(r=1\)+pole tails share a leading exponential rate but have different curvature decay and prefactor structure. They must remain distinct model layouts.

### 28.3 The compact family does not subsume the polynomial family

Current polynomial exactness must be retained through a separate polynomial-\(z\) potential family and fast path.

### 28.4 Endpoint independence is leading-order

Subleading terms can couple the two sides through the single global coordinate.

### 28.5 Structural certificate zeros must be factored analytically

Canonical layouts produce forced endpoint zeros in naive cleared certificate polynomials. The cone should operate on a reduced residual polynomial.

### 28.6 Coordinate numerics are a real engineering subsystem

General real endpoint exponents require stable forward/inverse coordinate evaluation, endpoint asymptotics and central approximation. This work is tractable but should be treated as core numerical infrastructure.

### 28.7 Structural scale and pivot must be treated statistically when the span is not affine-invariant

The data-derived location/scale used for numerical conditioning cannot silently stand in for a model parameter. For non-affine-invariant layouts, the physical transition scale and pivot change the represented density family and must be matched, user-fixed, or profiled.

Only the arbitrary multiplicative gauge internal to the coordinate representation should be fixed by convention. If a structural scale \(s\) is profiled, point observations contribute the corresponding weighted \(\log s\) Jacobian term. Interval observations have no explicit \(\log s\) term but still depend on \(s\) through transformed bounds.

### 28.8 Automatic family selection is statistically nonregular

Class boundaries can create unidentified parameters and nonstandard likelihood-ratio behavior. Selection should be validated empirically rather than assuming ordinary asymptotic reference laws.

### 28.9 Structural profiles can be singular even inside one layout

A non-affine-invariant compact layout can contain coefficient subspaces on which pivot or scale becomes redundant. The profiler must detect these flat directions, retain stable equivariant representatives, suppress ordinary curvature-based standard errors for unidentified structural parameters, and avoid treating an ad hoc reduced parameter count as a formal BIC correction.

---

## 29. Overall design

The long-term representation should be understood as

\[
\boxed{
\text{potential family}
=
\text{coordinate geometry}
+
\text{linear energy feature space}
+
\text{exact curvature certificate map}.
}
\]

For the compact endpoint family,

\[
\boxed{
\text{endpoint layout}
+
\text{generic log/pole features}
+
\text{affine }z\text{ direction}
+
\text{smooth compact body}
}
\]

determines the potential.

The endpoint geometry decides what physical region a compact endpoint represents. The activated endpoint feature determines the leading density behavior there. The body basis handles smooth interior structure. The affine \(z\) direction preserves the zero-curvature location/tilt degree of freedom. A fixed positive interior warp can provide exact or more efficient representations when the base coordinate is insufficient.

The resulting architecture is deliberately not a claim that one coordinate family is universal. Instead, it gives Gibbus a common optimizer and certificate framework within which multiple mathematically structured potential families can coexist. The architecture keeps three layers distinct: discrete family/layout choice, low-dimensional continuous structural parameters (including physical scale/pivot when required), and high-dimensional linear energy coefficients handled by the convex solver.

That is the preferred direction because it gains flexible, independently specifiable endpoint behavior without sacrificing the properties that make the current machinery valuable:

- exact certified log-concavity;
- convex coefficient fitting;
- moment-based gradients;
- covariance Hessians;
- compact sufficient-statistic integration;
- global analytic potentials;
- tractable distribution evaluation.

---

## 30. Review critiques and implementation recommendations

The core design is strong. In particular, the proposed decomposition

\[
\text{potential family}
=
\text{coordinate geometry}
+
\text{linear energy feature space}
+
\text{exact curvature certificate map}
\]

is a useful long-term abstraction for Gibbus. The central technical advantage is not compactification by itself, but the fact that a nonlinear coordinate can coexist with a potential that remains linear in its fitted coefficients while log-concavity still reduces to univariate polynomial nonnegativity. This preserves the strongest part of the current solver architecture while allowing substantially richer endpoint geometry.

The following points should be addressed before treating the design as implementation-ready.

### 30.1 Qualify convexity claims for interval-censored likelihoods

For point observations with fixed structural parameters, the coefficient objective has the usual exponential-family form

\[
L(\theta)
=
\sum_i \theta^\top T_i
+n\log Z(\theta),
\]

with

\[
\nabla L
=
n\left(E_\theta[T]-\bar T\right),
\qquad
\nabla^2L
=
n\operatorname{Cov}_\theta(T),
\]

so the coefficient problem is convex subject to the linear/conic log-concavity constraints. Weighted point observations retain this structure.

Interval censoring changes the situation. An interval contribution has the form

\[
-\log P_\theta(a<X<b)
=
\log Z(\theta)-\log Z_{[a,b]}(\theta),
\]

which is generally a difference of convex log-partition terms. The coefficient objective is therefore not generally convex under interval censoring, and the observed Hessian need not be positive semidefinite.

Any statement such as

> The high-dimensional inner fit remains convex.

or

> The coefficient MLE is an exponential-family convex optimization problem.

should therefore be qualified as applying to point-observation likelihoods, or to observation models for which convexity has separately been established. This does not weaken the endpoint-family design itself; it only keeps the guarantee mathematically precise and consistent with the existing interval-censoring machinery.

### 30.2 Make properness part of the formal family constraint interface

The affine \(Z(t)\) direction is important because it supplies the zero-curvature tilt/location degree of freedom. The same fact means that log-concavity alone cannot guarantee normalizability. A potential may satisfy the curvature certificate while having zero or incorrectly directed recession slope at an infinite endpoint.

For that reason, a potential family should expose both

\[
\boxed{
\text{curvature certificate constraints}
+
\text{properness/recession constraints}
}
\]

as part of one formal constraint interface.

A family-specific constraint builder should return:

1. the exact polynomial nonnegativity representation needed for log-concavity; and
2. any exact linear inequalities or equalities required to ensure integrability and valid endpoint recession behavior.

Scalar properness inequalities can naturally be represented as \(1\times1\) PSD blocks or equivalent linear constraints within the same conic framework. Treating a nonnormalizable trial point as \(+\infty\) in the objective should remain a final numerical safety net, not the primary enforcement mechanism.

### 30.3 Keep the polynomial family as a sibling backend rather than generalizing its internals in place

The compact endpoint family should not be implemented by turning the current polynomial-natural layout into a universal layout class. The existing polynomial implementation exploits structure much more deeply than the public abstraction alone suggests. Its layout, cached moments, state representation, degree diagnostics, mixture machinery, serialization, post-fit state, and compiled kernels all assume polynomial-plus-boundary-log coordinates in specific ways.

The preferred architecture is therefore to place the abstraction one level above the current natural-model internals. Conceptually:

```text
PotentialFamilyBackend
    build_layout(...)
    build_state(...)
    build_objective(...)
    cone_representation(...)
    embed_model(...)
    build_runtime_state(...)
    serialize_model(...)
```

The current polynomial model can remain internally specialized and become the implementation of a `PolynomialPotentialFamily`. The new endpoint construction can have its own compact layout, coordinate object, sufficient-statistic state, integration machinery, and runtime representation.

Accordingly, the first implementation phase should aim to **wrap** the existing polynomial path behind the shared family interface rather than rewrite or route its internal numerical representation through the compact machinery. This preserves the polynomial fast path and reduces regression risk.

### 30.4 Treat structural pivot and scale as a potentially weakly identified outer problem

The distinction between numerical conditioning coordinates and physical structural parameters is correct. If changing \(\mu\) or \(s\) changes the exact represented function span, those quantities are statistical parameters and cannot silently be replaced by data-derived centering and scaling constants.

Profiling them is therefore the clean formulation. However, exact non-identifiability is not the only concern. As the compact body basis becomes richer, changes induced by shifting or rescaling the coordinate may be increasingly well approximated by changes in the body coefficients. This can create broad, shallow profile likelihoods even when no exact invariant subspace exists.

Structural diagnostics should therefore monitor approximate as well as exact non-identifiability. Useful quantities include:

- local information eigenvalues for the structural parameters;
- profile-likelihood width and asymmetry;
- projection of structural score directions onto the fitted coefficient tangent span;
- stability of the profiled optimum across degree increases;
- whether the same fitted density is recovered over materially different \((\mu,s)\) values.

Ordinary curvature-based standard errors should not be reported for structurally flat or nearly flat directions.

### 30.5 Represent affine-\(Z\) redundancy through an explicit canonical coefficient map

The affine \(Z\) direction should exist whenever it contributes an independent zero-curvature feature, but it is not independent for every endpoint layout. For example, when \(r_L=r_R=0\),

\[
Z=C\{\log t-\log(1-t)\},
\]

so adding \(Z\) separately to the two endpoint log features introduces exact linear dependence.

Each compact layout should therefore expose an explicit full-rank mapping

\[
\theta_{\text{canonical}}
\longmapsto
\theta_{\text{elementary}},
\]

where the canonical coefficient vector is guaranteed to be linearly independent. The curvature certificate, properness constraints, sufficient statistics, diagnostics, and serialization should all operate through that map rather than allowing different subsystems to rediscover rank deficiencies independently.

This is preferable to case-by-case feature deletion because it centralizes the algebraic identity of a layout.

### 30.6 Be conservative about general interior warp machinery

The generalized coordinate derivative

\[
Z'(t)
=
C t^{-r_L-1}(1-t)^{-r_R-1}e^{p(t)}h(t)
\]

is mathematically attractive because the logarithmic derivative remains rational when \(h\) is rational, preserving the polynomial curvature-certificate strategy after denominator clearing.

Architecturally, however, this is the point at which a finite family design can turn into a framework for inventing arbitrary coordinate systems. That would expand the numerical and validation surface substantially.

The recommended sequence is:

1. implement the unwarped canonical endpoint layouts;
2. implement the hyperbolic warp as one named, fixed layout with its own exact oracle;
3. add further named warped layouts only when they solve demonstrated approximation problems;
4. abstract common warp machinery only after multiple concrete layouts justify it.

A public user-defined \(h(t)\) mechanism should be deferred until the family contracts, conditioning behavior, certificate factoring, inverse-coordinate numerics, and serialization semantics are very stable.

### 30.7 Give the \(r=1\) algebraic exponential class enough freedom to control its logarithmic prefactor

For a base endpoint geometry with \(r=1\), the pole feature produces the desired leading exponential behavior, but the opposite endpoint contributes to the logarithmic correction. In the base coordinate one obtains behavior of the form

\[
\frac{B}{t}
=
\frac{B}{C}|z|
-B(r_R+1)\log|z|
+O(1).
\]

Therefore, unless a corresponding endpoint log feature is available, the power-law prefactor is partly hard-wired by the global coordinate geometry. This is mathematically valid but potentially surprising from a modeling perspective.

For an `exponential_algebraic` layout, the endpoint log feature should therefore be strongly considered part of the standard feature set rather than merely an optional embellishment. That allows the leading exponential rate and algebraic prefactor to vary separately, subject to the global log-concavity certificate.

### 30.8 Start with three exact-oracle layouts before implementing the whole endpoint vocabulary

The first prototype should be deliberately narrow. Three layouts exercise nearly every important architectural concept:

1. **Logistic-type \(r=0\) with log features.** This tests compact infinite support, log endpoint features, structural pivot/scale handling, and a coordinate with exponentially settling physical slope.
2. **Gamma-type \((r_L,r_R)=(-1,1)\).** This tests asymmetric finite/infinite endpoint geometry, pole/log interaction, and the algebraic exponential class.
3. **Normal-type \(r_L=r_R=1/2\).** This tests fractional endpoint exponents, pole-type representation, and the affine \(Z\) direction against a target the existing polynomial family already represents especially well.

These should be exact or near-exact analytic oracles with coefficient recovery, normalization, CDF/quantile, curvature-certificate, and tail-asymptotic checks.

Only after those three work cleanly should the implementation expand to arbitrary real \(0<r<1\). General warp machinery should come after that.

### 30.9 Treat the compact family as a second numerical backend, not a small feature addition

The proposed design reaches well beyond the coefficient cone. A production implementation requires family-aware machinery for:

- forward and inverse coordinate evaluation;
- endpoint asymptotics;
- compact-state quadrature;
- first and second sufficient-statistic moments;
- stable feature evaluation near compact endpoints;
- structural profiling;
- coefficient embedding during degree growth;
- post-fit density/CDF/PPF evaluation;
- mixture updates and component geometry;
- serialization and compatibility contracts;
- diagnostics and failure reporting.

This should be budgeted and reviewed as a second numerical backend. The cone solver is one of the more reusable pieces; much of the surrounding state machinery is currently polynomial-specific.

Prototype status should therefore be retained until the exact-oracle layouts demonstrate not only mathematical correctness but also a cleaner and more stable implementation path than ad hoc extensions to the current polynomial family.

### 30.10 Preserve the endpoint taxonomy rather than collapsing it into one continuous tail knob

The canonical cases

\[
r=-1,\qquad r=0,\qquad 0<r<1,\qquad r=1
\]

should remain semantically distinct layouts even if they share common low-level code.

In particular, logistic-type \(r=0\)+log tails and \(r=1\)+pole algebraic-approach exponential tails can share the same leading exponential rate while having materially different curvature decay, subleading behavior, and power-prefactor structure. They should not be presented as interchangeable parameter settings of a single generic "exponential tail" layout.

The existing design is correct to make family/layout choice discrete and leave continuous structural profiling for quantities such as physical pivot and scale.

### 30.11 Keep the separation between endpoint asymptotics and body flexibility as the main design objective

The most compelling statistical reason to pursue this architecture is that the current polynomial-potential degree simultaneously controls interior flexibility and asymptotic behavior. The compact endpoint design can separate those concerns:

\[
\boxed{
\text{endpoint asymptotics}
\quad\text{from}\quad
\text{interior smoothness}
}
\]

The coordinate geometry and endpoint features determine the physical asymptotic class, while the compact body basis controls smooth interior departures. This separation is more important than merely increasing the menu of representable tails.

It also gives Gibbus a clearer long-term model identity: a collection of finite-dimensional analytic potential families that share an exact log-concavity certificate and a common likelihood/optimization framework, rather than a single polynomial potential family with increasingly many special-case tail corrections.

### 30.12 Recommended implementation order

A conservative development sequence is:

1. define the family-level interface above the current polynomial internals;
2. wrap the existing polynomial family without changing its numerical representation;
3. formalize family-returned curvature and properness constraints;
4. implement canonical full-rank feature maps for compact layouts;
5. implement the \(r=0\), \((-1,1)\), and \((1/2,1/2)\) exact-oracle compact families;
6. add structural-profile diagnostics for weak identification;
7. validate density, CDF, quantile, normalization, coefficient recovery, certificate residuals, and tail asymptotics against the analytic oracles;
8. add arbitrary \(0<r<1\) endpoint exponents only after those cases are stable;
9. add a named hyperbolic warp only after the unwarped family is mature;
10. generalize warp infrastructure only if multiple concrete warped layouts justify the abstraction;
11. add interval-censored fitting only after the point-likelihood implementation is validated, without assuming the coefficient objective remains convex;
12. extend mixtures, automatic family selection, and serialization only after single-component semantics and structural profiling are stable.

The central recommendation is therefore to preserve the mathematical design while tightening three contracts before implementation: **convexity claims must be observation-model-specific, properness must be part of the family constraint interface, and the compact family should be introduced as a sibling numerical backend rather than by generalizing the existing polynomial internals in place.**
