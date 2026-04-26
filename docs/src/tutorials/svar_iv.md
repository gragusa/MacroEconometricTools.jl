# SVAR-IV: Identification with External Instruments

This tutorial demonstrates how to identify structural shocks using external
instruments (proxies), following the methodology of Stock & Watson (2012, 2018)
and Mertens & Ravn (2013). We cover all available inference methods:
Jentsch–Lunsford moving block bootstrap, Anderson–Rubin confidence sets, and
Montiel Olea–Stock–Watson analytic confidence sets.

## Motivation

Cholesky identification imposes a recursive ordering on contemporaneous
relationships, which is hard to justify in many applications. External
instrument (proxy) identification offers an alternative: instead of restricting
the causal order among endogenous variables, we use an external variable $Z_t$
that is correlated with one structural shock but uncorrelated with the others.

**Examples in macroeconomics**:

- Narrative monetary policy shocks as a proxy for the true monetary shock
  (Romer & Romer, Gertler & Karadi)
- Narrative tax shocks for fiscal policy identification (Mertens & Ravn)
- Oil supply disruptions for commodity market shocks

## Methodology

Consider the reduced-form VAR:

$$Y_t = A_1 Y_{t-1} + \cdots + A_p Y_{t-p} + u_t, \qquad u_t = H \varepsilon_t$$

where $\varepsilon_t$ are orthogonal structural shocks and $H$ is the
structural impact matrix. We want to identify the column of $H$ corresponding
to a target shock (say shock 1, $H_1$).

An external instrument $Z_t$ satisfies:

1. **Relevance**: $E[Z_t \varepsilon_{1,t}] \neq 0$
2. **Exogeneity**: $E[Z_t \varepsilon_{j,t}] = 0$ for $j \neq 1$

Under these conditions, $H_1$ is identified (up to scale) from the covariance
of residuals with the proxy:

$$\Sigma_{um} = E[u_t Z_t], \qquad \phi = \sqrt{\Sigma_{um}^\top \Sigma_{uu}^{-1} \Sigma_{um}}, \qquad H_1 = \Sigma_{um} / \phi$$

---

## Setting Up the Model

### Loading the Example Data

We use the bivariate VAR(2) cross-validation dataset shipped with the package
under `test/data/`. It is a 200-observation simulation from a known DGP with a
moderately strong proxy ($F \approx 45$).

```julia
using MacroEconometricTools
using LinearAlgebra, Statistics, Random
using CSV, DataFrames

datadir = joinpath(pkgdir(MacroEconometricTools), "test", "data")
Y = Matrix(CSV.read(joinpath(datadir, "jl_crossval_Y.csv"), DataFrame))
proxy = vec(Matrix(CSV.read(joinpath(datadir, "jl_crossval_proxy.csv"), DataFrame)))

size(Y), length(proxy)
# ((200, 2), 200)
```

The dataset has two endogenous variables (`Y1`, `Y2`) and one external
instrument that targets the first structural shock.

### Estimating the Reduced-Form VAR

The current API separates estimation from identification: estimate a
reduced-form VAR with `OLSVAR`, then attach the instrument to the
identification scheme.

```julia
p = 2
model = fit(OLSVAR, Y, p; names = [:Y1, :Y2])
```

Output:

```
VARModel{Float64, OLSVAR}
  Variables: Y1, Y2
  Observations: 198 (200 total)
  Lags: 2
```

### Building the Identification

Wrap the proxy in an `IVIdentification`. The proxy can have the same number of
rows as `Y` (the first `p` rows are dropped automatically to align with the VAR
residuals); a vector is auto-reshaped to `T×1`.

```julia
# Identify shock 1 with the external instrument
id = IVIdentification(proxy, 1)
```

The second argument is the target shock index. You can also pass it by name
when the model has named variables:

```julia
id = IVIdentification(ExternalInstrument(proxy; target_shock = :Y1))
```

Or use `ProxyIV`, which adds a relevance threshold check:

```julia
id = IVIdentification(ProxyIV(proxy; target_shock = 1, relevance_threshold = 10.0))
```

!!! note "Old API still works"
    The legacy form `fit(IVSVAR, Y, p; instrument = ExternalInstrument(proxy, 1))`
    followed by `irf(model, IVIdentification(); ...)` is supported for
    backwards compatibility. New code should prefer the
    `OLSVAR` + `IVIdentification(Z, target)` form shown above.

---

## Diagnostics

### First-Stage F-Statistic

The first-stage F-statistic tests whether the proxy is sufficiently correlated
with the target reduced-form residual. The conventional threshold is $F > 10$
(Stock, Wright & Yogo, 2002).

```julia
F = first_stage_F(model, id)
println("First-stage F = ", round(F; digits = 2))
# First-stage F = 44.9
```

A value above 10 indicates a strong instrument. For a full summary:

```julia
iv_summary(model, id)
```

```
SVAR-IV Identification Summary
==================================================
Target shock: variable Y1 (index 1)
First-stage F-statistic: 44.9  ✓

Impact coefficients (unit effect normalization):
  Y1: 1.0
  Y2: -0.394
```

If $F < 10$, the instrument is considered weak. Standard bootstrap CIs may be
unreliable, and you should use Anderson–Rubin or MSW confidence sets instead
(see below).

---

## Computing Impulse Responses

### Point Estimates

```julia
result = irf(model, id; horizon = 20)
```

The result wraps the IRF in an `AxisArray` with axes
`(:variable, :shock, :horizon)`:

```julia
size(result.irf)        # (2, 2, 21) = (n_vars, n_shocks, horizon+1)
varnames(result)        # [:Y1, :Y2]
```

Access responses by symbol or by integer index:

```julia
# Response of Y2 to the identified shock (shock 1)
result.irf[:Y2, :Y1, 0:5]   # AxisArray slice over the first six horizons

# Equivalently, by integer indices
println("Y2 response to identified shock (shock 1):")
for h in (0, 5, 10, 20)
    println("  h=$h: ", round(result.irf[2, 1, h + 1]; digits = 4))
end
```

!!! note "Identified column only"
    Only the column of `H` corresponding to the target shock is identified by
    the proxy. The remaining columns are filled via a Cholesky-based
    decomposition of the orthogonal complement and should not be interpreted
    structurally.

### Bootstrap Bands via `irf()`

To get confidence bands directly from `irf`, pass an `inference` argument and
choose coverage levels:

```julia
result_mbb = irf(model, id;
    horizon = 20,
    inference = ProxySVARMBB(2000; block_length = 4),
    coverage = [0.68, 0.90, 0.95],
    rng = Random.Xoshiro(123))

# coverage levels are sorted ascending; lower[2]/upper[2] are the 90% bands
lower_90 = result_mbb.lower[2]   # AxisArray (variable, shock, horizon)
upper_90 = result_mbb.upper[2]
```

The wrapped bands inherit the same axes as `result_mbb.irf`, so you can index
them by variable/shock symbols.

---

## Inference Methods for SVAR-IV

Standard bootstrap methods (i.i.d., wild) are not asymptotically valid for
proxy-SVARs because the proxy must be jointly resampled with the VAR
residuals. MacroEconometricTools.jl provides three inference methods designed
specifically for SVAR-IV.

### Method 1: Jentsch–Lunsford Moving Block Bootstrap

Jentsch & Lunsford (2022) propose a moving block bootstrap (MBB) that jointly
resamples overlapping blocks of (residuals, proxy) with position-specific
centering. This is the recommended method when the instrument is strong
($F \gg 10$).

#### Through `irf()`

```julia
result_mbb = irf(model, id;
    horizon = 20,
    inference = ProxySVARMBB(2000; block_length = 4),
    coverage = [0.68, 0.90, 0.95],
    rng = Random.Xoshiro(123))
```

#### Direct Interface for Full Control

`proxy_svar_mbb` returns a richer result with percentile CIs, Hall's
bias-corrected CIs, and FEVD CIs:

```julia
mbb = proxy_svar_mbb(model, id, 20,
    ProxySVARMBB(2000; block_length = 4, norm_scale = -1.0);
    rng = Random.Xoshiro(123))
```

#### Returned Fields

The named tuple has the following entries (axes
`(K, n_imp)` for point matrices; `(2, n_imp, K)` for CIs, where the first
dimension is `[lower, upper]`):

| Field | Shape | Description |
|-------|-------|-------------|
| `point_irf` | `(K, n_imp)` | Point estimate of the identified IRF column |
| `point_irf_norm` | `(K, n_imp)` | Normalized point estimate (target hits `norm_scale` on impact) |
| `point_fevd` | `(K, n_imp)` | Forecast error variance contributions of the identified shock |
| `point_svma` | `(K, n_imp)` | Structural VMA coefficients |
| `ci68_irf` / `ci95_irf` | `(2, n_imp, K)` | Percentile CIs for the unnormalized IRF |
| `ci68_irf_norm` / `ci95_irf_norm` | `(2, n_imp, K)` | Percentile CIs for the normalized IRF |
| `halls68_irf_norm` / `halls95_irf_norm` | `(2, n_imp, K)` | Hall's bias-corrected CIs |
| `ci68_fevd` / `ci95_fevd` | `(2, n_imp, K)` | Percentile CIs for FEVD |
| `ar` | named tuple or `nothing` | Anderson–Rubin output (see below) |
| `irf_store` | `(K, n_imp, n_boot)` | Raw bootstrap draws (unnormalized) |
| `n_failed` | `Int` | Number of failed bootstrap replications |

#### Accessing Results

```julia
# Normalized IRF point estimate at horizon 5 for Y2
h_idx = 6   # 1-indexed: h=0 is column 1
k_idx = 2   # Y2
point_h5_Y2 = mbb.point_irf_norm[k_idx, h_idx]

# 95% percentile CI
lower = mbb.ci95_irf_norm[1, h_idx, k_idx]
upper = mbb.ci95_irf_norm[2, h_idx, k_idx]
println("Y2 at h=5: ", round(point_h5_Y2; digits = 4),
    " [", round(lower; digits = 4), ", ", round(upper; digits = 4), "]")
```

#### Percentile vs. Hall's Intervals

- **Percentile intervals** use the empirical quantiles of the bootstrap
  distribution directly.
- **Hall's bias-corrected intervals** apply
  $\text{lower} = 2\hat\theta - q_{1-\alpha/2}$,
  $\text{upper} = 2\hat\theta - q_{\alpha/2}$, which corrects for bootstrap
  bias.

In practice the two coincide when the bootstrap distribution is approximately
symmetric.

#### Choosing the Block Length

The block length $\ell$ controls the trade-off between capturing serial
dependence and keeping enough unique blocks. A common rule of thumb is
$\ell \approx T^{1/3}$. For $T = 200$ this gives $\ell \approx 6$;
Jentsch & Lunsford (2022) use $\ell = 4$ in their applications.

```julia
for bl in (3, 4, 6, 8)
    m = proxy_svar_mbb(model, id, 20,
        ProxySVARMBB(1000; block_length = bl, norm_scale = -1.0);
        rng = Random.Xoshiro(42))
    w95 = m.ci95_irf_norm[2, 1, 2] - m.ci95_irf_norm[1, 1, 2]
    println("  block_length=$bl: 95% CI width at impact (Y2) = ",
        round(w95; digits = 4))
end
```

---

### Method 2: Anderson–Rubin Confidence Sets

Anderson–Rubin (AR) confidence sets are robust to weak instruments. They are
constructed by inverting a test statistic over a grid of hypothesized IRF
values. Even when $F < 10$, AR sets maintain correct coverage.

```julia
ar_grid = collect(range(-3.0, 3.0; length = 601))

mbb_ar = proxy_svar_mbb(model, id, 20,
    ProxySVARMBB(2000;
        block_length = 4,
        compute_ar = true,
        ar_grid = ar_grid,
        norm_scale = -1.0);
    rng = Random.Xoshiro(456))
```

The AR results are stored in `mbb_ar.ar`:

| Field | Shape | Description |
|-------|-------|-------------|
| `index68` | `(n_grid, n_imp, K)` | Boolean: grid point in 68% set |
| `index90` | `(n_grid, n_imp, K)` | Boolean: grid point in 90% set |
| `index95` | `(n_grid, n_imp, K)` | Boolean: grid point in 95% set |
| `grid` | `(n_grid,)` | The grid values |
| `rates` | `(K, n_imp, n_grid)` | Rejection rates at each grid point |

If `compute_ar = true` is set without supplying `ar_grid`, a default grid of
`range(-10, 10; length = 201)` is used.

#### Extracting Confidence Sets

```julia
# 95% AR confidence set for Y2 at horizon 5
h_idx = 6
k_idx = 2
included = mbb_ar.ar.index95[:, h_idx, k_idx]
if any(included)
    grid_vals = ar_grid[included]
    println("95% AR set for Y2 at h=5: [",
        round(minimum(grid_vals); digits = 3), ", ",
        round(maximum(grid_vals); digits = 3), "]")
else
    println("95% AR set for Y2 at h=5: empty")
end
```

**Interpretation**: AR sets are nested — the 68% set is contained in the 90%
set, which is contained in the 95% set. If the instrument is weak, AR sets may
be very wide or even unbounded; this correctly reflects the large uncertainty
under weak identification.

#### When to Use AR Sets

- Always appropriate, but most valuable when the F-statistic is below or near
  10.
- Computationally more expensive than percentile CIs (requires a grid search).
- Finer grids give more precise set boundaries but take longer to evaluate.

When `compute_ar = true` is passed through `irf()`, the returned `lower`/
`upper` bands are the convex hull `[min, max]` of included grid points
(conservative but plottable). The full boolean masks are still available via
`proxy_svar_mbb` for more detailed analysis.

---

### Method 3: MSW Confidence Sets

Montiel Olea, Stock & Watson (2021) provide analytic confidence sets that are
robust to weak instruments. No bootstrap is required, making them very fast to
compute.

```julia
msw = msw_confidence_set(model, id; norm_scale = -1.0, horizon = 20)
```

#### Returned Fields

| Field | Type | Description |
|-------|------|-------------|
| `cs68_irf_norm` | Array | 68% confidence set for normalized IRFs |
| `cs95_irf_norm` | Array | 95% confidence set for normalized IRFs |
| `vcv_matrix` | Matrix | Full variance–covariance matrix |
| `wald_stat` | Float64 | Wald statistic for instrument strength |
| `bounded68` | Bool | Whether the 68% set is bounded |
| `bounded95` | Bool | Whether the 95% set is bounded |

#### Bounded vs. Unbounded Sets

When the instrument is strong, confidence sets are bounded intervals with
shape `(2, n_imp, K)`:

```julia
if msw.bounded95
    h_idx, k_idx = 6, 2   # Y2 at h=5
    lower = msw.cs95_irf_norm[1, h_idx, k_idx]
    upper = msw.cs95_irf_norm[2, h_idx, k_idx]
    println("95% MSW set for Y2 at h=5: [", lower, ", ", upper, "]")
end
```

When the instrument is weak, the confidence set becomes unbounded with shape
`(4, n_imp, K)`, encoding two disjoint intervals
$(-\infty,\ \texttt{cs[2]}] \cup [\texttt{cs[3]},\ +\infty)$:

```julia
if !msw.bounded95
    h_idx, k_idx = 6, 2
    println("95% MSW set: (-Inf, ", msw.cs95_irf_norm[2, h_idx, k_idx],
        "] ∪ [", msw.cs95_irf_norm[3, h_idx, k_idx], ", +Inf)")
end
```

#### Wald Statistic

The Wald statistic tests the null that the proxy is irrelevant. It is
analogous to the first-stage F-statistic but uses heteroskedasticity-robust
standard errors:

```julia
println("MSW Wald statistic: ", round(msw.wald_stat; digits = 2))
println("68% set bounded: ", msw.bounded68)
println("95% set bounded: ", msw.bounded95)
```

---

## Comparing Inference Methods

| Method | Valid with weak IV? | Requires bootstrap? | Computational cost | Output |
|--------|:-------------------:|:-------------------:|:------------------:|--------|
| **Percentile CI** (MBB) | No  | Yes | Moderate    | Intervals |
| **Hall's CI** (MBB)     | No  | Yes | Moderate    | Intervals |
| **Anderson–Rubin**      | Yes | Yes | High (grid) | Confidence sets |
| **MSW**                 | Yes | No  | Very low    | Intervals / disjoint |

**Recommendations**:

- **Strong instrument** ($F > 20$): use `ProxySVARMBB` percentile or Hall's
  CIs.
- **Moderate instrument** ($10 < F < 20$): compare MBB percentile CIs with AR
  and MSW sets. Substantial divergence is a sign that the standard CIs may be
  unreliable.
- **Weak instrument** ($F < 10$): use AR or MSW only. Percentile CIs have
  incorrect coverage under weak identification.

### Side-by-Side Comparison

```julia
mbb = proxy_svar_mbb(model, id, 20,
    ProxySVARMBB(2000; block_length = 4, compute_ar = true,
        ar_grid = collect(range(-3.0, 3.0; length = 301)),
        norm_scale = -1.0);
    rng = Random.Xoshiro(42))
msw = msw_confidence_set(model, id; norm_scale = -1.0, horizon = 20)

h_idx, k_idx = 6, 2   # Y2 at h=5
println("Y2 at h=5:")
println("  Percentile 95%: [",
    round(mbb.ci95_irf_norm[1, h_idx, k_idx]; digits = 3), ", ",
    round(mbb.ci95_irf_norm[2, h_idx, k_idx]; digits = 3), "]")
println("  Hall's 95%:     [",
    round(mbb.halls95_irf_norm[1, h_idx, k_idx]; digits = 3), ", ",
    round(mbb.halls95_irf_norm[2, h_idx, k_idx]; digits = 3), "]")

ar_in = mbb.ar.index95[:, h_idx, k_idx]
grid = mbb.ar.grid
if any(ar_in)
    println("  AR 95%:         [",
        round(minimum(grid[ar_in]); digits = 3), ", ",
        round(maximum(grid[ar_in]); digits = 3), "]")
end

if msw.bounded95
    println("  MSW 95%:        [",
        round(msw.cs95_irf_norm[1, h_idx, k_idx]; digits = 3), ", ",
        round(msw.cs95_irf_norm[2, h_idx, k_idx]; digits = 3), "]")
end
```

---

## Advanced Topics

### FEVD Confidence Intervals

The MBB also produces bootstrap CIs for the forecast error variance
decomposition:

```julia
mbb = proxy_svar_mbb(model, id, 20,
    ProxySVARMBB(2000; block_length = 4, norm_scale = -1.0);
    rng = Random.Xoshiro(42))

# FEVD point estimate and 95% CI for Y2 at horizon 10
h_idx = 11   # 1-indexed
k_idx = 2
println("FEVD of Y2 at h=10:")
println("  Point: ", round(mbb.point_fevd[k_idx, h_idx]; digits = 3))
println("  95% CI: [",
    round(mbb.ci95_fevd[1, h_idx, k_idx]; digits = 3), ", ",
    round(mbb.ci95_fevd[2, h_idx, k_idx]; digits = 3), "]")
```

### Normalization

The `norm_scale` parameter controls how normalized IRFs are scaled. The
default `norm_scale = -1.0` means a one-unit *negative* shock: the impact
response of the target variable equals $-1$.

```julia
mbb_neg = proxy_svar_mbb(model, id, 20,
    ProxySVARMBB(500; norm_scale = -1.0);
    rng = Random.Xoshiro(1))
mbb_pos = proxy_svar_mbb(model, id, 20,
    ProxySVARMBB(500; norm_scale =  1.0);
    rng = Random.Xoshiro(1))

# Normalized point estimates are mirror images
@assert mbb_neg.point_irf_norm ≈ -mbb_pos.point_irf_norm
```

For the high-level `irf()` interface, scaling is controlled by the
`normalization` and `scale` keyword arguments rather than `norm_scale`. See
the [Getting Started](getting_started.md) tutorial for examples.

### Multiple Instruments

`ExternalInstrument` accepts a matrix of instruments ($T \times k_z$):

```julia
Z_matrix = hcat(proxy1, proxy2)   # Two instruments for one shock
id = IVIdentification(ExternalInstrument(Z_matrix; target_shock = 1, method = :tsls))
```

The `method` keyword controls the estimation approach:

- `:tsls` (default) — two-stage least squares
- `:liml` — limited information maximum likelihood
- `:fuller` — Fuller's modified LIML

---

## References

- Anderson, T. W., & Rubin, H. (1949). Estimation of the parameters of a
  single equation in a complete system of stochastic equations.
  *Annals of Mathematical Statistics*, 20(1), 46–63.

- Jentsch, C., & Lunsford, K. G. (2022). Asymptotically valid bootstrap
  inference for proxy SVARs. *Journal of Business & Economic Statistics*,
  40(4), 1876–1891.

- Mertens, K., & Ravn, M. O. (2013). The dynamic effects of personal and
  corporate income tax changes in the United States. *American Economic
  Review*, 103(4), 1212–1247.

- Montiel Olea, J. L., Stock, J. H., & Watson, M. W. (2021). Inference in
  structural vector autoregressions identified with an external instrument.
  *Journal of Econometrics*, 225(1), 74–87.

- Stock, J. H., & Watson, M. W. (2012). Disentangling the channels of the
  2007–2009 recession. *Brookings Papers on Economic Activity*, Spring,
  81–135.

- Stock, J. H., & Watson, M. W. (2018). Identification and estimation of
  dynamic causal effects in macroeconomics using external instruments.
  *Economic Journal*, 128(610), 917–948.

- Stock, J. H., Wright, J. H., & Yogo, M. (2002). A survey of weak instruments
  and weak identification in generalized method of moments. *Journal of
  Business & Economic Statistics*, 20(4), 518–529.
