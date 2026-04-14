# Proxy SVAR: Identification with External Instruments

This tutorial demonstrates how to identify structural shocks using external instruments (proxies), following the methodology of Stock & Watson (2012, 2018) and Mertens & Ravn (2013). We cover all available inference methods: Jentsch-Lunsford moving block bootstrap, Anderson-Rubin confidence sets, and Montiel Olea-Stock-Watson analytic confidence sets.

## Motivation

Cholesky identification imposes a recursive ordering on contemporaneous relationships. This is hard to justify in many applications. External instrument (proxy) identification offers an alternative: instead of restricting the causal order among endogenous variables, we use an external variable $Z_t$ that is correlated with one structural shock but uncorrelated with the others.

**Examples in macroeconomics**:
- Narrative monetary policy shocks as a proxy for the true monetary shock (Romer & Romer, Gertler & Karadi)
- Narrative tax shocks for fiscal policy identification (Mertens & Ravn)
- Oil supply disruptions for commodity market shocks

## Methodology

Consider the reduced-form VAR:

$$Y_t = A_1 Y_{t-1} + \cdots + A_p Y_{t-p} + u_t, \qquad u_t = H \varepsilon_t$$

where $\varepsilon_t$ are orthogonal structural shocks and $H$ is the structural impact matrix. We want to identify the first column $H_1$ (the impact of shock 1).

An external instrument $Z_t$ satisfies:
1. **Relevance**: $E[Z_t \varepsilon_{1,t}] \neq 0$
2. **Exogeneity**: $E[Z_t \varepsilon_{j,t}] = 0$ for $j \neq 1$

Under these conditions, $H_1$ is identified (up to scale) from the covariance of residuals with the proxy:

$$\Sigma_{um} = E[u_t Z_t], \qquad \phi = \sqrt{\Sigma_{um}^\top \Sigma_{uu}^{-1} \Sigma_{um}}, \qquad H_1 = \Sigma_{um} / \phi$$

---

## Setting Up the Model

### Simulated Data

We demonstrate using a bivariate VAR(2) with known structural parameters. This makes it easy to verify that the identification recovers the true impact.

```julia
using MacroEconometricTools
using LinearAlgebra, Statistics, Random

# Simulate a bivariate VAR(2) with known structural parameters
rng = Random.Xoshiro(42)
T = 200; K = 2; p = 2

# True structural impact matrix
H_true = [sqrt(2)/2  sqrt(2)/2;
          sqrt(2)/2 -sqrt(2)/2]

# True VAR lag coefficients
A1 = [0.6 0.1; 0.0 0.5]
A2 = [0.1 0.0; 0.0 0.1]

# Generate structural shocks
epsilon = randn(rng, T + p, K)

# Generate data from the VAR
Y = zeros(T + p, K)
for t in (p+1):(T+p)
    Y[t, :] .= A1 * Y[t-1, :] + A2 * Y[t-2, :] + H_true * epsilon[t, :]
end

# Generate proxy: correlated with shock 1, uncorrelated with shock 2
proxy = epsilon[:, 1] .+ 0.5 .* randn(rng, T + p)
```

### Creating the Instrument

The proxy should cover the same sample as the data Y. When you pass a proxy with $T$ rows (same as Y), the first $p$ rows are dropped automatically to align with the VAR residuals. There is no need to trim manually.

```julia
# Create instrument from full-length proxy: identify shock 1
instrument = ExternalInstrument(proxy, 1)
```

The first argument is the proxy (vector or matrix), the second is the index of the target shock. A vector is automatically reshaped to a $T \times 1$ matrix.

**Alternative**: `ProxyIV` provides an additional relevance threshold check:

```julia
# ProxyIV with relevance threshold (warns if F < threshold)
instrument = ProxyIV(proxy, 1; relevance_threshold=10.0)
```

### Fitting the IV-SVAR

The instrument is passed at estimation time. Internally, `fit` runs OLS on the VAR, then performs 2SLS on the residuals using the proxy.

```julia
model = fit(IVSVAR, Y, p;
    instrument = ExternalInstrument(proxy, 1),
    names = [:Y1, :Y2])
```

Output:
```
VARModel{Float64,IVSVAR{ExternalInstrument{Float64}}}
  Variables: Y1, Y2
  Observations: 198 (200 total)
  Lags: 2
```

---

## Diagnostics

### First-Stage F-Statistic

The first-stage F-statistic tests whether the proxy is sufficiently correlated with the target reduced-form residual. The conventional threshold is $F > 10$ (Stock, Wright & Yogo 2002).

```julia
F = first_stage_F(model)
println("First-stage F = $(round(F; digits=2))")
```

Expected output:
```
First-stage F = 85.23
```

A value well above 10 indicates a strong instrument. For a full summary:

```julia
iv_summary(model)
```

Output:
```
Proxy-SVAR Identification Summary
==================================================
Target shock: variable Y1 (index 1)
First-stage F-statistic: 85.23  ✓

Impact coefficients (unit effect normalization):
  Y1: 1.0000
  Y2: 0.9837
```

If $F < 10$, the instrument is considered weak. Standard bootstrap CIs may be unreliable, and you should use Anderson-Rubin or MSW confidence sets instead (see below).

---

## Computing Impulse Responses

### Point Estimates

```julia
result = irf(model, IVIdentification(); horizon=20)

println("IRF dimensions: ", size(result.irf))
# (21, 2, 2) = (horizons, variables, shocks)
```

Access specific responses:

```julia
# Response of Y2 to the identified shock (shock 1)
println("Y2 response to identified shock:")
for h in [0, 5, 10, 20]
    r = result.irf[h+1, 2, 1]
    println("  h=$h: $(round(r; digits=4))")
end
```

**Note**: Only the first shock column is identified by the proxy. The remaining columns use a Cholesky fill and should be interpreted with caution.

---

## Inference Methods for Proxy SVARs

Standard bootstrap methods (wild, i.i.d.) are not asymptotically valid for proxy-SVARs because the proxy must be jointly resampled with the VAR residuals. MacroEconometricTools.jl provides three inference methods specifically designed for proxy-SVARs.

### Method 1: Jentsch-Lunsford Moving Block Bootstrap

Jentsch & Lunsford (2022) propose a moving block bootstrap (MBB) that jointly resamples overlapping blocks of (residuals, proxy) with position-specific centering. This is the recommended method when the instrument is strong ($F \gg 10$).

#### High-Level Interface via `irf()`

```julia
result_mbb = irf(model, IVIdentification();
    horizon = 20,
    inference = ProxySVARMBB(2000; block_length=4),
    coverage = [0.68, 0.90, 0.95])

# Access confidence bands
lower_90 = result_mbb.lower[2]  # 90% lower bounds: (21, 2, 2)
upper_90 = result_mbb.upper[2]  # 90% upper bounds: (21, 2, 2)
```

#### Direct Interface for Full Control

The `proxy_svar_mbb()` function returns a richer result with percentile CIs, Hall's bias-corrected CIs, and FEVD CIs:

```julia
mbb = proxy_svar_mbb(model, 20,
    ProxySVARMBB(2000; block_length=4, norm_scale=-1.0);
    rng = Random.Xoshiro(123))
```

**Output fields**:

| Field | Shape | Description |
|-------|-------|-------------|
| `ci68_irf_norm` | (2, 21, 2) | 68% percentile CI for normalized IRFs |
| `ci95_irf_norm` | (2, 21, 2) | 95% percentile CI for normalized IRFs |
| `halls68_irf_norm` | (2, 21, 2) | 68% Hall's bias-corrected CI |
| `halls95_irf_norm` | (2, 21, 2) | 95% Hall's bias-corrected CI |
| `ci68_fevd` | (2, 21, 2) | 68% CI for forecast error variance decomposition |
| `ci95_fevd` | (2, 21, 2) | 95% CI for forecast error variance decomposition |
| `point_irf_norm` | (2, 21) | Point estimate of normalized IRFs |
| `point_svma` | (2, 21) | Structural VMA coefficients |
| `n_failed` | Int | Number of failed bootstrap replications |

The first dimension of each CI array is `[lower, upper]`.

#### Accessing Results

```julia
# Normalized IRF point estimate at horizon h for variable k
point_h5_Y2 = mbb.point_irf_norm[2, 6]  # k=2, h=5 (1-indexed)

# 95% percentile CI
lower = mbb.ci95_irf_norm[1, 6, 2]
upper = mbb.ci95_irf_norm[2, 6, 2]
println("Y2 at h=5: $(round(point_h5_Y2; digits=4)) [$(round(lower; digits=4)), $(round(upper; digits=4))]")
```

#### Percentile vs. Hall's Intervals

- **Percentile intervals** use the empirical quantiles of bootstrap draws directly.
- **Hall's bias-corrected intervals** apply the transformation: $[\hat\theta - (q_{1-\alpha/2} - \hat\theta), \; \hat\theta - (q_{\alpha/2} - \hat\theta)]$, or equivalently, $\text{lower} = 2\hat\theta - q_{1-\alpha/2}$ and $\text{upper} = 2\hat\theta - q_{\alpha/2}$. This corrects for bootstrap bias.

In practice, Hall's intervals often give similar results to percentile intervals when the bootstrap distribution is approximately symmetric.

#### Choosing Block Length

The block length $\ell$ controls the trade-off between capturing serial dependence and having enough unique blocks. A common rule of thumb is $\ell \approx T^{1/3}$. For monthly macro data with $T = 200$, this gives $\ell \approx 6$. Jentsch & Lunsford (2022) use $\ell = 4$ in their applications.

```julia
# Try different block lengths
for bl in [3, 4, 6, 8]
    m = proxy_svar_mbb(model, 20, ProxySVARMBB(1000; block_length=bl);
        rng = Random.Xoshiro(42))
    w95 = m.ci95_irf_norm[2, 1, 2] - m.ci95_irf_norm[1, 1, 2]  # Y2 impact width
    println("  block_length=$bl: 95% CI width at impact = $(round(w95; digits=4))")
end
```

---

### Method 2: Anderson-Rubin Confidence Sets

Anderson-Rubin (AR) confidence sets are robust to weak instruments. They are constructed by inverting a test statistic over a grid of hypothesized IRF values. Even when $F < 10$, AR sets maintain correct coverage.

```julia
ar_grid = collect(range(-3.0, 3.0; length=601))

mbb_ar = proxy_svar_mbb(model, 20,
    ProxySVARMBB(2000;
        block_length = 4,
        compute_ar = true,
        ar_grid = ar_grid,
        norm_scale = -1.0);
    rng = Random.Xoshiro(456))
```

The AR results are in `mbb_ar.ar`:

| Field | Shape | Description |
|-------|-------|-------------|
| `index68` | (601, 21, 2) | Boolean: grid point in 68% set |
| `index90` | (601, 21, 2) | Boolean: grid point in 90% set |
| `index95` | (601, 21, 2) | Boolean: grid point in 95% set |
| `grid` | (601,) | The grid values |
| `rates` | (2, 21, 601) | Rejection rates at each grid point |

#### Extracting Confidence Sets

```julia
# 95% AR confidence set for Y2 at horizon 5
h = 6  # 1-indexed
k = 2  # Y2
included = mbb_ar.ar.index95[:, h, k]
if any(included)
    grid_vals = ar_grid[included]
    println("95% AR set for Y2 at h=5: [$(round(minimum(grid_vals); digits=3)), $(round(maximum(grid_vals); digits=3))]")
else
    println("95% AR set for Y2 at h=5: empty")
end
```

**Interpretation**: AR sets are nested: the 68% set is contained in the 90% set, which is contained in the 95% set. If the instrument is weak, AR sets may be very wide or even unbounded (covering the entire grid). This is a feature, not a bug: it correctly reflects the large uncertainty when identification is weak.

#### When to Use AR Sets

- Always appropriate, but most valuable when the F-statistic is below or near 10
- Computationally more expensive than percentile CIs (requires grid search)
- Finer grid gives more precise set boundaries but takes longer

---

### Method 3: MSW Confidence Sets

Montiel Olea, Stock & Watson (2021) provide analytic confidence sets that are robust to weak instruments. No bootstrap is required, making them very fast to compute.

```julia
msw = msw_confidence_set(model; norm_scale=-1.0, horizon=20)
```

**Output fields**:

| Field | Type | Description |
|-------|------|-------------|
| `cs68_irf_norm` | Array | 68% confidence set for normalized IRFs |
| `cs95_irf_norm` | Array | 95% confidence set for normalized IRFs |
| `vcv_matrix` | Matrix | Full variance-covariance matrix |
| `wald_stat` | Float64 | Wald statistic for instrument strength |
| `bounded68` | Bool | Whether the 68% set is bounded |
| `bounded95` | Bool | Whether the 95% set is bounded |

#### Bounded vs. Unbounded Sets

When the instrument is strong, confidence sets are bounded intervals (shape `(2, n_imp, K)`):

```julia
if msw.bounded95
    # Standard bounded interval
    lower = msw.cs95_irf_norm[1, 6, 2]  # Y2 at h=5
    upper = msw.cs95_irf_norm[2, 6, 2]
    println("95% MSW set for Y2 at h=5: [$lower, $upper]")
end
```

When the instrument is weak, the confidence set becomes unbounded, consisting of two disjoint intervals (shape `(4, n_imp, K)`):

```julia
if !msw.bounded95
    # Unbounded: two disjoint intervals
    # (-Inf, cs[2,...]) ∪ (cs[3,...], +Inf)
    println("95% MSW set: (-Inf, $(msw.cs95_irf_norm[2,h,k])] ∪ [$(msw.cs95_irf_norm[3,h,k]), +Inf)")
end
```

#### Wald Statistic

The Wald statistic tests the null that $\gamma_1 = 0$ (the proxy is irrelevant). It is analogous to the first-stage F-statistic but uses heteroskedasticity-robust standard errors:

```julia
println("MSW Wald statistic: $(round(msw.wald_stat; digits=2))")
println("68% set bounded: $(msw.bounded68)")
println("95% set bounded: $(msw.bounded95)")
```

---

## Comparing Inference Methods

| Method | Valid with weak IV? | Requires bootstrap? | Computational cost | Output |
|--------|:------------------:|:--------------------:|:-----------------:|--------|
| **Percentile CI** (MBB) | No | Yes | Moderate | Intervals |
| **Hall's CI** (MBB) | No | Yes | Moderate | Intervals |
| **Anderson-Rubin** | Yes | Yes | High (grid) | Confidence sets |
| **MSW** | Yes | No | Very low | Intervals / disjoint |

**Recommendations**:

- **Strong instrument** ($F > 20$): Use ProxySVARMBB percentile or Hall's CIs. They are the most straightforward and well-powered.
- **Moderate instrument** ($10 < F < 20$): Compare MBB percentile CIs with AR and MSW sets. If they diverge substantially, the standard CIs may be unreliable.
- **Weak instrument** ($F < 10$): Use AR or MSW only. Percentile CIs have incorrect coverage under weak identification.

### Side-by-Side Comparison

```julia
# All three methods on the same model
mbb = proxy_svar_mbb(model, 20,
    ProxySVARMBB(2000; block_length=4, compute_ar=true,
        ar_grid=collect(range(-3.0, 3.0; length=301)), norm_scale=-1.0);
    rng=Random.Xoshiro(42))
msw = msw_confidence_set(model; norm_scale=-1.0, horizon=20)

h = 6; k = 2  # Y2 at horizon 5
println("Y2 at h=5:")
println("  Percentile 95%: [$(round(mbb.ci95_irf_norm[1,h,k]; digits=3)), $(round(mbb.ci95_irf_norm[2,h,k]; digits=3))]")
println("  Hall's 95%:     [$(round(mbb.halls95_irf_norm[1,h,k]; digits=3)), $(round(mbb.halls95_irf_norm[2,h,k]; digits=3))]")

ar_in = mbb.ar.index95[:, h, k]
grid = mbb.ar.grid
if any(ar_in)
    println("  AR 95%:         [$(round(minimum(grid[ar_in]); digits=3)), $(round(maximum(grid[ar_in]); digits=3))]")
end

if msw.bounded95
    println("  MSW 95%:        [$(round(msw.cs95_irf_norm[1,h,k]; digits=3)), $(round(msw.cs95_irf_norm[2,h,k]; digits=3))]")
end
```

---

## Advanced Topics

### FEVD Confidence Intervals

The MBB also produces bootstrap CIs for the forecast error variance decomposition:

```julia
mbb = proxy_svar_mbb(model, 20,
    ProxySVARMBB(2000; block_length=4, norm_scale=-1.0);
    rng=Random.Xoshiro(42))

# FEVD point estimate and 95% CI for Y2 at horizon 10
h = 11  # 1-indexed
k = 2
println("FEVD of Y2 at h=10:")
println("  Point: $(round(mbb.point_fevd[k, h]; digits=3))")
println("  95% CI: [$(round(mbb.ci95_fevd[1,h,k]; digits=3)), $(round(mbb.ci95_fevd[2,h,k]; digits=3))]")
```

### Normalization

The `norm_scale` parameter controls how normalized IRFs are scaled. The default `norm_scale = -1.0` means a one-unit negative shock: the impact response of the target variable equals $-1$.

```julia
# Negative one-unit shock (default)
mbb_neg = proxy_svar_mbb(model, 20, ProxySVARMBB(500; norm_scale=-1.0);
    rng=Random.Xoshiro(1))

# Positive one-unit shock
mbb_pos = proxy_svar_mbb(model, 20, ProxySVARMBB(500; norm_scale=1.0);
    rng=Random.Xoshiro(1))

# They are mirror images
@assert mbb_neg.point_irf_norm ≈ -mbb_pos.point_irf_norm
```

### Multiple Instruments

`ExternalInstrument` accepts a matrix of instruments ($T \times k_z$):

```julia
Z_matrix = hcat(proxy1, proxy2)  # Two instruments for one shock
instrument = ExternalInstrument(Z_matrix, 1; method=:tsls)
```

The `method` keyword controls the estimation approach:
- `:tsls` (default) — two-stage least squares
- `:liml` — limited information maximum likelihood
- `:fuller` — Fuller's modified LIML

---

## References

- Anderson, T. W., & Rubin, H. (1949). Estimation of the Parameters of a Single Equation in a Complete System of Stochastic Equations. *Annals of Mathematical Statistics*, 20(1), 46-63.

- Jentsch, C., & Lunsford, K. G. (2022). Asymptotically Valid Bootstrap Inference for Proxy SVARs. *Journal of Business & Economic Statistics*, 40(4), 1876-1891.

- Mertens, K., & Ravn, M. O. (2013). The Dynamic Effects of Personal and Corporate Income Tax Changes in the United States. *American Economic Review*, 103(4), 1212-1247.

- Montiel Olea, J. L., Stock, J. H., & Watson, M. W. (2021). Inference in Structural Vector Autoregressions Identified with an External Instrument. *Journal of Econometrics*, 225(1), 74-87.

- Stock, J. H., & Watson, M. W. (2012). Disentangling the Channels of the 2007-2009 Recession. *Brookings Papers on Economic Activity*, Spring, 81-135.

- Stock, J. H., & Watson, M. W. (2018). Identification and Estimation of Dynamic Causal Effects in Macroeconomics Using External Instruments. *Economic Journal*, 128(610), 917-948.

- Stock, J. H., Wright, J. H., & Yogo, M. (2002). A Survey of Weak Instruments and Weak Identification in Generalized Method of Moments. *Journal of Business & Economic Statistics*, 20(4), 518-529.
