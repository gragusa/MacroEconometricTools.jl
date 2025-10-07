# Quick Reference Guide

Fast lookup for common tasks in MacroEconometricTools.jl.

## Installation

```julia
using Pkg
Pkg.add("MacroEconometricTools")
```

## Basic Workflow

```julia
using MacroEconometricTools

# 1. Estimate VAR
var = estimate(OLSVAR, Y, p; names=varnames)

# 2. Identify structural shocks
P = identify(var, CholeskyID())

# 3. Compute IRFs
irf_result = irf(var, CholeskyID(); horizon=24, bootstrap_reps=1000)
```

---

## VAR Estimation

### Basic OLS VAR

```julia
# Matrix Y: T × n_vars
# p: number of lags
var = estimate(OLSVAR, Y, p)
```

### With variable names

```julia
names = [:gdp, :inflation, :interest_rate]
var = estimate(OLSVAR, Y, p; names=names)
```

### With demeaning

```julia
var = estimate(OLSVAR, Y, p; demean=true)
```

---

## Constraints

### Zero constraint (one variable, one lag)

```julia
# Interest rate doesn't affect GDP at lag 1
c = ZeroConstraint(:gdp, [:interest_rate], [1])
var = estimate(OLSVAR, Y, p; constraints=[c])
```

### Zero constraint (all lags)

```julia
# Interest rate doesn't affect GDP at any lag
c = ZeroConstraint(:gdp, [:interest_rate], Int[])  # empty = all lags
var = estimate(OLSVAR, Y, p; constraints=[c])
```

### Block exogeneity

```julia
# Foreign variables don't respond to domestic
c = BlockExogeneity([:domestic_gdp, :domestic_rate], [:foreign_gdp, :foreign_rate])
var = estimate(OLSVAR, Y, p; constraints=[c])
```

### Fixed value constraint

```julia
# Fix GDP persistence to 0.9
c = FixedConstraint(:gdp, :gdp, 1, 0.9)  # (equation, regressor, lag, value)
var = estimate(OLSVAR, Y, p; constraints=[c])
```

### Multiple constraints

```julia
constraints = [
    ZeroConstraint(:var1, [:var2, :var3], [1, 2]),
    FixedConstraint(:var2, :var1, 1, 0.5),
    BlockExogeneity([:foreign], [:domestic])
]
var = estimate(OLSVAR, Y, p; constraints=constraints)
```

---

## Identification

### Cholesky (default ordering)

```julia
id = CholeskyID()
P = identify(var, id)
```

### Cholesky (custom ordering)

```julia
id = CholeskyID(ordering=[:oil_price, :production, :activity])
P = identify(var, id)
```

### Sign restrictions

```julia
# Rows = variables, Columns = shocks
# +1 = positive, -1 = negative, 0 = unrestricted
restrictions = [
    -1   1   0;
     0   1   0;
     1   1   1
]

id = SignRestriction(restrictions, 12)  # apply for horizons 0-12
P = identify(var, id; max_draws=10000)
```

---

## Impulse Response Functions

### Basic IRF (no inference)

```julia
irf_result = irf(var, id; horizon=24, inference=:none)

# Access point estimates
point_irf = irf_result.irf  # (horizon+1, n_vars, n_shocks)
```

### IRF with bootstrap

```julia
irf_result = irf(var, id;
                 horizon=24,
                 inference=:bootstrap,
                 bootstrap_reps=1000,
                 coverage=[0.68, 0.90, 0.95])

# Point estimate
point = irf_result.irf[h+1, response_var, shock_var]

# Confidence bands
lower_90 = irf_result.lower[2][h+1, response_var, shock_var]  # 90% = index 2
upper_90 = irf_result.upper[2][h+1, response_var, shock_var]
```

### Bootstrap methods

```julia
# Wild bootstrap (default, heteroskedasticity-robust)
irf_result = irf(var, id; bootstrap_method=:wild)

# Standard bootstrap (iid errors)
irf_result = irf(var, id; bootstrap_method=:standard)

# Block bootstrap (time series dependence)
irf_result = irf(var, id; bootstrap_method=:block, block_length=4)
```

### Cumulative IRF

```julia
cirf = cumulative_irf(irf_result)
```

---

## Variance Decomposition

```julia
fevd = variance_decomposition(irf_result)

# fevd[horizon, variable, shock] = contribution (between 0 and 1)
# Sum over shocks = 1 for each variable and horizon
```

---

## Historical Decomposition

```julia
hd = historical_decomposition(var, id)

# hd[time, variable, shock] = contribution of shock to variable at time
```

---

## Forecasting

```julia
fc = forecast(var, h;  # h = forecast horizon
              coverage=[0.68, 0.90],
              bootstrap_reps=1000)

# Point forecast
fc.mean  # (h, n_vars)

# Confidence intervals
fc.lower[1]  # 68% lower bound
fc.upper[1]  # 68% upper bound
```

---

## Model Diagnostics

### Extract components

```julia
# Coefficients
coefs = coef(var)
intercept = coefs.intercept  # Vector
lags = coefs.lags            # Array (n_vars, n_vars, n_lags)

# Residuals
u = residuals(var)

# Residual covariance
Σ = vcov(var)

# Fitted values
y_hat = fitted(var)
```

### Information criteria

```julia
aic(var)
bic(var)
hqic(var)
```

### Lag selection

```julia
# Try different lags
results = []
for p in 1:12
    var_p = estimate(OLSVAR, Y, p)
    push!(results, (p=p, aic=aic(var_p), bic=bic(var_p)))
end

# Find minimum
best_aic = argmin([r.aic for r in results])
best_bic = argmin([r.bic for r in results])
```

### Stability

```julia
# Eigenvalues of companion matrix
λ = eigvals(var.companion)
max_eigenval = maximum(abs.(λ))

is_stable = max_eigenval < 1
```

### Residual diagnostics

```julia
# Residual correlation
D_inv = Diagonal(1 ./ sqrt.(diag(Σ)))
R = D_inv * Σ * D_inv

# Residual autocorrelation (manual)
u = residuals(var)
for lag in 1:4
    acf = cor(u[1:end-lag,:], u[(lag+1):end,:])
    println("Lag $lag autocorrelation:\n", round.(acf, digits=3))
end
```

---

## Utility Functions

### Accessor functions

```julia
n_vars(var)  # Number of variables
n_lags(var)  # Number of lags
n_obs(var)   # Effective observations (after losing lags)
```

### Companion form

```julia
F = companion_form(coefs.lags)
# F is (n_vars*p × n_vars*p)
```

### Lag operation

```julia
x_lagged = lag(x, 2)  # 2-period lag
# Returns NaN for first 2 observations
```

---

## Advanced Features

### Simulation

```julia
# Simulate from estimated model
innovations = randn(T, n_vars)
Y_sim = simulate_var(var, innovations, Y_init)
```

### Bootstrap IRFs (manual control)

```julia
irf_boot = bootstrap_irf(var, id, horizon, reps;
                         method=:wild,
                         coverage=[0.90])
```

---

## Common Patterns

### Complete analysis workflow

```julia
# 1. Load data
Y = load_data()  # Your data loading function
names = [:var1, :var2, :var3]

# 2. Lag selection
bic_values = [bic(estimate(OLSVAR, Y, p)) for p in 1:12]
p_opt = argmin(bic_values)

# 3. Estimate
var = estimate(OLSVAR, Y, p_opt; names=names)

# 4. Check stability
@assert maximum(abs.(eigvals(var.companion))) < 1 "VAR is unstable!"

# 5. Identify
id = CholeskyID()

# 6. IRFs with inference
irf_result = irf(var, id; horizon=24, bootstrap_reps=1000)

# 7. Variance decomposition
fevd = variance_decomposition(irf_result)

# 8. Historical decomposition
hd = historical_decomposition(var, id)

# 9. Forecast
fc = forecast(var, 12; bootstrap_reps=1000)
```

### Comparing identification schemes

```julia
# Estimate once
var = estimate(OLSVAR, Y, p)

# Multiple identification schemes
id_chol = CholeskyID()
id_chol_alt = CholeskyID(ordering=[:var3, :var1, :var2])
id_sign = SignRestriction(restrictions, 12)

# Compare IRFs
irf_chol = irf(var, id_chol; horizon=24)
irf_chol_alt = irf(var, id_chol_alt; horizon=24)
irf_sign = irf(var, id_sign; horizon=24)
```

### Robustness to lag length

```julia
irf_results = []
for p in [6, 12, 18, 24]
    var = estimate(OLSVAR, Y, p)
    push!(irf_results, irf(var, id; horizon=48))
end

# Compare point estimates across specifications
```

---

## Type Reference

### VAR Specifications

- `OLSVAR` - OLS estimation
- `BayesianVAR(prior)` - Bayesian estimation (planned)
- `IVSVAR(instrument)` - IV-SVAR (planned)
- `LocalProjection` - Local projections (planned)

### Identification Schemes

- `CholeskyID(; ordering=nothing)` - Recursive identification
- `SignRestriction(restrictions, horizon)` - Sign restrictions
- `IVIdentification()` - External instruments (planned)

### Constraints

- `ZeroConstraint(variable, regressors, lags)` - Zero restrictions
- `FixedConstraint(variable, regressor, lag, value)` - Fixed values
- `BlockExogeneity(from, to)` - Block exogeneity

---

## Error Troubleshooting

### "Model is unstable"

```julia
# Check eigenvalues
λ = eigvals(var.companion)
println("Max eigenvalue: ", maximum(abs.(λ)))

# Solutions:
# 1. Reduce lag length
# 2. Add constraints
# 3. Check for outliers in data
# 4. Consider differencing if variables are integrated
```

### "Could not find valid rotation"

Sign restrictions too strict:

```julia
# Solutions:
# 1. Increase max_draws
identify(var, id; max_draws=50000)

# 2. Relax restrictions (fewer sign constraints)
# 3. Shorten restriction horizon
id = SignRestriction(restrictions, 6)  # instead of 12
```

### "Not enough observations"

```julia
# Need T > p + 1
# If T = 100 and you want p = 12:
# Effective observations = T - p = 88

# Solutions:
# 1. Reduce lag length
# 2. Get more data
# 3. Use Bayesian methods with informative priors
```

---

## Performance Tips

### Pre-allocate for multiple estimations

```julia
# Slower
for p in 1:12
    var = estimate(OLSVAR, Y, p)
    # ...
end

# Faster (reuse data structures where possible)
# Just run - Julia's compiler optimizes
```

### Parallel bootstrap

```julia
# Use fewer reps for testing
irf(var, id; bootstrap_reps=100)  # quick test

# Use more for final results
irf(var, id; bootstrap_reps=5000)  # publication
```

---

For complete documentation, see:
- [Getting Started Tutorial](src/tutorials/getting_started.md)
- [Mathematical Theory](src/mathematical/theory.md)
- [Technical Documentation](TECHNICAL.md)
