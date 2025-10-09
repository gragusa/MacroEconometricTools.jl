# Getting Started with MacroEconometricTools.jl

This tutorial introduces the basic functionality of MacroEconometricTools.jl through a running example based on oil market dynamics.

## Installation

```julia
using Pkg
Pkg.add("MacroEconometricTools")
```

Or from the Julia REPL:

```julia
] add MacroEconometricTools
```

## Loading the Package

```julia
using MacroEconometricTools
using LinearAlgebra, Statistics
```

## Example: Oil Market VAR

We'll replicate the analysis of Kilian (2009) who studied the effects of oil supply and demand shocks on the global economy.

### Data Structure

The oil market is modeled with three variables:

1. **Oil production growth** (`oil_prod_growth`): Percentage change in global oil production
2. **Real activity index** (`real_activity`): Measure of global economic activity
3. **Real oil price** (`log_oil_price`): Log of real oil price

### Loading Data

```julia
# Load simulated oil market data
include("data/oil_data.jl")
data = load_oil_data()

Y = data.data              # 456 × 3 matrix
dates = data.dates
names = data.names

println("Data dimensions: ", size(Y))
println("Variables: ", names)
println("Sample: ", dates[1], " to ", dates[end])
```

Output:

```
Data dimensions: (456, 3)
Variables: [:oil_prod_growth, :real_activity, :log_oil_price]
Sample: 1990-01 to 2027-12
```

### Data Exploration

```julia
# Summary statistics
using Statistics

for (i, name) in enumerate(names)
    m = mean(Y[:,i])
    s = std(Y[:,i])
    println("$name: mean = $(round(m, digits=3)), std = $(round(s, digits=3))")
end
```

Output:
```
oil_prod_growth: mean = -0.002, std = 0.312
real_activity: mean = 0.015, std = 2.134
log_oil_price: mean = 3.998, std = 0.042
```

## Estimating a VAR Model

### Basic Estimation

Estimate a VAR(24) model with OLS:

```julia
# Choose lag length
p = 24  # Two years of monthly data

# Estimate VAR
var_model = fit(OLSVAR, Y, p; names=names)

println(var_model)
```

Output:
```
VARModel{Float64,OLSVAR}
  Variables: oil_prod_growth, real_activity, log_oil_price
  Observations: 432 (456 total)
  Lags: 24
```

### Model Properties

```julia
# Access model properties
println("Variable names: ", varnames(var_model))
println("Number of variables: ", n_vars(var_model))
println("Number of lags: ", n_lags(var_model))
println("Effective observations: ", n_obs(var_model))
println("Total observations: ", raw_nobs(var_model))
```

### Examining Coefficients

```julia
# Extract coefficients
coefs = coef(var_model)

# Intercepts
println("Intercepts:")
for (i, name) in enumerate(names)
    println("  $name: ", round(coefs.intercept[i], digits=4))
end

# First lag coefficients (3×3 matrix)
println("\nFirst lag coefficients (A₁):")
display(round.(coefs.lags[:,:,1], digits=3))
```

### Model Diagnostics

```julia
# Residual covariance matrix
Σ = vcov(var_model)
println("Residual covariance:")
display(round.(Σ, digits=4))

# Residual correlation
D_inv = Diagonal(1 ./ sqrt.(diag(Σ)))
R = D_inv * Σ * D_inv
println("\nResidual correlation:")
display(round.(R, digits=3))

# Information criteria
println("\nModel selection criteria:")
println("  AIC:  ", round(aic(var_model), digits=2))
println("  BIC:  ", round(bic(var_model), digits=2))
println("  HQIC: ", round(hqic(var_model), digits=2))
```

### Stability Check

```julia
# Check eigenvalues of companion matrix
F = var_model.companion
eigenvals = eigvals(F)
max_eigenval = maximum(abs.(eigenvals))

println("Maximum eigenvalue modulus: ", round(max_eigenval, digits=4))
println("Model is ", max_eigenval < 1 ? "stable" : "unstable")
```

---

## Identification and Structural Analysis

### Cholesky (Recursive) Identification

The Cholesky decomposition imposes a recursive structure:

```julia
# Identification scheme
# Ordering: production → activity → price
id = CholeskyID()

# Compute rotation matrix (structural impact matrix)
P = rotation_matrix(var_model, id)

println("Structural impact matrix (P):")
display(round.(P, digits=3))

# Verify: P*P' = Σ
println("\nVerification: ||P*P' - Σ|| = ",
        round(norm(P*P' - Matrix(Σ)), digits=10))
```

**Interpretation**:
- Oil production responds only to oil supply shocks contemporaneously
- Real activity responds to supply and demand shocks
- Oil price responds to all three shocks (supply, demand, speculative)

### Alternative Ordering

```julia
# Try different ordering: price → production → activity
id_alt = CholeskyID(ordering=[:log_oil_price, :oil_prod_growth, :real_activity])
P_alt = rotation_matrix(var_model, id_alt)

println("Alternative structural impact matrix:")
display(round.(P_alt, digits=3))
```

---

## Impulse Response Functions

### Computing IRFs

```julia
# Compute IRFs with bootstrap confidence intervals
irf_result = irf(var_model, id;
                 horizon = 48,              # 4 years ahead
                 inference = :bootstrap,
                 bootstrap_reps = 1000,
                 coverage = [0.68, 0.90])

println("IRF dimensions: ", size(irf_result.irf))
# (49, 3, 3) = (horizons, variables, shocks)
```

### Examining Point Estimates

```julia
# Response of oil price to oil supply shock
# irf_result.irf[h+1, response_var, shock_var]

shock_idx = 1  # Oil supply shock
resp_idx = 3   # Oil price response

println("Oil price response to supply shock:")
for h in [0, 6, 12, 24]
    response = irf_result.irf[h+1, resp_idx, shock_idx]
    println("  Horizon $h: ", round(response, digits=4))
end
```

Expected output (signs):
```
Oil price response to supply shock:
  Horizon 0: -0.0523   # Negative (supply ↑ → price ↓)
  Horizon 6: -0.0312
  Horizon 12: -0.0187
  Horizon 24: -0.0045
```

### Confidence Intervals

```julia
# 90% confidence interval for oil price response to supply shock
h_plot = 0:48
point_est = irf_result.irf[:, 3, 1]
lower_90 = irf_result.lower[2][:, 3, 1]  # 90% is second coverage level
upper_90 = irf_result.upper[2][:, 3, 1]

println("Horizon | Point Est | 90% CI")
println("--------|-----------|---------------")
for h in [0, 6, 12, 18, 24]
    println(@sprintf("%7d | %9.4f | [%6.4f, %6.4f]",
            h, point_est[h+1], lower_90[h+1], upper_90[h+1]))
end
```

### Visualizing IRFs

```julia
# Note: Plotting requires RecipesBase extension
# using Plots

# plot(h_plot, point_est, ribbon=(point_est - lower_90, upper_90 - point_est),
#      label="Oil Price Response to Supply Shock",
#      xlabel="Months", ylabel="Percent",
#      title="Impulse Response with 90% Bootstrap CI")
```

---

## Forecast Error Variance Decomposition

```julia
# Compute FEVD
fevd_result = variance_decomposition(irf_result)

# FEVD of oil price at horizon 24 (2 years)
h = 24
println("Oil price variance decomposition at horizon $h:")
for (j, shock_name) in enumerate(["Supply", "Demand", "Speculative"])
    contrib = fevd_result[h+1, 3, j]  # variable 3 = oil price
    println("  $shock_name shock: ", round(contrib * 100, digits=1), "%")
end
```

Expected output:
```
Oil price variance decomposition at horizon 24:
  Supply shock: 15.3%
  Demand shock: 62.8%
  Speculative shock: 21.9%
```

**Interpretation**: After 2 years, demand shocks explain most of the variation in oil prices.

---

## Constrained VAR Estimation

### Block Exogeneity

Suppose we want to impose that oil production is weakly exogenous (doesn't respond to activity or prices):

```julia
# Define constraints
constraints = [
    # Oil production doesn't respond to real activity
    ZeroConstraint(:oil_prod_growth, [:real_activity], Int[]),  # all lags

    # Oil production doesn't respond to oil prices
    ZeroConstraint(:oil_prod_growth, [:log_oil_price], Int[])
]

# Estimate constrained VAR
var_constrained = fit(OLSVAR, Y, p;
                           names=names,
                           constraints=constraints)

println(var_constrained)
```

Output:
```
VARModel{Float64,OLSVAR}
  Variables: oil_prod_growth, real_activity, log_oil_price
  Observations: 432 (456 total)
  Lags: 24
  Constraints: 2 applied
```

### Checking Constraints

```julia
# Verify constraints are satisfied
coefs_c = coef(var_constrained)

# Check that oil production equation has zeros
println("Checking constraints in oil production equation:")

# Position of real_activity in first lag
activity_idx = findfirst(==(names[2]), names)
println("  A₁[oil_prod, real_activity] = ",
        round(coefs_c.lags[1, activity_idx, 1], digits=10))

# Position of oil_price in first lag
price_idx = findfirst(==(names[3]), names)
println("  A₁[oil_prod, oil_price] = ",
        round(coefs_c.lags[1, price_idx, 1], digits=10))
```

Expected:
```
Checking constraints in oil production equation:
  A₁[oil_prod, real_activity] = 0.0
  A₁[oil_prod, oil_price] = 0.0
```

### Comparing Models

```julia
# Likelihood ratio test (informal)
ll_unrestricted = log_likelihood(var_model)
ll_restricted = log_likelihood(var_constrained)

LR_stat = 2 * (ll_unrestricted - ll_restricted)
df = 2 * p  # 2 constraints × p lags

println("Likelihood Ratio Test:")
println("  LR statistic: ", round(LR_stat, digits=2))
println("  Degrees of freedom: ", df)
println("  Critical value (5%): ", round(quantile(Chisq(df), 0.95), digits=2))
```

---

## Advanced Topics

### Historical Decomposition

```julia
# Decompose observed oil price into contributions from shocks
hd = historical_decomposition(var_model, id)

# hd[:, var_idx, shock_idx] = contribution of shock_idx to variable var_idx

println("Historical decomposition dimensions: ", size(hd))
# (432, 3, 3) = (time periods, variables, shocks)
```

### Cumulative IRFs

```julia
# Compute cumulative response (for levels instead of growth rates)
cirf_result = cumulative_irf(irf_result)

# Cumulative effect of supply shock on oil production
println("Cumulative oil production response to supply shock:")
for h in [0, 12, 24, 48]
    cum_resp = cirf_result.irf[h+1, 1, 1]
    println("  Horizon $h: ", round(cum_resp, digits=4))
end
```

### Forecasting

```julia
# Generate forecasts
h_forecast = 12  # 1 year ahead
fc = forecast(var_model, h_forecast;
              coverage=[0.68, 0.90],
              bootstrap_reps=1000)

println("12-month ahead forecast:")
for (i, name) in enumerate(names)
    println("  $name: ", round(fc.mean[end, i], digits=3),
            " ± ", round(fc.std[end, i], digits=3))
end
```

---

## Next Steps

### Tutorials

1. **Sign Restrictions**: Identify shocks using economic theory
2. **Local Projections**: Alternative to VAR for IRF estimation
3. **Panel VAR**: Extension to panel data
4. **Bayesian VAR**: Incorporate prior information

### Documentation

- See [`docs/src/mathematical/theory.md`](../mathematical/theory.md) for mathematical details
- See [`docs/TECHNICAL.md`](../../TECHNICAL.md) for implementation details

### Examples

- Oil market shocks (Kilian 2009)
- Monetary policy (Gertler-Karadi 2015)
- Fiscal multipliers (Ramey 2011)
- Technology shocks (Galí 1999)

---

## References

- Kilian, L. (2009). Not All Oil Price Shocks Are Alike: Disentangling Demand and Supply Shocks in the Crude Oil Market. *American Economic Review*, 99(3), 1053-1069.

- Kilian, L., & Lütkepohl, H. (2017). *Structural Vector Autoregressive Analysis*. Cambridge University Press.

- Lütkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*. Springer.
