# Technical Implementation Documentation

## MacroEconometricTools.jl - Architecture and Implementation

This document describes the internal architecture, design decisions, and implementation details of MacroEconometricTools.jl.

---

## Table of Contents

1. [Design Philosophy](#design-philosophy)
2. [Type System Architecture](#type-system-architecture)
3. [Constraint System Implementation](#constraint-system-implementation)
4. [VAR Estimation Details](#var-estimation-details)
5. [Identification Schemes](#identification-schemes)
6. [Bootstrap and Inference](#bootstrap-and-inference)
7. [Performance Considerations](#performance-considerations)
8. [Extending the Package](#extending-the-package)

---

## Design Philosophy

### Type-Based Dispatch

The package uses Julia's multiple dispatch system extensively. Instead of a unified interface with method flags (e.g., `fit(..., method=:ols)`), we use type-based dispatch:

```julia
# Good: Type-based dispatch
var_ols = fit(OLSVAR, Y, lags)
irfs = irf(var_ols, CholeskyID())
irfs_sign = irf(var_ols, SignRestriction(restrictions))

# Avoided: Method flags
var = fit(Y, lags; method=:ols)  # NOT used
```

**Rationale**:
- Type stability: Compiler can optimize better
- Extensibility: Easy to add new estimators without modifying existing code
- Clarity: The type system documents the API

### Parametric Types Throughout

All core types are fully parametric:

```julia
struct VARModel{T<:AbstractFloat, S<:AbstractVARSpec, D<:Union{Nothing,AbstractVector}}
    spec::S
    Y::Matrix{T}
    coefficients::VARCoefficients{T}
    # ...
end
```

**Benefits**:
- No type instabilities from abstract containers
- Compiler can generate specialized code for each concrete type
- Memory layout is predictable and cache-friendly

### No Missing Values in Core Computations

The package uses `NaN` instead of `missing` for unavailable data points:

```julia
# lag() function returns NaN for initial observations
lag(x::Vector{Float64}, 1)  # Returns Vector{Float64}, not Vector{Union{Missing,Float64}}
```

**Rationale**:
- Maintains concrete types throughout
- Better performance (no Union splitting)
- Simpler type signatures
- Standard in numerical computing

---

## Type System Architecture

### Abstract Type Hierarchy

```
AbstractVARSpec
├── OLSVAR (empty struct, pure dispatch)
└── IVSVAR{I<:AbstractInstrument} (deprecated; use OLSVAR + IVIdentification)

AbstractInstrument
└── ExternalInstrument{T,S}

AbstractConstraint
├── ZeroConstraint
├── FixedConstraint
└── BlockExogeneity

AbstractIdentification
├── CholeskyID
├── SignRestriction
└── IVIdentification
```

### Core Data Structures

#### VARCoefficients{T}

Stores estimated coefficients with constraint metadata:

```julia
struct VARCoefficients{T<:AbstractFloat}
    intercept::Vector{T}              # Length n_vars
    lags::Array{T,3}                  # (n_vars, n_vars, n_lags)
    constraints::Union{Nothing, Vector{AbstractConstraint}}  # nothing when unconstrained
end
```

**Layout**: Lag coefficients are stored as `(response_var, regressor_var, lag_number)`. For example:
- `lags[i,j,k]` = coefficient of variable j at lag k in equation for variable i
- This matches the standard VAR notation: `Y_t = c + A_1 Y_{t-1} + ... + A_p Y_{t-p} + ε_t`

#### VARModel{T,S,D}

Main model container:

```julia
struct VARModel{T<:AbstractFloat, S<:AbstractVARSpec, D<:Union{Nothing,AbstractVector}}
    spec::S                                    # Estimation specification
    Y::Matrix{T}                              # Original data (T × n_vars)
    X::Matrix{T}                              # Design matrix with lags
    coefficients::VARCoefficients{T}          # Estimated coefficients
    residuals::Matrix{T}                      # Residuals (T-p × n_vars)
    Σ::Symmetric{T,Matrix{T}}                 # Residual covariance
    companion::Matrix{T}                      # Companion form matrix
    names::Vector{Symbol}                     # Variable names
    dates::D                                  # Residual-row dates, or nothing
    metadata::NamedTuple                      # Fit information not read by the package
end
```

**Memory layout**:
- `Y` is never modified after estimation
- `X` includes intercept column and all lags: size `(T, 1 + n_vars * p)`
- `residuals` has `T - p` rows (loses first p observations)

#### IRF Result Types

The package uses an abstract type hierarchy for impulse response results to support both point-identified and set-identified models:

```julia
abstract type AbstractIRFResult{T<:AbstractFloat} end

struct IRFResult{T} <: AbstractIRFResult{T}
    irf::Array{T,3}                    # Point estimate (horizon+1, n_vars, n_shocks)
    stderr::Array{T,3}                 # Standard errors
    lower::Vector{Array{T,3}}          # Lower bounds for each coverage level
    upper::Vector{Array{T,3}}          # Upper bounds for each coverage level
    coverage::Vector{Float64}          # Coverage levels (e.g., [0.68, 0.90])
    identification::AbstractIdentification
    inference::Symbol                  # :bootstrap, :delta, or :none
    metadata::NamedTuple
end

struct SignRestrictedIRFResult{T} <: AbstractIRFResult{T}
    irf_median::Array{T,3}             # Median IRF (horizon+1, n_vars, n_shocks)
    irf_draws::Array{T,4}              # All draws (n_draws, horizon+1, n_vars, n_shocks)
    lower::Vector{Array{T,3}}          # Pointwise quantile bands
    upper::Vector{Array{T,3}}
    coverage::Vector{Float64}
    rotation_matrices::Vector{Matrix{T}}  # All valid P matrices
    identification::SignRestriction
    metadata::NamedTuple
end
```

**Design rationale**:
- **Separate types**: Point vs set identification have fundamentally different data structures
- **Abstract supertype**: Enables polymorphic dispatch on accessor methods like `horizon()`, `n_vars()`
- **Type stability**: Each struct has concrete field types, no `Union{T, Nothing}` fields
- **Extensibility**: Easy to add new result types (e.g., `BayesianIRFResult`) without modifying existing code

**Key distinction**:
- `IRFResult`: Single IRF with confidence bands (Cholesky, IV)
- `SignRestrictedIRFResult`: Multiple IRF draws representing identified set

---

## Constraint System Implementation

The constraint system allows flexible restrictions on VAR coefficients during estimation.

### Constraint Types

#### 1. ZeroConstraint

Sets specific coefficients to zero:

```julia
struct ZeroConstraint <: AbstractConstraint
    variable::Symbol              # Equation (dependent variable)
    regressors::Vector{Symbol}    # RHS variables to zero out
    lags::Vector{Int}            # Which lags (empty = all lags)
end
```

**Example**:
```julia
# Interest rate doesn't affect GDP at lag 1
ZeroConstraint(:GDP, [:InterestRate], [1])
```

#### 2. FixedConstraint

Fixes a coefficient to a specific value:

```julia
struct FixedConstraint <: AbstractConstraint
    variable::Symbol
    regressor::Symbol
    lag::Int
    value::Float64
end
```

**Use case**: Unit root restrictions, calibrated parameters.

#### 3. BlockExogeneity

Block exogeneity: variables in `from` don't affect variables in `to`:

```julia
struct BlockExogeneity <: AbstractConstraint
    from::Vector{Symbol}
    to::Vector{Symbol}
end
```

**Example**: Small open economy where foreign variables don't respond to domestic variables.

### Estimation with Constraints

Every constraint is first resolved against the variable names and lag order by
`_resolve_constraints`, which returns a `(1 + n_vars * n_lags) × n_vars` mask of
restricted coefficients (layout of the coefficient matrix `B`: intercept first,
column = equation) and their values. It throws `ArgumentError` for unknown
variable names, lags outside `1:n_lags`, and conflicting values for the same
coefficient. `check_constraints`, `apply_constraints!`, `build_selection_matrix`,
and `constrained_ols` all go through it.

`constrained_ols` estimates equation by equation: the restricted coefficients take
their values (zero or fixed), their contribution is subtracted from the dependent
variable, and the free coefficients are estimated by OLS on the remaining
regressors:

```julia
for eq in 1:n_vars
    fixed = findall(restricted[:, eq])
    free = findall(.!restricted[:, eq])
    y = Y[:, eq] - X[:, fixed] * values[fixed, eq]
    A[fixed, eq] = values[fixed, eq]
    A[free, eq] = X[:, free] \ y
end
```

**Mathematical formulation**: given `y_k = X β_k + ε_k` for equation `k`, minimize
`||y_k - X β_k||²` subject to `β_k[fixed] = r_k`. Because every constraint
restricts coefficients of a single equation, the equations are estimated
separately.

---

## VAR Estimation Details

### Data Preparation

#### Lag Matrix Construction

The function `create_lags(Y, p)` builds the design matrix:

```julia
function create_lags(X::AbstractMatrix{T}, p::Int) where T<:AbstractFloat
    n_obs, n_vars = size(X)
    lagged = Matrix{T}(undef, n_obs, 1 + n_vars * p)

    # Column 1: intercept
    lagged[:, 1] .= one(T)

    # Remaining columns: lags
    for lag_num in 1:p
        for var_idx in 1:n_vars
            col_idx = 1 + (lag_num - 1) * n_vars + var_idx
            lagged[:, col_idx] = lag(X[:, var_idx], lag_num; default=T(NaN))
        end
    end

    return lagged
end
```

**Column ordering**: `[1, Y₁(t-1), Y₂(t-1), ..., Yₙ(t-1), Y₁(t-2), ..., Yₙ(t-p)]`

**First p rows**: Contain NaN values (insufficient lags). These rows are excluded from estimation.

### OLS Estimation

```julia
function fit(::Type{OLSVAR}, Y, n_lags; constraints=[], demean=false)
    # Create lagged matrix
    X = create_lags(Y, n_lags)

    # Remove first n_lags rows (contain NaN)
    valid_idx = (n_lags + 1):size(Y, 1)
    Y_est = Y[valid_idx, :]
    X_est = X[valid_idx, :]

    # Estimate: A = (X'X)⁻¹X'Y
    A = X_est \ Y_est  # Uses QR decomposition internally

    # Residuals
    residuals = Y_est - X_est * A

    # Covariance (degrees of freedom correction)
    n_eff = size(residuals, 1)
    df = n_eff - n_vars * n_lags - 1
    Σ = Symmetric((residuals' * residuals) / df)

    # Build companion form
    F = companion_form(lags_matrix)

    return VARModel(...)
end
```

**Numerical stability**: Julia's `\` operator automatically chooses the best algorithm (QR for overdetermined systems).

### Companion Form

The companion form representation is:

```
[Y_t    ]   [A₁  A₂  ... A_{p-1}  A_p] [Y_{t-1}  ]   [ε_t]
[Y_{t-1}] = [I   0   ... 0        0  ] [Y_{t-2}  ] + [0  ]
[...    ]   [0   I   ... 0        0  ] [  ...    ]   [...]
[Y_{t-p+1}] [0   0   ... I        0  ] [Y_{t-p}  ]   [0  ]
```

Matrix size: `(n_vars × p) × (n_vars × p)`

```julia
function companion_form(A::Array{T,3}) where T
    n_vars, _, n_lags = size(A)
    n = n_vars * n_lags
    F = zeros(T, n, n)

    # Top block: lag coefficients
    for lag in 1:n_lags
        F[1:n_vars, ((lag-1)*n_vars+1):(lag*n_vars)] = A[:, :, lag]
    end

    # Lower blocks: identity matrices
    for i in 1:(n_lags-1)
        row_idx = (n_vars*i+1):(n_vars*(i+1))
        col_idx = (n_vars*(i-1)+1):(n_vars*i)
        F[row_idx, col_idx] = I(n_vars)
    end

    return F
end
```

**Uses**: IRF computation, stability analysis (eigenvalues), forecasting.

---

## Identification Schemes

### Cholesky (Recursive) Identification

**Mathematical foundation**: Given reduced-form residual covariance Σ, find P such that:
- Σ = PP'
- P is lower triangular

**Implementation**:

```julia
function identify_cholesky(model::VARModel, id::CholeskyID)
    Σ = residual_covariance(model)

    # Handle ordering if specified
    if !isnothing(id.ordering)
        perm = [findfirst(==(v), model.names) for v in id.ordering]
        Σ_ordered = Σ[perm, perm]
        P_ordered = cholesky(Symmetric(Σ_ordered)).L
        # Permute back
        P = P_ordered[invperm(perm), :]
    else
        P = cholesky(Symmetric(Σ)).L
    end

    return Matrix(P)
end
```

**Properties**:
- `P * P' = Σ` (verified in tests)
- Lower triangular structure imposes recursive ordering
- Fast and unique (given ordering)

### Sign Restrictions

**Algorithm** (Rubio-Ramírez et al., 2010):

1. Compute any decomposition Σ = PP' (e.g., Cholesky)
2. Draw random orthonormal matrix Q (from Haar measure)
3. Compute candidate: P̃ = PQ
4. Check if IRFs from P̃ satisfy sign restrictions
5. Accept if yes, reject and redraw if no

```julia
function identify_sign(model, id::SignRestriction; max_draws=10000)
    Σ = residual_covariance(model)
    P_chol = cholesky(Symmetric(Σ)).L

    for draw in 1:max_draws
        Q = random_orthonormal(size(Σ, 1))
        P_candidate = P_chol * Q

        # Compute IRFs
        irfs = compute_irf_point(model, P_candidate, id.horizon)

        # Check restrictions
        if check_sign_restrictions(irfs, id.restrictions, id.horizon)
            return P_candidate
        end
    end

    error("Could not find rotation satisfying sign restrictions")
end
```

**Sign restriction matrix**:
- `+1`: Positive response required
- `-1`: Negative response required
- `0` or `NaN`: Unrestricted

### IV Identification

**Stub implementation**: Placeholder for instrumental variable identification (Gertler-Karadi, Mertens-Ravn, etc.).

**Mathematical setup**: Given external instrument z_t for shock ε_t:
1. Relevance: E[z_t ε_t] ≠ 0
2. Exogeneity: E[z_t ε_s] = 0 for s ≠ t
3. Identification: First column of P via IV regression

---

## Bootstrap and Inference

### Bootstrap Methods

Three bootstrap methods are implemented:

#### 1. Wild Bootstrap

**Procedure**:
```julia
function wild_bootstrap_residuals(rng::AbstractRNG, residuals::Matrix)
    T, n = size(residuals)
    # Rademacher weights: ±1 with equal probability
    weights = rand(rng, (-1, 1), T)
    return residuals .* weights
end
```

**Use case**: Heteroskedasticity-robust inference.

**Properties**: Preserves cross-equation correlation in residuals.

#### 2. Standard Bootstrap

```julia
function standard_bootstrap_residuals(rng::AbstractRNG, residuals::Matrix)
    T, n = size(residuals)
    indices = rand(rng, 1:T, T)
    return residuals[indices, :]
end
```

**Use case**: Default for iid errors.

**Properties**: Resamples entire rows (preserves cross-equation correlation).

#### 3. Block Bootstrap

```julia
function block_bootstrap_residuals(residuals::Matrix, block_length::Int)
    T, n = size(residuals)
    # Create overlapping blocks
    # Resample blocks
    # Truncate to original length
end
```

**Use case**: Time series dependence beyond VAR lags.

### IRF Bootstrap Procedure

```julia
function bootstrap_irf(model, identification, horizon, reps; method=:wild)
    # Storage
    irf_draws = zeros(horizon+1, n_vars, n_vars, reps)

    for b in 1:reps
        # 1. Resample residuals
        ε_boot = resample_residuals(model.residuals, method)

        # 2. Simulate VAR with bootstrap residuals
        Y_boot = simulate_var(model, ε_boot, Y_init)

        # 3. Re-estimate VAR
        model_boot = fit(typeof(model.spec), Y_boot, n_lags)

        # 4. Compute rotation matrix
        P_boot = rotation_matrix(model_boot, identification)

        # 5. Compute IRF
        irf_draws[:,:,:,b] = compute_irf_point(model_boot, P_boot, horizon)
    end

    return irf_draws
end
```

**Confidence intervals**: Computed as percentiles of bootstrap distribution.

**Bias correction**: Can be added by computing `2 × θ̂ - mean(θ̂_boot)`.

---

## Performance Considerations

### Memory Allocation

**Preallocated arrays**: The package provides mutating functions where appropriate:

```julia
create_lags!(dest, X, p)  # In-place version
```

**Trade-off**: Code clarity vs. performance. Allocating versions are preferred unless profiling shows bottlenecks.

### Type Stability

All functions maintain type stability:

```julia
# Good: Type-stable
function foo(x::Vector{Float64})
    y = similar(x)  # Vector{Float64}
    y .= 2 .* x
    return y        # Vector{Float64}
end

# Bad: Type-unstable (not in package)
function bar(x::Vector)
    if length(x) > 10
        return sum(x)      # Float64
    else
        return collect(x)  # Vector
    end
end
```

**Check type stability**: Use `@code_warntype` during development.

### Broadcasting

Dot syntax is used extensively for fused operations:

```julia
# Fused: single loop, no temporaries
residuals .= Y .- X * A

# Not fused: creates temporary
residuals = Y - X * A
```

**Performance**: Broadcasting can be 2-3× faster for element-wise operations.

---

## Extending the Package

### Adding a New Estimator

1. **Define estimator type**:

```julia
struct MyEstimator <: AbstractVARSpec
    hyperparameter::Float64
end
```

2. **Implement estimation method**:

```julia
function fit(::Type{MyEstimator}, Y::AbstractMatrix{T}, n_lags::Int;
                  kwargs...) where T<:AbstractFloat
    # Estimation logic
    # ...
    return VARModel(MyEstimator(param), Y, X, coefs, residuals, Σ, F, names, metadata)
end
```

3. **Export the type**:

```julia
export MyEstimator
```

### Adding a New Identification Scheme

1. **Define identification type**:

```julia
struct MyIdentification <: AbstractIdentification
    param::Float64
end
```

2. **Implement rotation matrix method**:

```julia
function rotation_matrix(model::VARModel, id::MyIdentification)
    # Return structural impact matrix P such that PP' = Σ
    return P
end
```

### Adding a New Constraint Type

1. **Define constraint**:

```julia
struct MyConstraint <: AbstractConstraint
    # fields
end
```

2. **List the restricted coefficients**: add a method returning
   `(equation, coefficient row, value)` triples; resolution, validation, and
   estimation then handle the new type:

```julia
function _restricted_cells(c::MyConstraint, varnames::Vector{Symbol}, n_lags::Int)
    # return [(eq, _coef_row(reg_idx, lag, length(varnames)), value), ...]
end
```

---

## Testing Strategy

### Unit Tests

Test individual components in isolation:
- Lag operations
- Companion form construction
- Constraint application
- Identification schemes

### Integration Tests

Test complete workflows:
- VAR estimation → IRF computation
- Constrained estimation → Bootstrap inference

### Property-Based Tests

Verify mathematical properties:
- `P * P' ≈ Σ` for Cholesky identification
- Companion form eigenvalues for stability
- Constrained estimates satisfy constraints

### Performance Tests

Monitor performance regressions:
- Benchmark estimation for various problem sizes
- Check allocation counts

---

## References

- Lütkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*
- Kilian, L., & Lütkepohl, H. (2017). *Structural Vector Autoregressive Analysis*
- Rubio-Ramírez, J. F., Waggoner, D. F., & Zha, T. (2010). Structural Vector Autoregressions: Theory of Identification and Algorithms for Inference. *Review of Economic Studies*
