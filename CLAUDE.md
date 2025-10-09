# MacroEconometricTools.jl — Claude Session Guide

**Purpose**: This file guides Claude (AI assistant) in working on MacroEconometricTools.jl, ensuring consistency with established design patterns when extending the package or integrating new code.

**For comprehensive design philosophy**: Read `docs/IMPLEMENTATION_PHILOSOPHY.md` first.

## Quick Start for New Sessions

### First Steps
1. **Read this file** to understand package-specific conventions
2. **Consult `docs/IMPLEMENTATION_PHILOSOPHY.md`** for detailed design rationale
3. **Check `docs/RECENT_CHANGES.md`** for recent API changes
4. **Review `docs/TECHNICAL.md`** for implementation details

### Key References During Work
- **Type hierarchy**: See `src/types.jl` for all struct definitions
- **Examples**: Check existing code in `src/var/` for patterns
- **Tests**: Run `julia --project=. -e 'using Pkg; Pkg.test()'` after changes

## Core Package Design Principles

### 1. Type-Based Dispatch is Primary

**Pattern**: Use Julia's type system and multiple dispatch as the API design mechanism.

```julia
# ✓ CORRECT: Type-based dispatch
var = estimate(OLSVAR, Y, p)
bayesian_var = estimate(BayesianVAR(prior), Y, p)

# ✗ WRONG: Method flags
var = estimate(Y, p; method=:ols)  # Avoid this pattern
```

**When integrating new code**: Convert method flag patterns to type-based dispatch.

**Example translation**:
```julia
# Old code you might see:
function estimate_var(Y, p; method=:ols, prior=nothing)
    if method == :ols
        # ...
    elseif method == :bayesian
        # ...
    end
end

# Translate to:
struct OLSVAR <: AbstractVARSpec end
struct BayesianVAR{P<:AbstractPrior} <: AbstractVARSpec
    prior::P
end

function estimate(spec::OLSVAR, Y, p; kwargs...)
    # OLS-specific estimation
end

function estimate(spec::BayesianVAR, Y, p; kwargs...)
    # Bayesian-specific estimation
end
```

### 2. Separate Types for Different Data Structures

**Principle**: When different use cases require fundamentally different data structures, use separate concrete types with a shared abstract parent. Don't force different structures into one type with `Union{T, Nothing}` fields.

**Example - IRF Results**:

```julia
# ✓ CORRECT: Separate types for different structures
abstract type AbstractIRFResult{T<:AbstractFloat} end

struct IRFResult{T} <: AbstractIRFResult{T}
    irf::Array{T,3}              # Point-identified: single IRF
    stderr::Array{T,3}
    # ... all fields are concrete types
end

struct SignRestrictedIRFResult{T} <: AbstractIRFResult{T}
    irf_median::Array{T,3}       # Set-identified: median of many draws
    irf_draws::Array{T,4}        # Store all draws (4D array)
    rotation_matrices::Vector{Matrix{T}}
    # ... all fields are concrete types
end

# Generic interface via dispatch
horizon(irf::IRFResult) = size(irf.irf, 1) - 1
horizon(irf::SignRestrictedIRFResult) = size(irf.irf_median, 1) - 1

# ✗ WRONG: One type with Union fields (type-unstable)
struct IRFResult{T}
    irf::Array{T,3}
    irf_draws::Union{Array{T,4}, Nothing}     # Type-unstable!
    rotation_matrices::Union{Vector{Matrix{T}}, Nothing}  # Wasteful!
    stderr::Union{Array{T,3}, Nothing}        # Error-prone!
end
```

**When to use this pattern**:

- Different fields needed for different use cases
- Would require `Union{T, Nothing}` in unified type
- Each variant has distinct semantic meaning

**When to use parameterization instead**:

- Same fields for all variants
- Type parameter affects computation, not storage
- Example: `VARModel{T, S}` where T is Float64 vs Float32

### 3. Accessor Functions Over Field Access

**Pattern**: Export accessor functions, treat struct fields as implementation details.

```julia
# ✓ CORRECT: Use accessor functions
n = n_vars(model)
h = horizon(irf_result)
names = varnames(model)

# ✗ WRONG: Direct field access from user code
n = size(model.Y, 2)
h = size(irf_result.irf, 1) - 1
names = model.names
```

**Rationale**:

- Allows internal structure changes without breaking user code
- Enables dispatch (works polymorphically on abstract types)
- Provides consistent interface

**Standard accessor names in this package**:

- `n_vars(x)` - Number of variables
- `n_lags(x)` - Number of lags
- `n_obs(x)` - Effective observations (after losing lags)
- `raw_nobs(x)` - Total observations before lags
- `varnames(x)` - Variable names as Vector{Symbol}
- `horizon(x)` - IRF horizon
- `n_shocks(x)` - Number of structural shocks

### 4. NaN for Numeric Missing Data

**Pattern**: Use `NaN` instead of `Missing` for unavailable numeric values.

```julia
# ✓ CORRECT: NaN maintains type stability
function lag(x::Vector{Float64}, k::Int)
    result = similar(x)
    result[1:k] .= NaN  # Type: Vector{Float64}
    result[k+1:end] = x[1:end-k]
    return result
end

# ✗ WRONG: Missing creates Union types
function lag(x::Vector{Float64}, k::Int)
    result = Vector{Union{Missing, Float64}}(missing, length(x))
    # Type: Vector{Union{Missing, Float64}} - type-unstable!
end
```

### 5. Fully Parametric Types

**Pattern**: All core types use concrete type parameters.

```julia
# ✓ CORRECT: Parametric types throughout
struct VARModel{T<:AbstractFloat, S<:AbstractVARSpec}
    Y::Matrix{T}                    # Concrete element type
    coefficients::VARCoefficients{T}
    residuals::Matrix{T}
    Σ::Symmetric{T,Matrix{T}}
end

struct IRFResult{T<:AbstractFloat} <: AbstractIRFResult{T}
    irf::Array{T,3}
    stderr::Array{T,3}
    # ...
end

# ✗ WRONG: Abstract fields (runtime dispatch)
struct VARModel
    Y::AbstractMatrix  # Field access incurs runtime dispatch!
    residuals::AbstractMatrix
end
```

**When integrating new code**: Add type parameters, propagate them through call stack.

## Package-Specific API Conventions

### Naming Conventions

#### Functions

- **Estimation**: `estimate(spec::AbstractVARSpec, Y, p; kwargs...)`
- **Identification**: `rotation_matrix(model, id::AbstractIdentification)`
  - **Note**: We use `rotation_matrix`, NOT `identify` (breaking change from earlier versions)
  - **Rationale**: "rotation matrix" precisely describes what it returns (the P matrix)
- **IRF computation**: `irf(model, identification; kwargs...)`
- **Accessors**: Lowercase with underscores (e.g., `n_vars`, `raw_nobs`, `varnames`)

#### Types

- **Specifications**: `OLSVAR`, `BayesianVAR`, `LocalProjection` (concrete types)
- **Abstract types**: `AbstractVARSpec`, `AbstractIdentification`, `AbstractIRFResult`
- **Results**: `IRFResult`, `SignRestrictedIRFResult`, `ForecastResult`
- **Constraints**: `ZeroConstraint`, `FixedConstraint`, `BlockExogeneity`

### Function Signature Patterns

#### Estimation

```julia
function estimate(spec::ConcreteVARSpec, Y::Matrix{T}, p::Int;
                 names::Union{Vector{Symbol}, Nothing}=nothing,
                 constraints::Union{Vector{<:AbstractConstraint}, Nothing}=nothing,
                 demean::Bool=false) where T<:AbstractFloat
    # Returns VARModel{T, typeof(spec)}
end
```

#### Identification

```julia
function rotation_matrix(model::VARModel{T}, id::AbstractIdentification;
                        kwargs...) where T
    # Returns Matrix{T}
end
```

#### IRF Computation

```julia
# Point-identified (Cholesky, IV)
function irf(model::VARModel{T}, id::PointIdentified;
            horizon::Int=24,
            inference::Symbol=:bootstrap,
            bootstrap_reps::Int=1000,
            coverage::Vector{Float64}=[0.68, 0.90, 0.95],
            rng::AbstractRNG=Random.default_rng()) where T
    # Returns IRFResult{T}
end

# Set-identified (Sign restrictions)
function irf(model::VARModel{T}, id::SignRestriction;
            n_draws::Int=1000,
            max_attempts::Int=10000,
            horizon::Int=24,
            coverage::Vector{Float64}=[0.68, 0.90, 0.95],
            rng::AbstractRNG=Random.default_rng()) where T
    # Returns SignRestrictedIRFResult{T}
end
```

**Note**: Always accept `rng` parameter for reproducibility.


## Translation Checklist for Integrating New Code

When translating external code or integrating new features (e.g., BayesianVAR, LinearStateSpace):

### Step 1: Identify Patterns to Convert

- [ ] **Method flags** → Type-based dispatch
- [ ] **Keyword method selection** → Separate methods on concrete types
- [ ] **Abstract containers** → Parametric types with concrete type parameters
- [ ] **Missing values** → NaN for numeric data
- [ ] **Direct field access** → Accessor functions
- [ ] **Monolithic functions** → Small, composable functions

### Step 2: Define Type Hierarchy

```julia
# 1. Add abstract type if needed
abstract type AbstractStateSpace end

# 2. Define concrete type(s)
struct LinearStateSpace{T<:AbstractFloat} <: AbstractStateSpace
    # All fields have concrete types
    F::Matrix{T}       # Transition matrix
    H::Matrix{T}       # Measurement matrix
    Q::Matrix{T}       # State noise covariance
    R::Matrix{T}       # Measurement noise covariance
    μ0::Vector{T}      # Initial state mean
    Σ0::Matrix{T}      # Initial state covariance
end

# 3. Add to exports in src/MacroEconometricTools.jl
export AbstractStateSpace, LinearStateSpace
```

### Step 3: Implement Core Methods

```julia
# Estimation
function estimate(spec::StateSpaceSpec, Y::Matrix{T}, ...) where T
    # Return StateSpaceModel{T, typeof(spec)}
end

# Accessors
n_vars(ss::LinearStateSpace) = size(ss.H, 1)
n_states(ss::LinearStateSpace) = size(ss.F, 1)

# Core functionality
function kalman_filter(ss::LinearStateSpace{T}, Y::Matrix{T}) where T
    # Implementation
end

function kalman_smoother(ss::LinearStateSpace{T}, Y::Matrix{T}) where T
    # Implementation
end
```

### Step 4: Integrate with Existing Infrastructure

```julia
# If StateSpace models can compute IRFs, implement:
function irf(model::StateSpaceModel{T}, shock_idx::Int; horizon::Int=24) where T
    # Return IRFResult{T} or custom result type
end

# If they support constraints:
function estimate(spec::StateSpaceSpec, Y, ...;
                 constraints::Vector{<:AbstractConstraint}=nothing)
    # Apply constraints during estimation
end
```

### Step 5: Add Accessor Methods

Follow the established pattern:
```julia
# Required accessors for all model types
n_vars(model::NewModelType) = ...
n_obs(model::NewModelType) = ...
raw_nobs(model::NewModelType) = ...
varnames(model::NewModelType) = ...

# Type-specific accessors
n_states(model::StateSpaceModel) = ...
# etc.
```

### Step 6: Documentation and Tests

```julia
# 1. Add docstring to every exported function
"""
    estimate(spec::StateSpaceSpec, Y::Matrix, ...)

Estimate a state space model.

# Arguments
- `spec::StateSpaceSpec`: Specification (e.g., `KalmanFilter()`)
- `Y::Matrix{T}`: Data matrix (T × n_vars)

# Returns
- `StateSpaceModel{T}`: Estimated model

# Examples
```julia
ss = estimate(KalmanFilter(), Y)
```
"""

# 2. Add tests in test/test_statespace.jl
@testset "State Space Models" begin
    # Test estimation
    # Test filtering
    # Test smoothing
    # Test type stability
end

# 3. Add tutorial in docs/src/tutorials/statespace.md
```

### Step 7: Update Documentation

- [ ] Add to feature list in README.md
- [ ] Add examples to QUICK_REFERENCE.md
- [ ] Update TECHNICAL.md with implementation details
- [ ] Update RECENT_CHANGES.md
- [ ] Create tutorial if major feature

---

## Specific Examples: BayesianVAR Integration

Here's how to integrate BayesianVAR following our patterns:

### Type Definitions

```julia
# In src/types.jl

# Abstract prior type
abstract type AbstractPrior end

# Concrete prior types
struct MinnesotaPrior{T<:AbstractFloat} <: AbstractPrior
    λ1::T  # Overall tightness
    λ2::T  # Cross-variable tightness
    λ3::T  # Lag decay
    λ4::T  # Exogenous tightness
end

struct NormalWishartPrior{T<:AbstractFloat} <: AbstractPrior
    μ::Vector{T}         # Prior mean
    Ψ::Matrix{T}         # Prior scale
    ν::Int               # Prior degrees of freedom
end

# Specification type
struct BayesianVAR{P<:AbstractPrior} <: AbstractVARSpec
    prior::P
end

# Model type (reuse VARModel, add Bayesian-specific fields in metadata)
# OR create BayesianVARModel{T, P} if significantly different

# Result type
struct BayesianIRFResult{T} <: AbstractIRFResult{T}
    irf_mean::Array{T,3}        # Posterior mean
    irf_draws::Array{T,4}       # Posterior draws
    lower::Vector{Array{T,3}}   # Credible bands
    upper::Vector{Array{T,3}}
    coverage::Vector{Float64}
    metadata::NamedTuple
end
```

### Implementation

```julia
# In src/var/estimation.jl

function estimate(spec::BayesianVAR{P}, Y::Matrix{T}, p::Int;
                 names::Union{Vector{Symbol}, Nothing}=nothing,
                 n_draws::Int=1000,
                 burn_in::Int=500,
                 rng::AbstractRNG=Random.default_rng()) where {T, P}

    # Setup
    n_vars_val = size(Y, 2)
    names = names === nothing ? [Symbol("Y_$i") for i in 1:n_vars_val] : names

    # Prior
    prior_params = setup_prior(spec.prior, Y, p)

    # MCMC sampling
    draws = posterior_sampler(Y, p, prior_params, n_draws + burn_in; rng=rng)
    posterior_draws = draws[(burn_in+1):end]

    # Posterior mean as point estimate
    coefficients = posterior_mean(posterior_draws)

    # Build model
    return VARModel{T, typeof(spec)}(
        spec=spec,
        Y=Y,
        coefficients=coefficients,
        # ... standard fields ...
        metadata=(
            prior=spec.prior,
            n_draws=n_draws,
            burn_in=burn_in,
            posterior_draws=posterior_draws,  # Store for IRFs
            # ...
        )
    )
end

# In src/var/irfs.jl

function irf(model::VARModel{T, BayesianVAR{P}}, id::AbstractIdentification;
            horizon::Int=24,
            use_posterior::Bool=true,
            coverage::Vector{Float64}=[0.68, 0.90, 0.95]) where {T, P}

    if use_posterior
        # Compute IRF for each posterior draw
        posterior_draws = model.metadata.posterior_draws
        n_draws = length(posterior_draws)
        irf_draws = zeros(T, n_draws, horizon+1, n_vars(model), n_vars(model))

        for (i, draw) in enumerate(posterior_draws)
            # Build model from this draw
            model_i = build_model_from_draw(draw, model)
            P_i = rotation_matrix(model_i, id)
            irf_draws[i, :, :, :] = compute_irf_point(model_i, P_i, horizon)
        end

        # Compute posterior mean and credible bands
        irf_mean = dropdims(mean(irf_draws; dims=1); dims=1)
        lower, upper = compute_credible_bands(irf_draws, coverage)

        return BayesianIRFResult{T}(irf_mean, irf_draws, lower, upper, coverage,
                                    (identification=id, horizon=horizon))
    else
        # Use point estimate (posterior mean) with delta method
        # Falls back to standard IRFResult
        P = rotation_matrix(model, id)
        irf_point = compute_irf_point(model, P, horizon)
        # ... standard inference
        return IRFResult{T}(...)
    end
end

# Accessor methods
horizon(irf::BayesianIRFResult) = size(irf.irf_mean, 1) - 1
n_vars(irf::BayesianIRFResult) = size(irf.irf_mean, 2)
```

### Key Translation Points

1. **Prior specification** → `BayesianVAR{P<:AbstractPrior}` (type parameter)
2. **MCMC draws** → Store in `model.metadata.posterior_draws`
3. **Credible intervals** → `BayesianIRFResult` with `irf_draws`
4. **Random number generation** → Always accept `rng` parameter
5. **Dispatch on model type** → `irf(model::VARModel{T, BayesianVAR{P}}, ...)`

---

## Common Patterns in This Package

### Pattern: Companion Form Construction

```julia
function companion_form(lags::Array{T,3}) where T
    n_vars, _, n_lags = size(lags)
    m = n_vars * n_lags
    F = zeros(T, m, m)

    # First n_vars rows: VAR lag coefficients
    for lag in 1:n_lags
        F[1:n_vars, ((lag-1)*n_vars + 1):(lag*n_vars)] = lags[:, :, lag]
    end

    # Remaining rows: identity matrix (state transition)
    if n_lags > 1
        F[(n_vars+1):m, 1:(m-n_vars)] = I(m - n_vars)
    end

    return F
end
```

### Pattern: MA Representation from Companion

```julia
function compute_ma_matrices(F::Matrix{T}, horizon::Int, n_vars::Int, n_lags::Int) where T
    Φ = zeros(T, n_vars, n_vars, horizon + 1)

    # Impact response (horizon 0)
    Φ[:, :, 1] = I(n_vars)

    # Subsequent horizons via companion powers
    F_power = copy(F)
    J = [I(n_vars) zeros(T, n_vars, n_vars * (n_lags - 1))]

    for h in 1:horizon
        Φ[:, :, h + 1] = J * F_power * J'
        F_power = F_power * F
    end

    return Φ
end
```

### Pattern: Bootstrap with RNG

```julia
function bootstrap_irf(model::VARModel{T}, id::AbstractIdentification,
                      horizon::Int, reps::Int;
                      method::Symbol=:wild,
                      rng::AbstractRNG=Random.default_rng()) where T

    # Preallocate
    irf_boot = zeros(T, reps, horizon+1, n_vars(model), n_vars(model))

    for b in 1:reps
        # Resample with provided RNG
        ε_boot = resample_residuals(model.residuals, method, rng)

        # Simulate, re-estimate, re-identify
        Y_boot = simulate_var(model, ε_boot, ...)
        model_boot = estimate(typeof(model.spec), Y_boot, n_lags(model))
        P_boot = rotation_matrix(model_boot, id)

        # Compute IRF
        irf_boot[b, :, :, :] = compute_irf_point(model_boot, P_boot, horizon)
    end

    return irf_boot
end
```

### Pattern: Constraint Application

```julia
function apply_constraint!(A::Matrix{T}, c::ZeroConstraint,
                          names::Vector{Symbol}, n_lags::Int) where T
    var_idx = findfirst(==(c.variable), names)

    for regressor in c.regressors
        reg_idx = findfirst(==(regressor), names)

        lags_to_zero = isempty(c.lags) ? (1:n_lags) : c.lags

        for lag in lags_to_zero
            col_idx = 1 + (lag - 1) * length(names) + reg_idx
            A[col_idx, var_idx] = zero(T)
        end
    end
end
```

---

## Performance Guidelines

### Type Stability Verification

Before committing new code, verify type stability on critical paths:

```julia
using Test

@testset "Type stability" begin
    var = estimate(OLSVAR, Y, 4)

    # Check estimation
    @inferred estimate(OLSVAR, Y, 4)

    # Check identification
    @inferred rotation_matrix(var, CholeskyID())

    # Check IRF computation
    @inferred compute_irf_point(var, P, 24)
end
```

### Allocation Profiling

Profile before optimizing:

```julia
using BenchmarkTools

# Benchmark critical functions
var = estimate(OLSVAR, Y, 4)
P = rotation_matrix(var, CholeskyID())

@btime compute_irf_point($var, $P, 24)  # Note the $ for interpolation

# Check allocations
@benchmark compute_irf_point($var, $P, 24) samples=100
```

**Optimization hierarchy**:
1. Ensure type stability (biggest impact)
2. Reduce allocations in hot loops
3. Use BLAS operations (mul!, etc.)
4. Consider views for slicing
5. Profile before micro-optimizing

---

## Testing Guidelines

### Test Organization Principles

**Critical**: Follow these principles to maintain test quality and isolation.

#### 1. Top-Level runtests.jl

The high-level `test/runtests.jl` should **only shuttle to other test files**. Do not write tests directly in this file.

```julia
# test/runtests.jl
using Test
using SafeTestsets

@time @safetestset "VAR Estimation" include("test_estimation.jl")
@time @safetestset "Identification Schemes" include("test_identification.jl")
@time @safetestset "IRF Computation" include("test_irfs.jl")
@time @safetestset "Bootstrap Inference" include("test_bootstrap.jl")
@time @safetestset "Constraints" include("test_constraints.jl")
@time @safetestset "Sign Restrictions" include("test_sign_restrictions.jl")
```

#### 2. Use @safetestset, Not @testset

**Always use `@safetestset`** to avoid leaking variables (especially functions) between test blocks.

```julia
# ✓ CORRECT: Use @safetestset for full isolation
@safetestset "Feature Tests" begin
    using MacroEconometricTools
    using Test
    using StableRNGs

    # Test code here - fully isolated
end

# ✗ WRONG: @testset can leak function definitions
@testset "Feature Tests" begin
    # Functions defined here can leak to other testsets!
end
```

**Why**: Standard `@testset` does not fully enclose all defined values. Functions defined in a `@testset` can "leak" to other testsets, causing hard-to-debug interactions.

#### 3. One-Line Test Includes

Test includes should be written in **one line** with `@time` and `@safetestset`:

```julia
# ✓ CORRECT: One-line include with timing
@time @safetestset "Jacobian Tests" include("interface/jacobian_tests.jl")
@time @safetestset "IRF Tests" include("test_irfs.jl")

# ✗ WRONG: Multi-line or without @safetestset
@testset "IRF Tests" begin
    include("test_irfs.jl")
end
```

#### 4. Every Test Script is Fully Reproducible in Isolation

**Each test file must be independently runnable**. You should be able to copy-paste any test script and run it standalone.

```julia
# test/test_estimation.jl
# This file can be run standalone: julia test/test_estimation.jl

using MacroEconometricTools
using Test
using StableRNGs
using LinearAlgebra

@testset "VAR Estimation" begin
    # All dependencies loaded above
    # All setup code is self-contained

    @testset "Basic OLS estimation" begin
        rng = StableRNG(123)
        Y = randn(rng, 100, 3)
        var = estimate(OLSVAR, Y, 4)

        @test n_vars(var) == 3
        @test n_lags(var) == 4
    end

    @testset "Estimation with constraints" begin
        # Fully reproducible test code
    end
end
```

**Test**: Can you run `julia test/test_estimation.jl` directly? If not, the test script needs fixing.

#### 5. Group Test Scripts by Category

Organize tests into logical categories:

```
test/
├── runtests.jl                 # Top-level shuttle
├── test_basic.jl               # Smoke tests (fast, essential)
├── test_estimation.jl          # VAR estimation tests
├── test_identification.jl      # Identification schemes
├── test_irfs.jl                # IRF computation
├── test_bootstrap.jl           # Bootstrap inference
├── test_constraints.jl         # Constraint system
├── test_sign_restrictions.jl   # Sign restrictions
├── test_forecasting.jl         # Forecasting
└── test_utils.jl               # Utility functions
```

**Categories**:
- **Core functionality**: Estimation, identification, IRFs
- **Inference**: Bootstrap, delta method
- **Advanced features**: Constraints, sign restrictions, forecasting
- **Utilities**: Helper functions, data handling

### Test Structure Pattern

Each test file should follow this structure:

```julia
# test/test_feature.jl
using MacroEconometricTools
using Test
using StableRNGs
using LinearAlgebra  # Add all dependencies

@testset "Feature Name" begin
    # Setup (if needed)
    rng = StableRNG(123)
    Y = randn(rng, 100, 3)

    @testset "Basic functionality" begin
        # Test that it works
        result = some_function(input)
        @test result isa ExpectedType
    end

    @testset "Mathematical properties" begin
        # Test correctness (e.g., P*P' ≈ Σ)
        var = estimate(OLSVAR, Y, 4)
        P = rotation_matrix(var, CholeskyID())
        @test P * P' ≈ var.Σ
    end

    @testset "Type stability" begin
        # Test with @inferred
        @inferred estimate(OLSVAR, Y, 4)
    end

    @testset "Edge cases" begin
        # Test boundary conditions
        @test_throws ArgumentError estimate(OLSVAR, Y, 0)
    end

    @testset "Reproducibility" begin
        # Test RNG reproducibility
        rng1 = StableRNG(456)
        result1 = random_function(input; rng=rng1)

        rng2 = StableRNG(456)
        result2 = random_function(input; rng=rng2)

        @test result1 ≈ result2
    end
end
```

### Test Reproducibility

**Always use StableRNGs** for any random number generation in tests:

```julia
using StableRNGs

@testset "Bootstrap reproducibility" begin
    rng = StableRNG(123)
    Y = randn(rng, 100, 3)  # Use rng for data generation
    var = estimate(OLSVAR, Y, 4)

    # Test that same seed gives same results
    rng1 = StableRNG(123)
    irf1 = irf(var, CholeskyID(); bootstrap_reps=100, rng=rng1)

    rng2 = StableRNG(123)
    irf2 = irf(var, CholeskyID(); bootstrap_reps=100, rng=rng2)

    @test irf1.irf ≈ irf2.irf  # Should be exactly equal
end
```

### Example: Complete Test File

```julia
# test/test_estimation.jl
# Fully reproducible standalone test file

using MacroEconometricTools
using Test
using StableRNGs
using LinearAlgebra

@testset "VAR Estimation" begin
    # Setup
    rng = StableRNG(123)
    Y = randn(rng, 100, 3)
    p = 4

    @testset "Basic OLS estimation" begin
        var = estimate(OLSVAR, Y, p)

        @test n_vars(var) == 3
        @test n_lags(var) == 4
        @test n_obs(var) == size(Y, 1) - p
        @test raw_nobs(var) == size(Y, 1)
    end

    @testset "Estimation with variable names" begin
        names = [:GDP, :Inflation, :Rate]
        var = estimate(OLSVAR, Y, p; names=names)

        @test varnames(var) == names
    end

    @testset "Coefficient structure" begin
        var = estimate(OLSVAR, Y, p)
        coefs = coef(var)

        @test size(coefs.intercept) == (3,)
        @test size(coefs.lags) == (3, 3, 4)
    end

    @testset "Type stability" begin
        @inferred estimate(OLSVAR, Y, p)
        @inferred coef(estimate(OLSVAR, Y, p))
    end

    @testset "Mathematical properties" begin
        var = estimate(OLSVAR, Y, p)
        Σ = vcov(var)

        # Covariance matrix is symmetric
        @test Σ ≈ Σ'

        # Covariance matrix is positive semi-definite
        @test all(eigvals(Σ) .>= -1e-10)
    end

    @testset "Edge cases" begin
        # Zero lags
        @test_throws ArgumentError estimate(OLSVAR, Y, 0)

        # Negative lags
        @test_throws ArgumentError estimate(OLSVAR, Y, -1)

        # More lags than observations
        @test_throws ArgumentError estimate(OLSVAR, Y, 200)
    end
end
```

### Running Tests

```bash
# Run all tests
julia --project=. -e 'using Pkg; Pkg.test()'

# Run specific test file in isolation
julia --project=. test/test_estimation.jl

# Run with coverage
julia --project=. --code-coverage=user -e 'using Pkg; Pkg.test()'
```

### Test Performance Expectations

- **Smoke tests** (`test_basic.jl`): < 5 seconds
- **Unit tests**: < 30 seconds per file
- **Integration tests**: < 2 minutes per file
- **Full test suite**: < 5 minutes

If tests are slower, consider:
- Reducing bootstrap replications in tests
- Using smaller datasets
- Splitting slow tests into separate optional file

---

## Documentation Requirements

### Every Exported Function Needs

1. **Docstring** with examples:
```julia
"""
    rotation_matrix(model::VARModel, id::AbstractIdentification)

Compute the structural impact matrix P that relates reduced-form shocks to structural shocks.

The matrix P satisfies PP' = Σ, where Σ is the residual covariance matrix.

# Arguments
- `model::VARModel`: Estimated VAR model
- `id::AbstractIdentification`: Identification scheme (e.g., `CholeskyID()`, `SignRestriction(...)`)

# Returns
- `Matrix{T}`: Structural impact matrix P

# Examples
```julia
var = estimate(OLSVAR, Y, 4)
P = rotation_matrix(var, CholeskyID())
@test P * P' ≈ var.Σ
```

# See Also
- [`irf`](@ref): Compute impulse response functions
- [`CholeskyID`](@ref), [`SignRestriction`](@ref): Identification schemes
"""
```

2. **Tutorial section** (for major features)
3. **Entry in QUICK_REFERENCE.md**
4. **Update to RECENT_CHANGES.md**

---

## Commit Guidelines

### Commit Message Format

```
<type>: <short summary>

<optional detailed description>

<optional breaking changes note>
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `refactor`: Code restructuring (no behavior change)
- `docs`: Documentation only
- `test`: Test additions/changes
- `perf`: Performance improvement
- `style`: Code style (formatting, naming)

**Examples**:
```
feat: Add BayesianVAR with Minnesota prior

- Implement BayesianVAR{P<:AbstractPrior} specification
- Add MinnesotaPrior and NormalWishartPrior types
- Implement posterior sampler with Gibbs steps
- Add BayesianIRFResult for credible intervals
- Tests with StableRNG for reproducibility

docs: Update QUICK_REFERENCE with Bayesian VAR examples
```

### Before Committing

- [ ] Run tests: `julia --project=. -e 'using Pkg; Pkg.test()'`
- [ ] Verify test files are independently runnable
- [ ] Ensure new tests use `@safetestset` in runtests.jl
- [ ] Check type stability on new critical paths with `@inferred`
- [ ] Verify StableRNG usage for all randomness in tests
- [ ] Update docstrings for any changed functions
- [ ] Update RECENT_CHANGES.md if API changes
- [ ] Update QUICK_REFERENCE.md if adding user-facing features

---

## Working with External Code/Papers

When translating code from papers or other packages:

### 1. Understand the Mathematical Concept

Read the paper/documentation to understand:
- What is being estimated?
- What are the key parameters?
- What are the assumptions?

### 2. Map to Our Type System

- Estimation method → `AbstractVARSpec` subtype
- Prior/regularization → Type parameter or field
- Results → Appropriate result type (reuse existing or create new)

### 3. Translate Idiomatically

Don't just copy-paste. Translate to our patterns:

```julia
# Original code (hypothetical):
function estimate_bvar(Y, lags, lambda1, lambda2, method="minnesota")
    if method == "minnesota"
        prior = setup_minnesota_prior(lambda1, lambda2)
    elseif method == "normalwishart"
        prior = setup_nw_prior(lambda1, lambda2)
    end
    # ... estimation
end

# Our translation:
struct MinnesotaPrior{T} <: AbstractPrior
    λ1::T
    λ2::T
end

function estimate(spec::BayesianVAR{MinnesotaPrior{T}}, Y::Matrix{T}, p::Int;
                 kwargs...) where T
    prior = spec.prior
    # ... estimation using prior.λ1, prior.λ2
end

# Usage
prior = MinnesotaPrior(0.2, 0.5)
bvar = estimate(BayesianVAR(prior), Y, 4)
```

### 4. Maintain Mathematical Correctness

- Verify against test cases from paper/package
- Test mathematical properties (e.g., posterior mean is unbiased estimator)
- Check edge cases

### 5. Document the Source

```julia
"""
    estimate(spec::BayesianVAR{MinnesotaPrior}, ...)

Bayesian VAR estimation with Minnesota prior.

Implementation follows Litterman (1986) and Bańbura et al. (2010).

# References
- Litterman, R. (1986). Forecasting with Bayesian Vector Autoregressions...
- Bańbura, M., Giannone, D., & Reichlin, L. (2010). Large Bayesian VARs...
"""
```

---

## Final Reminders

### Always
- ✓ Use type-based dispatch
- ✓ Keep types fully parametric
- ✓ Provide accessor functions
- ✓ Use NaN for missing numeric data
- ✓ Accept `rng` parameter for randomness
- ✓ Test with StableRNG for reproducibility
- ✓ Write docstrings with examples
- ✓ Update documentation

### Never
- ✗ Use method flags instead of dispatch
- ✗ Put `Union{T, Nothing}` in struct fields (use separate types)
- ✗ Use `Missing` for numeric data
- ✗ Use `rand()` without RNG parameter
- ✗ Expose struct internals as API
- ✗ Commit without running tests
- ✗ Add features without documentation

### When in Doubt
1. Check existing code in `src/var/` for patterns
2. Consult `docs/IMPLEMENTATION_PHILOSOPHY.md`
3. Ask yourself: "Does this follow our type-based dispatch philosophy?"
4. Verify type stability with `@inferred`

---

## Quick Reference: File Locations

- **Type definitions**: `src/types.jl` (define ALL structs here first)
- **VAR estimation**: `src/var/estimation.jl`
- **Identification**: `src/var/identification.jl`
- **IRFs**: `src/var/irfs.jl`
- **Constraints**: `src/var/constraints.jl`
- **Bootstrap**: `src/bootstrap.jl`
- **Utilities**: `src/utils.jl` (lag, companion, MA matrices)
- **Exports**: `src/MacroEconometricTools.jl`
- **Tests**: `test/test_*.jl`
- **Tutorials**: `docs/src/tutorials/`

---

**Remember**: This package prioritizes **clarity, correctness, and type stability** over cleverness. When integrating new code, translate it to our patterns rather than adapting our patterns to the code.

For comprehensive design philosophy and detailed examples, always refer to `docs/IMPLEMENTATION_PHILOSOPHY.md`.
