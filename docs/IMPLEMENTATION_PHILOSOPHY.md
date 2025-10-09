# Implementation Philosophy

## Purpose of This Document

This document explains the **design philosophy** and **implementation patterns** of MacroEconometricTools.jl. It is intended to be read by AI assistants (Claude) and developers to quickly understand **how** and **why** the code is structured the way it is, enabling effective collaboration and extension of the package.

---

## Core Design Principles

### 1. Type-Based Dispatch Over Method Flags

**Philosophy**: Use Julia's type system and multiple dispatch as the primary API design pattern.

**Example**:
```julia
# ✓ GOOD: Type-based dispatch
var = fit(OLSVAR, Y, p)
bayesian_var = fit(BayesianVAR(prior), Y, p)

# ✗ AVOID: Method flags
var = fit(Y, p; method=:ols)  # Type-unstable, harder to extend
```

**Rationale**:
- **Type stability**: Compiler knows return type at compile time
- **Extensibility**: Adding new methods doesn't modify existing code
- **Self-documenting**: Type names communicate intent
- **Performance**: Specialized code generation per concrete type

**Application throughout package**:
- Estimation: `fit(::AbstractVARSpec, ...)`
- Identification: `rotation_matrix(::VARModel, ::AbstractIdentification)`
- Constraints: `apply_constraint!(::AbstractConstraint, ...)`

### 2. Parametric Types for Performance

**Philosophy**: All core data structures use concrete parametric types, avoiding abstract containers.

**Example**:
```julia
# ✓ GOOD: Parametric types
struct VARModel{T<:AbstractFloat, S<:AbstractVARSpec}
    Y::Matrix{T}                    # Concrete element type
    coefficients::VARCoefficients{T}
    residuals::Matrix{T}
    Σ::Symmetric{T,Matrix{T}}
    # ...
end

# ✗ AVOID: Abstract containers
struct VARModel
    Y::Matrix{AbstractFloat}  # Type-unstable field access
    residuals::AbstractMatrix  # Runtime dispatch on every access
end
```

**Rationale**:
- **Type inference**: Compiler can infer field types from outer type parameter
- **Memory layout**: Predictable, cache-friendly memory access
- **No runtime dispatch**: Field accesses compile to direct memory loads
- **SIMD**: Enables vectorization in numerical operations

**Key pattern**: Propagate type parameters through the call stack:
```julia
function irf(model::VARModel{T}, id::AbstractIdentification; ...) where T
    P = rotation_matrix(model, id)  # Returns Matrix{T}
    irf_array = zeros(T, horizon+1, n_vars, n_vars)  # Concrete type
    return IRFResult{T}(irf_array, ...)  # Propagate T
end
```

### 3. NaN Over Missing for Numeric Data

**Philosophy**: Use `NaN` (not-a-number) to represent unavailable numeric values instead of `Missing`.

**Example**:
```julia
# ✓ GOOD: NaN for unavailable data
function lag(x::Vector{Float64}, k::Int)
    result = similar(x)
    result[1:k] .= NaN  # First k values unavailable
    result[k+1:end] = x[1:end-k]
    return result  # Vector{Float64}
end

# ✗ AVOID: Missing values
function lag(x::Vector{Float64}, k::Int)
    result = Vector{Union{Missing, Float64}}(undef, length(x))
    result[1:k] .= missing
    # Return type: Vector{Union{Missing, Float64}} - type unstable!
end
```

**Rationale**:
- **Type stability**: `Vector{Float64}` not `Vector{Union{Missing, Float64}}`
- **Performance**: No union splitting overhead
- **Standard practice**: NumPy, MATLAB use NaN for numeric missing data
- **Numerical operations**: Most functions naturally propagate NaN

**Trade-off**: Cannot distinguish "truly missing" from "computed to be NaN", but in econometric context this is rarely needed.

### 4. Separate Types for Different Data Structures

**Philosophy**: When different use cases require fundamentally different data structures, use separate concrete types with a shared abstract parent.

**Example - IRF Results**:

```julia
# Abstract parent for polymorphic interface
abstract type AbstractIRFResult{T<:AbstractFloat} end

# Point-identified IRFs (Cholesky, IV)
struct IRFResult{T} <: AbstractIRFResult{T}
    irf::Array{T,3}           # Single IRF: (horizon+1, n_vars, n_shocks)
    stderr::Array{T,3}        # Standard errors
    lower::Vector{Array{T,3}}  # Confidence bands
    upper::Vector{Array{T,3}}
    # ...
end

# Set-identified IRFs (sign restrictions)
struct SignRestrictedIRFResult{T} <: AbstractIRFResult{T}
    irf_median::Array{T,3}       # Median: (horizon+1, n_vars, n_shocks)
    irf_draws::Array{T,4}        # All draws: (n_draws, horizon+1, n_vars, n_shocks)
    rotation_matrices::Vector{Matrix{T}}  # Multiple valid P matrices
    # ...
end

# Generic interface via dispatch
horizon(irf::IRFResult) = size(irf.irf, 1) - 1
horizon(irf::SignRestrictedIRFResult) = size(irf.irf_median, 1) - 1
```

**Why NOT a single parameterized struct?**

```julia
# ✗ CONSIDERED BUT REJECTED:
struct IRFResult{T, ID<:AbstractIdentification}
    irf::Array{T,3}                      # Used by Cholesky
    irf_draws::Union{Array{T,4}, Nothing}  # Only used by sign restrictions
    stderr::Union{Array{T,3}, Nothing}    # Only used by Cholesky
    # ... many Union{T, Nothing} fields
end
```

**Problems with parameterized approach**:
- **Type instability**: `Union{T, Nothing}` fields require runtime type checks
- **Memory waste**: Every instance allocates space for unused fields
- **Unclear semantics**: Which fields are valid for which ID type?
- **Error-prone**: Easy to access `nothing` fields by mistake

**Rationale for separate types**:
- **Type stability**: All fields have concrete types
- **Memory efficiency**: Only store relevant data
- **Clear semantics**: Each type represents a well-defined concept
- **Extensibility**: Add new result types (e.g., `BayesianIRFResult`) without modifying existing types

**When to use this pattern**:
- Data structures differ significantly between use cases
- Fields would be `Union{T, Nothing}` in a unified approach
- Each variant has distinct semantic meaning

**When to use parameterization instead**:
- All variants use the same fields
- Type parameter only affects computation, not storage
- Example: `VARModel{T, S}` where both Float32 and Float64 variants have identical structure

### 5. Small, Composable Functions

**Philosophy**: Write small, focused functions that do one thing well. Compose them to build complex functionality.

**Example**:
```julia
# Small, focused functions
function compute_companion(lags::Array{T,3}) where T
    # Single responsibility: build companion matrix
end

function compute_ma_matrices(F::Matrix{T}, horizon::Int) where T
    # Single responsibility: MA representation
end

function compute_irf_point(model::VARModel{T}, P::Matrix{T}, horizon::Int) where T
    # Compose the above
    F = model.companion
    Φ = compute_ma_matrices(F, horizon)
    # Compute IRF_h = Φ_h * P
end
```

**Benefits**:
- **Testability**: Easy to unit test each component
- **Reusability**: Functions compose in different contexts
- **Readability**: Each function has clear purpose
- **Debugging**: Narrow scope for errors

**Anti-pattern to avoid**:
```julia
# ✗ AVOID: Monolithic functions
function compute_everything(model, identification, horizon, inference, ...)
    # 500 lines doing VAR estimation, identification, IRF computation,
    # bootstrap, plotting, and file I/O
end
```

---

## Key API Patterns

### Pattern 1: Estimation Workflow

**Standard sequence**:
```julia
# 1. Estimate VAR
var = fit(OLSVAR, Y, p; names=names)

# 2. Compute rotation matrix (identification)
P = rotation_matrix(var, CholeskyID())

# 3. Compute IRFs with inference
irf_result = irf(var, CholeskyID();
                horizon=24,
                inference=:bootstrap,
                bootstrap_reps=1000)
```

**Why `rotation_matrix()` is separate from `irf()`**:
- User might want just the impact matrix P for inspection
- Allows flexibility: `compute_irf_point(var, custom_P, horizon)`
- Clear separation: identification vs IRF computation

**Why `irf()` takes `identification` instead of `P`**:
- Bootstrap needs to re-identify for each draw
- Encapsulates identification logic (e.g., random draws for sign restrictions)
- Convenience: most common use case

**Name change: `identify()` → `rotation_matrix()`**:
- **Old name**: `identify()` was ambiguous (identify what?)
- **New name**: `rotation_matrix()` is precise (returns the matrix P that rotates Cholesky to structural shocks)
- This is a **breaking change** but improves clarity

### Pattern 2: Sign Restrictions (Set Identification)

**Conceptual difference**: Sign restrictions produce multiple valid IRFs (set-identified).

**API design**:
```julia
# For sign restrictions, irf() finds multiple draws
irf_result = irf(var, SignRestriction(...);
                n_draws=1000,        # How many valid rotations to find
                max_attempts=10000,  # Max attempts per draw
                horizon=48)

# Returns SignRestrictedIRFResult with:
# - irf_median: pointwise median across draws
# - irf_draws: all IRF draws (set identification)
# - rotation_matrices: all valid P matrices
```

**Alternative design considered**:
```julia
# ✗ REJECTED: Separate function
irfs = irf_set_identified(var, SignRestriction(...); n_draws=1000)
```

**Why unified `irf()` function?**
- Dispatch handles the difference automatically
- User doesn't need to remember different function names
- Return type is statically known from identification type
- Consistent with Julia philosophy (one function, many methods)

### Pattern 3: Accessor Functions

**Philosophy**: Use accessor functions, not direct field access.

**Example**:
```julia
# ✓ GOOD: Use accessors
n = n_vars(model)
h = horizon(irf_result)

# ✗ AVOID: Direct field access
n = size(model.Y, 2)  # Fragile if internal structure changes
h = size(irf_result.irf, 1) - 1  # Relies on implementation detail
```

**Benefits**:
- **Encapsulation**: Internal structure can change
- **Dispatch**: Works polymorphically on AbstractIRFResult
- **Documentation**: Accessor names are searchable
- **Validation**: Can add checks/warnings

**Accessor naming convention**:
- `n_vars(x)` - Number of variables
- `n_lags(x)` - Number of lags
- `n_obs(x)` - Effective observations (after losing lags)
- `raw_nobs(x)` - Total observations before lags
- `varnames(x)` - Variable names as Vector{Symbol}
- `horizon(x)` - IRF horizon

### Pattern 4: Mutating vs Non-Mutating Functions

**Convention**: Follow Julia standard library conventions.

```julia
# Non-mutating (returns new array)
function create_lags(X::Matrix{T}, p::Int) where T
    # Allocates new matrix
    return lagged
end

# Mutating (modifies in-place, marked with !)
function apply_constraint!(A::Matrix, constraint::ZeroConstraint, ...)
    # Modifies A directly
    A[idx] = 0.0
    return nothing  # Or return A for chaining
end
```

**When to use mutating**:
- Performance-critical inner loops
- Pre-allocated arrays (e.g., bootstrap iterations)
- In-place constraint application

**When to use non-mutating**:
- Top-level API functions (clearer, safer)
- When allocation overhead is negligible
- When immutability helps reasoning

**Current state**: Package currently favors non-mutating API for clarity. Mutating internals can be added when profiling identifies bottlenecks.

---

## Architecture Patterns

### Module Organization

```
src/
├── MacroEconometricTools.jl  # Main module, exports
├── types.jl                   # All struct definitions
├── utils.jl                   # Generic utilities (lag, companion, etc.)
├── bootstrap.jl               # Bootstrap infrastructure
├── parallel.jl                # Distributed computing support
├── plots_recipes.jl           # RecipesBase recipes for Plots.jl
├── var/
│   ├── estimation.jl          # VAR estimation (OLS, constraints)
│   ├── identification.jl      # Identification schemes (Cholesky, sign)
│   ├── irfs.jl                # IRF computation and inference
│   └── constraints.jl         # Constraint types and application
└── ext/                       # Package extensions
    └── MacroEconometricToolsMakieExt.jl  # Makie plotting
```

**Rationale**:
- **types.jl first**: All structs defined before methods (avoids circular dependencies)
- **var/ subdirectory**: VAR-specific code isolated
- **Separate concerns**: Estimation ≠ identification ≠ inference
- **Extensions**: Optional dependencies (Makie) in ext/

### Type Hierarchy Design

**Abstract types define interfaces**:
```julia
abstract type AbstractVARSpec end
abstract type AbstractIdentification end
abstract type AbstractConstraint end
abstract type AbstractIRFResult{T<:AbstractFloat} end
```

**Concrete types implement behavior**:
```julia
struct OLSVAR <: AbstractVARSpec end  # Empty: pure dispatch
struct CholeskyID <: AbstractIdentification
    ordering::Union{Vector{Symbol}, Nothing}
end
```

**Methods dispatch on concrete types**:
```julia
function fit(spec::OLSVAR, Y::Matrix{T}, p::Int; ...) where T
    # OLS-specific estimation
end

function rotation_matrix(model::VARModel{T}, id::CholeskyID) where T
    # Cholesky-specific identification
end
```

**Benefits**:
- **Open for extension**: Add new concrete types without modifying existing code
- **Closed for modification**: Abstract interface remains stable
- **Type stability**: Compiler generates specialized code per concrete type

### Constraint System Architecture

**Problem**: Allow flexible coefficient restrictions during estimation.

**Solution**: Abstract constraint type + multiple concrete implementations.

```julia
abstract type AbstractConstraint end

struct ZeroConstraint <: AbstractConstraint
    variable::Symbol
    regressors::Vector{Symbol}
    lags::Vector{Int}  # Empty means all lags
end

struct FixedConstraint <: AbstractConstraint
    variable::Symbol
    regressor::Symbol
    lag::Int
    value::Float64
end

struct BlockExogeneity <: AbstractConstraint
    from::Vector{Symbol}  # Variables in 'from' block
    to::Vector{Symbol}    # ... don't affect 'to' block
end
```

**Application in estimation**:
```julia
function fit(spec::OLSVAR, Y, p; constraints=nothing, ...)
    if constraints === nothing
        # Standard OLS: equation-by-equation
        A = unconstrained_ols(X, Y)
    else
        # Dispatch based on constraint types
        A = constrained_ols(X, Y, constraints, names, p, n_vars)
    end
end

function constrained_ols(X, Y, constraints, ...)
    only_zero = all(c -> c isa Union{ZeroConstraint, BlockExogeneity}, constraints)

    if only_zero
        # Selection method: estimate with subset of regressors
        return constrained_ols_selection(X, Y, constraints, ...)
    else
        # General method: handles FixedConstraint too
        return constrained_ols_general(X, Y, constraints, ...)
    end
end
```

**Key insight**: VAR equations are independent under zero constraints, so we can estimate equation-by-equation with restricted regressors.

**Extensibility**: To add new constraint type:
1. Define `struct NewConstraint <: AbstractConstraint`
2. Add case to `constrained_ols` or implement `apply_constraint!` method
3. Update constraint handling logic

---

## Performance Patterns

### Type Stability Checklist

When writing a new function, ensure:

1. **Return type is determined by input types**:
```julia
# ✓ Type-stable
function foo(x::Vector{Float64})
    return 2.0 .* x  # Always returns Vector{Float64}
end

# ✗ Type-unstable
function foo(x::Vector{Float64}, flag::Bool)
    if flag
        return 2.0 .* x  # Vector{Float64}
    else
        return nothing   # Nothing
    end
    # Return type: Union{Vector{Float64}, Nothing}
end
```

2. **No changes to variable types within function**:
```julia
# ✓ Type-stable
function bar(x::Float64)
    y = 2.0 * x     # Float64
    z = y + 1.0     # Float64
    return z
end

# ✗ Type-unstable
function bar(x::Float64)
    y = 2.0 * x     # Float64
    y = [y, y]      # Now Vector{Float64} - type changed!
    return y
end
```

3. **Container element types are concrete**:
```julia
# ✓ Concrete element type
results = Vector{Matrix{Float64}}(undef, n_bootstrap)

# ✗ Abstract element type
results = Vector{AbstractMatrix}(undef, n_bootstrap)  # Runtime dispatch
```

**Verification**: Use `@code_warntype` to check type stability.

### Allocation Patterns

**General principle**: Allocate at top level, reuse in inner loops.

**Example - Bootstrap**:
```julia
function bootstrap_irf(model, id, horizon, reps; ...)
    n = n_vars(model)

    # Allocate once
    irf_boot = zeros(T, reps, horizon+1, n, n)

    for b in 1:reps
        # Reuse preallocated slice
        compute_irf_point!(view(irf_boot, b, :, :, :), model_boot, P_boot, horizon)
    end

    return irf_boot
end
```

**When to preallocate**:
- Loop bodies (if allocation shows up in profiler)
- Large arrays used repeatedly
- Bootstrap/simulation contexts

**When NOT to preallocate**:
- Top-level API (readability > micro-optimization)
- Small arrays (allocation overhead negligible)
- Until profiling shows it's a bottleneck

**Current state**: Package favors clarity. Add preallocation when benchmarks show benefit.

### Broadcasting and Fusion

**Use dot syntax for element-wise operations**:
```julia
# ✓ Fused broadcast (single loop, no temporaries)
y .= 2.0 .* x .+ 1.0

# ✗ Separate operations (two allocations)
temp = 2.0 * x
y = temp + 1.0
```

**Benefits**:
- Compiler fuses multiple operations into single loop
- No intermediate allocations
- Better cache behavior

**Pattern in IRF computation**:
```julia
# Update IRF in-place
for h in 1:horizon
    mul!(view(irf_array, h+1, :, :), Φ[:, :, h+1], P)
end
```

---

## Extending the Package

### Adding a New Identification Scheme

**Steps**:

1. **Define identification type**:
```julia
struct MyIdentification <: AbstractIdentification
    param::Float64
end
```

2. **Implement rotation matrix method**:
```julia
function rotation_matrix(model::VARModel{T}, id::MyIdentification) where T
    Σ = model.Σ
    # Your identification logic
    # Must satisfy: P * P' = Σ
    return P::Matrix{T}
end
```

3. **Optional: Special IRF handling**:
```julia
# Only if standard IRF computation doesn't apply
function irf(model::VARModel{T}, id::MyIdentification; kwargs...) where T
    # Custom IRF computation
end
```

4. **Add tests**:
```julia
@testset "MyIdentification" begin
    var = fit(OLSVAR, Y, 4)
    P = rotation_matrix(var, MyIdentification(param=0.5))
    @test P * P' ≈ var.Σ
end
```

5. **Document**:
- Add example to docs/src/tutorials/
- Explain economic interpretation
- Show comparison to other schemes

### Adding a New Result Type

**When**: If a new identification produces fundamentally different output structure.

**Steps**:

1. **Define result type**:
```julia
struct MyIRFResult{T} <: AbstractIRFResult{T}
    custom_field::Array{T,4}
    # ... other fields specific to this identification
    metadata::NamedTuple
end
```

2. **Implement accessor methods**:
```julia
horizon(irf::MyIRFResult) = size(irf.custom_field, 2) - 1
n_vars(irf::MyIRFResult) = size(irf.custom_field, 3)
# ... etc
```

3. **Add plotting recipe** (optional):
```julia
@recipe function f(irf::MyIRFResult; ...)
    # Custom plotting logic
end
```

4. **Implement `cumulative_irf`** (if applicable):
```julia
function cumulative_irf(irf::MyIRFResult{T}) where T
    # ...
    return MyIRFResult{T}(...)
end
```

### Adding a New Constraint Type

**Steps**:

1. **Define constraint**:
```julia
struct MyConstraint <: AbstractConstraint
    variable::Symbol
    param::Float64
end
```

2. **Implement constraint logic**:

Option A: Update `constrained_ols`:
```julia
function constrained_ols(X, Y, constraints, ...)
    has_my_constraint = any(c -> c isa MyConstraint, constraints)
    if has_my_constraint
        return constrained_ols_with_my_constraint(X, Y, constraints, ...)
    end
    # ... existing logic
end
```

Option B: Implement `apply_constraint!`:
```julia
function apply_constraint!(A::Matrix, c::MyConstraint, names, n_lags, n_vars)
    # Modify coefficient matrix A in-place
end
```

3. **Add tests and documentation**.

---

## Common Pitfalls and Solutions

### Pitfall 1: Type Instability from Union{T, Nothing}

**Problem**:
```julia
struct MyResult{T}
    irf::Array{T,3}
    stderr::Union{Array{T,3}, Nothing}  # Type-unstable!
end

function get_stderr(r::MyResult)
    return r.stderr  # Return type unclear
end
```

**Solution**: Use separate types:
```julia
abstract type MyResult{T} end

struct MyResultWithStderr{T} <: MyResult{T}
    irf::Array{T,3}
    stderr::Array{T,3}
end

struct MyResultNoStderr{T} <: MyResult{T}
    irf::Array{T,3}
end
```

### Pitfall 2: Overusing Parameterization

**Problem**: Adding type parameter for every configuration option.

```julia
# ✗ Over-parameterized
struct VARModel{T, S, HasIntercept, IsConstrained, InferenceMethod, ...}
    # Too many parameters, hard to work with
end
```

**Solution**: Use type parameters for data types, fields/metadata for configuration:
```julia
# ✓ Balanced
struct VARModel{T<:AbstractFloat, S<:AbstractVARSpec}
    # ... data fields
    metadata::NamedTuple  # Store configuration here
end
```

**Guideline**:
- Type parameter if it affects **data storage layout**
- Field/metadata if it's **configuration/options**

### Pitfall 3: Premature Optimization

**Problem**: Writing complex, hard-to-read code for hypothetical performance gains.

**Solution**:
1. Write clear, simple code first
2. Profile to find actual bottlenecks
3. Optimize hot paths only

**Example**:
```julia
# ✓ Start with this (clear)
function fit(spec::OLSVAR, Y, p; ...)
    X = create_lags(Y, p)  # Allocates
    # ... rest of estimation
end

# Only if profiling shows create_lags is a bottleneck:
function fit(spec::OLSVAR, Y, p; ...)
    X = Matrix{eltype(Y)}(undef, size(Y, 1), 1 + size(Y, 2) * p)
    create_lags!(X, Y, p)  # In-place version
end
```

### Pitfall 4: Breaking Changes Without Deprecation

**Problem**: Renaming functions without deprecation period.

**Solution**: Provide deprecated aliases:
```julia
# New API
rotation_matrix(model, id) = # ... implementation

# Deprecated alias (with warning)
function identify(model, id)
    @warn "identify() is deprecated, use rotation_matrix() instead" maxlog=1
    return rotation_matrix(model, id)
end
```

**Current state**: `identify()` → `rotation_matrix()` is a breaking change. Consider adding deprecated alias if users request it.

---

## Testing Philosophy

### Test Structure

```
test/
├── runtests.jl          # Main test runner
├── test_basic.jl        # Smoke tests (fast, essential)
├── test_estimation.jl   # VAR estimation tests
├── test_identification.jl  # Identification schemes
├── test_constraints.jl  # Constraint system
└── test_irfs.jl         # IRF computation and inference
```

### Test Levels

1. **Unit tests**: Test individual functions in isolation
```julia
@testset "lag function" begin
    x = [1.0, 2.0, 3.0, 4.0]
    x_lag1 = lag(x, 1)
    @test isnan(x_lag1[1])
    @test x_lag1[2:end] == x[1:end-1]
end
```

2. **Integration tests**: Test workflows
```julia
@testset "VAR estimation workflow" begin
    var = fit(OLSVAR, Y, 4)
    @test n_vars(var) == 3
    @test n_lags(var) == 4
    P = rotation_matrix(var, CholeskyID())
    @test P * P' ≈ var.Σ  # Identification property
end
```

3. **Regression tests**: Ensure consistency with known results
```julia
@testset "IRF values match reference" begin
    # Use fixed RNG for reproducibility
    rng = StableRNG(123)
    irf_result = irf(var, id; rng=rng)
    # Compare to stored reference values
    @test irf_result.irf ≈ reference_irf rtol=1e-10
end
```

### Testing Guidelines

- **Use fixed RNGs**: All random number generation should accept `rng` parameter
- **Test mathematical properties**: e.g., `P * P' ≈ Σ` for identification
- **Test edge cases**: Empty constraints, single variable, etc.
- **Test type stability**: `@inferred` for critical paths
- **Document test intent**: Use clear `@testset` descriptions

---

## Documentation Philosophy

### Documentation Types

1. **Docstrings**: For API reference
```julia
"""
    rotation_matrix(model::VARModel, identification::AbstractIdentification)

Compute the structural impact matrix P that relates reduced-form shocks to structural shocks.

# Arguments
- `model::VARModel`: Estimated VAR model
- `identification::AbstractIdentification`: Identification scheme

# Returns
- `Matrix{T}`: Structural impact matrix P such that PP' = Σ

# Examples
```julia
var = fit(OLSVAR, Y, 4)
P = rotation_matrix(var, CholeskyID())
```
"""
```

2. **Tutorials**: For learning workflows
- getting_started.md: Complete workflow
- sign_restrictions.md: Advanced identification
- Focus on **why** and **when**, not just **how**

3. **Technical docs**: For understanding implementation
- TECHNICAL.md: Architecture and algorithms
- IMPLEMENTATION_PHILOSOPHY.md (this file): Design principles

4. **Quick reference**: For looking up syntax
- QUICK_REFERENCE.md: Fast lookup, minimal explanation

### Documentation Principles

- **Examples that run**: All code examples should be executable
- **Explain trade-offs**: Why this design over alternatives?
- **Link concepts**: Connect math → API → implementation
- **Update with code**: Documentation and code changes together

---

## Summary: Key Takeaways for AI Assistants

When working with this codebase:

1. **Dispatch on types, not values**: Add new methods via new types, not flags
2. **Keep types concrete**: Use parametric types, avoid `Union{T, Nothing}` in structs
3. **Separate types for separate structures**: Don't force different data into one type
4. **NaN for numeric missing data**: Maintain type stability
5. **Small, composable functions**: Each function does one thing well
6. **Accessor methods over field access**: `n_vars(x)` not `size(x.Y, 2)`
7. **Profile before optimizing**: Clarity first, optimize bottlenecks
8. **Test mathematical properties**: Not just "runs without error"
9. **Document design decisions**: Explain **why**, not just **what**

**Before making changes**:
- Read relevant section of this file
- Check if pattern exists elsewhere in codebase
- Maintain consistency with existing patterns
- Add tests for new functionality
- Update documentation

**When extending**:
- Add new concrete type for new behavior
- Implement required interface methods (dispatch)
- Maintain type stability
- Follow existing naming conventions
- Add docstrings and examples

This philosophy enables fast, type-safe, extensible macroeconometric analysis in Julia.
