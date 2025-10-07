# ============================================================================
# Constraint System for VAR Estimation
# ============================================================================

"""
    AbstractConstraint

Abstract type for coefficient constraints in VAR models.
"""
abstract type AbstractConstraint end

"""
    ZeroConstraint <: AbstractConstraint

Constrain specific coefficients to zero.

# Fields
- `variable::Symbol`: Dependent variable (equation) name
- `regressors::Vector{Symbol}`: Right-hand side variables to zero out
- `lags::Vector{Int}`: Specific lags to constrain (empty = all lags)

# Examples
```julia
# Zero out Interest rate lags 1-2 in GDP equation
ZeroConstraint(:GDP, [:InterestRate], [1, 2])

# Zero out all lags of Foreign variables in Domestic equation
ZeroConstraint(:Domestic, [:Foreign], Int[])
```
"""
struct ZeroConstraint <: AbstractConstraint
    variable::Symbol
    regressors::Vector{Symbol}
    lags::Vector{Int}
end

# Convenience constructor: all lags
ZeroConstraint(var::Symbol, regs::Vector{Symbol}) = ZeroConstraint(var, regs, Int[])

"""
    FixedConstraint <: AbstractConstraint

Fix a specific coefficient to a given value.

# Fields
- `variable::Symbol`: Dependent variable name
- `regressor::Symbol`: Right-hand side variable
- `lag::Int`: Lag number
- `value::Float64`: Fixed value

# Example
```julia
# Fix GDP lag 1 coefficient to 0.5 in Inflation equation
FixedConstraint(:Inflation, :GDP, 1, 0.5)
```
"""
struct FixedConstraint <: AbstractConstraint
    variable::Symbol
    regressor::Symbol
    lag::Int
    value::Float64
end

"""
    BlockExogeneity <: AbstractConstraint

Impose block exogeneity: variables in `from` do not affect variables in `to` at any lag.

# Fields
- `from::Vector{Symbol}`: Exogenous variables
- `to::Vector{Symbol}`: Variables that are not affected

# Example
```julia
# Foreign variables don't affect domestic variables
BlockExogeneity([:ForeignGDP, :ForeignRate], [:DomesticGDP, :DomesticInflation])
```
"""
struct BlockExogeneity <: AbstractConstraint
    from::Vector{Symbol}
    to::Vector{Symbol}
end

# ============================================================================
# Constraint Application
# ============================================================================

"""
    apply_constraints!(A, constraints, varnames, n_lags)

Apply constraints to coefficient matrix `A` in-place.

# Arguments
- `A::Matrix`: Coefficient matrix (n_vars × (1 + n_vars * n_lags))
- `constraints::Vector{<:AbstractConstraint}`: Constraints to apply
- `varnames::Vector{Symbol}`: Variable names
- `n_lags::Int`: Number of lags

# Returns
- Modified `A` matrix with constraints applied
"""
function apply_constraints!(A::Matrix{T}, constraints::Vector{<:AbstractConstraint},
                            varnames::Vector{Symbol}, n_lags::Int) where T
    n_vars = length(varnames)

    for c in constraints
        if c isa ZeroConstraint
            apply_zero_constraint!(A, c, varnames, n_lags, n_vars)
        elseif c isa FixedConstraint
            apply_fixed_constraint!(A, c, varnames, n_vars)
        elseif c isa BlockExogeneity
            apply_block_exogeneity!(A, c, varnames, n_lags, n_vars)
        end
    end

    return A
end

function apply_zero_constraint!(A::Matrix, c::ZeroConstraint, varnames::Vector{Symbol},
                                n_lags::Int, n_vars::Int)
    # Find equation (row) index
    row_idx = findfirst(==(c.variable), varnames)
    row_idx === nothing && throw(ArgumentError("Variable $(c.variable) not found"))

    # Determine which lags to constrain
    lags_to_constrain = isempty(c.lags) ? (1:n_lags) : c.lags

    # Apply constraint for each regressor and lag
    for regressor in c.regressors
        reg_idx = findfirst(==(regressor), varnames)
        reg_idx === nothing && throw(ArgumentError("Variable $(regressor) not found"))

        for lag in lags_to_constrain
            (lag < 1 || lag > n_lags) && throw(ArgumentError("Invalid lag: $lag"))
            # Column index: 1 (intercept) + (lag-1)*n_vars + reg_idx
            col_idx = 1 + (lag - 1) * n_vars + reg_idx
            A[row_idx, col_idx] = 0
        end
    end
end

function apply_fixed_constraint!(A::Matrix, c::FixedConstraint,
                                 varnames::Vector{Symbol}, n_vars::Int)
    row_idx = findfirst(==(c.variable), varnames)
    row_idx === nothing && throw(ArgumentError("Variable $(c.variable) not found"))

    reg_idx = findfirst(==(c.regressor), varnames)
    reg_idx === nothing && throw(ArgumentError("Variable $(c.regressor) not found"))

    col_idx = 1 + (c.lag - 1) * n_vars + reg_idx
    A[row_idx, col_idx] = c.value
end

function apply_block_exogeneity!(A::Matrix, c::BlockExogeneity, varnames::Vector{Symbol},
                                  n_lags::Int, n_vars::Int)
    # Create equivalent zero constraints
    for to_var in c.to
        for lag in 1:n_lags
            apply_zero_constraint!(A, ZeroConstraint(to_var, c.from, [lag]),
                                  varnames, n_lags, n_vars)
        end
    end
end

"""
    build_selection_matrix(constraints, varnames, n_lags)

Build selection matrix `S` that selects free (unconstrained) parameters.

For restricted estimation: β̂ = S * θ where θ are the free parameters.

# Returns
- `S::Matrix`: Selection matrix mapping free parameters to full coefficient vector
- `n_free::Int`: Number of free parameters
"""
function build_selection_matrix(constraints::Vector{<:AbstractConstraint},
                                 varnames::Vector{Symbol}, n_lags::Int)
    n_vars = length(varnames)
    n_coef = n_vars * (1 + n_vars * n_lags)

    # Start with all parameters free
    is_free = trues(n_coef)

    # Mark constrained parameters as not free
    for c in constraints
        if c isa ZeroConstraint
            mark_zero_constraint!(is_free, c, varnames, n_lags, n_vars)
        elseif c isa BlockExogeneity
            for to_var in c.to
                for lag in 1:n_lags
                    mark_zero_constraint!(is_free, ZeroConstraint(to_var, c.from, [lag]),
                                        varnames, n_lags, n_vars)
                end
            end
        end
        # Note: FixedConstraint needs different handling (not just selection)
    end

    n_free = sum(is_free)
    free_indices = findall(is_free)

    # Build selection matrix
    S = zeros(n_coef, n_free)
    for (j, i) in enumerate(free_indices)
        S[i, j] = 1
    end

    return S, n_free
end

function mark_zero_constraint!(is_free::BitVector, c::ZeroConstraint,
                               varnames::Vector{Symbol}, n_lags::Int, n_vars::Int)
    row_idx = findfirst(==(c.variable), varnames)
    row_idx === nothing && return

    lags_to_constrain = isempty(c.lags) ? (1:n_lags) : c.lags

    for regressor in c.regressors
        reg_idx = findfirst(==(regressor), varnames)
        reg_idx === nothing && continue

        for lag in lags_to_constrain
            (lag < 1 || lag > n_lags) && continue
            # Linear index in vectorized form
            # vec(A) = [col1; col2; ...; coln]
            # A is n_vars × (1 + n_vars * n_lags)
            col_idx = 1 + (lag - 1) * n_vars + reg_idx
            linear_idx = (col_idx - 1) * n_vars + row_idx
            is_free[linear_idx] = false
        end
    end
end

"""
    check_constraints(constraints, varnames, n_lags)

Validate that constraints are well-specified.

# Throws
- `ArgumentError` if constraints reference non-existent variables or invalid lags
"""
function check_constraints(constraints::Vector{<:AbstractConstraint},
                          varnames::Vector{Symbol}, n_lags::Int)
    for c in constraints
        if c isa ZeroConstraint
            c.variable ∈ varnames || throw(ArgumentError("Unknown variable: $(c.variable)"))
            all(r -> r ∈ varnames, c.regressors) ||
                throw(ArgumentError("Unknown regressor in ZeroConstraint"))
            all(l -> 1 ≤ l ≤ n_lags, c.lags) ||
                throw(ArgumentError("Invalid lag in ZeroConstraint: $(c.lags)"))

        elseif c isa FixedConstraint
            c.variable ∈ varnames || throw(ArgumentError("Unknown variable: $(c.variable)"))
            c.regressor ∈ varnames || throw(ArgumentError("Unknown regressor: $(c.regressor)"))
            1 ≤ c.lag ≤ n_lags || throw(ArgumentError("Invalid lag: $(c.lag)"))

        elseif c isa BlockExogeneity
            all(v -> v ∈ varnames, c.from) ||
                throw(ArgumentError("Unknown variable in 'from' list"))
            all(v -> v ∈ varnames, c.to) ||
                throw(ArgumentError("Unknown variable in 'to' list"))
        end
    end
    return true
end
