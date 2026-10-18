# ============================================================================
# Constraint System for VAR Estimation
# ============================================================================
# Note: `abstract type AbstractConstraint end` is defined in types.jl
# to resolve load-order dependencies with VARCoefficients.

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
# Constraint Resolution
# ============================================================================

# Row of the coefficient matrix `B` ((1 + K p) × K, intercept first, then lag 1
# of every variable, lag 2, ...) that holds `regressor` at `lag`.
_coef_row(reg_idx::Int, lag::Int, n_vars::Int) = 1 + (lag - 1) * n_vars + reg_idx

function _constraint_var(name::Symbol, varnames::Vector{Symbol}, role::AbstractString)
    idx = findfirst(==(name), varnames)
    idx === nothing && throw(
        ArgumentError(
            "constraint $role $(repr(name)) is not a model variable; variables are $(varnames)"
        )
    )
    return idx
end

function _constraint_lag(lag::Integer, n_lags::Int)
    1 <= lag <= n_lags ||
        throw(ArgumentError("constraint lag $lag is outside 1:$n_lags"))
    return Int(lag)
end

# (equation, coefficient row, value) for every coefficient `c` restricts.
function _restricted_cells(c::ZeroConstraint, varnames::Vector{Symbol}, n_lags::Int)
    n_vars = length(varnames)
    eq = _constraint_var(c.variable, varnames, "variable")
    lags = isempty(c.lags) ? (1:n_lags) : c.lags
    return [
        (eq, _coef_row(_constraint_var(r, varnames, "regressor"), _constraint_lag(l, n_lags), n_vars), 0.0)
            for r in c.regressors for l in lags
    ]
end

function _restricted_cells(c::FixedConstraint, varnames::Vector{Symbol}, n_lags::Int)
    eq = _constraint_var(c.variable, varnames, "variable")
    reg = _constraint_var(c.regressor, varnames, "regressor")
    return [(eq, _coef_row(reg, _constraint_lag(c.lag, n_lags), length(varnames)), c.value)]
end

function _restricted_cells(c::BlockExogeneity, varnames::Vector{Symbol}, n_lags::Int)
    n_vars = length(varnames)
    from = [_constraint_var(v, varnames, "'from' variable") for v in c.from]
    to = [_constraint_var(v, varnames, "'to' variable") for v in c.to]
    return [(eq, _coef_row(reg, l, n_vars), 0.0) for eq in to for reg in from for l in 1:n_lags]
end

function _restricted_cells(c::AbstractConstraint, ::Vector{Symbol}, ::Int)
    throw(ArgumentError("unsupported constraint type $(typeof(c))"))
end

"""
    _resolve_constraints(constraints, varnames, n_lags) -> (restricted, values)

Resolve `constraints` against the model's variable names and lag order.
`restricted` is a `(1 + K p) × K` `BitMatrix` in the layout of the coefficient
matrix `B` (intercept first; column = equation) marking restricted coefficients,
and `values` holds their restricted values (zero for `ZeroConstraint` and
`BlockExogeneity`).

Throws `ArgumentError` for an unknown variable name, a lag outside `1:n_lags`,
or two constraints that restrict the same coefficient to different values.
"""
function _resolve_constraints(
        constraints::AbstractVector{<:AbstractConstraint},
        varnames::Vector{Symbol}, n_lags::Int
    )
    n_vars = length(varnames)
    n_coef = 1 + n_vars * n_lags
    restricted = falses(n_coef, n_vars)
    values = zeros(n_coef, n_vars)
    for c in constraints, (eq, row, v) in _restricted_cells(c, varnames, n_lags)
        if restricted[row, eq] && values[row, eq] != v
            throw(
                ArgumentError(
                    "conflicting constraints on the coefficient of $(repr(varnames[eq])) " *
                        "on row $row of B: $(values[row, eq]) and $v"
                )
            )
        end
        restricted[row, eq] = true
        values[row, eq] = v
    end
    return (; restricted, values)
end

"""
    check_constraints(constraints, varnames, n_lags)

Validate that constraints are well-specified.

# Throws
- `ArgumentError` if constraints reference non-existent variables or invalid
  lags, or restrict the same coefficient to different values
"""
function check_constraints(
        constraints::AbstractVector{<:AbstractConstraint},
        varnames::Vector{Symbol}, n_lags::Int
    )
    _resolve_constraints(constraints, varnames, n_lags)
    return true
end

# ============================================================================
# Constraint Application
# ============================================================================

"""
    apply_constraints!(A, constraints, varnames, n_lags)

Apply constraints to coefficient matrix `A` in-place.

# Arguments
- `A::AbstractMatrix`: Coefficient matrix (n_vars × (1 + n_vars * n_lags)), one
  row per equation
- `constraints::Vector{<:AbstractConstraint}`: Constraints to apply
- `varnames::Vector{Symbol}`: Variable names
- `n_lags::Int`: Number of lags

# Returns
- Modified `A` matrix with constraints applied
"""
function apply_constraints!(
        A::AbstractMatrix, constraints::AbstractVector{<:AbstractConstraint},
        varnames::Vector{Symbol}, n_lags::Int
    )
    n_vars = length(varnames)
    size(A) == (n_vars, 1 + n_vars * n_lags) || throw(
        DimensionMismatch(
            "A must be $(n_vars) × $(1 + n_vars * n_lags), got $(size(A))"
        )
    )
    Base.require_one_based_indexing(A)
    (; restricted, values) = _resolve_constraints(constraints, varnames, n_lags)
    for I in findall(restricted)
        row, eq = Tuple(I)
        A[eq, row] = values[row, eq]
    end
    return A
end

"""
    build_selection_matrix(constraints, varnames, n_lags)

Build selection matrix `S` that selects free (unconstrained) parameters.

For restricted estimation: `vec(A) = S * θ`, where `A` is the
`n_vars × (1 + n_vars * n_lags)` coefficient matrix (one row per equation) and `θ`
holds the free parameters. Every coefficient restricted by a constraint (zero or
fixed) is excluded.

# Returns
- `S::Matrix`: Selection matrix mapping free parameters to full coefficient vector
- `n_free::Int`: Number of free parameters
"""
function build_selection_matrix(
        constraints::AbstractVector{<:AbstractConstraint},
        varnames::Vector{Symbol}, n_lags::Int
    )
    (; restricted) = _resolve_constraints(constraints, varnames, n_lags)
    # `restricted` is (1 + K p) × K; `A` is its transpose.
    free_indices = findall(vec(permutedims(.!restricted)))
    n_free = length(free_indices)

    S = zeros(length(restricted), n_free)
    for (j, i) in enumerate(free_indices)
        S[i, j] = 1
    end

    return S, n_free
end

# ============================================================================
# Equality of specification types
# ============================================================================

# Field-wise `==`, `isequal`, and `hash` for specification types whose fields
# include arrays (the default `===` compares arrays by identity). Values that
# differ only in a type parameter (e.g. the date type `D` of
# `NarrativeRestriction{D}`) compare by their fields.
for S in (
        :ZeroConstraint, :FixedConstraint, :BlockExogeneity, :CholeskyID,
        :SignRestriction, :NarrativeShockRestriction, :NarrativeRestriction,
    )
    @eval begin
        function Base.:(==)(a::$S, b::$S)
            return all(f -> getfield(a, f) == getfield(b, f), fieldnames(typeof(a)))
        end
        function Base.isequal(a::$S, b::$S)
            return all(f -> isequal(getfield(a, f), getfield(b, f)), fieldnames(typeof(a)))
        end
        function Base.hash(x::$S, h::UInt)
            h = hash($(QuoteNode(S)), h)
            for f in fieldnames(typeof(x))
                h = hash(getfield(x, f), h)
            end
            return h
        end
    end
end
