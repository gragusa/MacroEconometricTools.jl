# ============================================================================
# Utility Functions
# ============================================================================

# ============================================================================
# Lag operations (replaces ShiftedArrays dependency)
# ============================================================================

"""
    lag(x::AbstractVector, n::Int; default=NaN)

Create lagged version of vector `x` by `n` periods.

# Arguments
- `x::AbstractVector`: Input vector
- `n::Int`: Number of lags (positive for lags, negative for leads)
- `default`: Value to fill for unavailable observations (default: NaN)

# Examples
```julia
x = [1, 2, 3, 4, 5]
lag(x, 1)  # [NaN, 1, 2, 3, 4]
lag(x, 2)  # [NaN, NaN, 1, 2, 3]
```
"""
function lag(x::AbstractVector{T}, n::Int; default=T(NaN)) where T<:AbstractFloat
    len = length(x)
    if n == 0
        return copy(x)
    elseif n > 0  # Positive lag
        result = Vector{T}(undef, len)
        result[1:n] .= default
        result[(n+1):end] .= @view x[1:(end-n)]
        return result
    else  # Negative lag (lead)
        n_abs = abs(n)
        result = Vector{T}(undef, len)
        result[(end-n_abs+1):end] .= default
        result[1:(end-n_abs)] .= @view x[(n_abs+1):end]
        return result
    end
end

function lag(X::AbstractMatrix{T}, n::Int; default=T(NaN)) where T<:AbstractFloat
    return hcat([lag(col, n; default=default) for col in eachcol(X)]...)
end

"""
    create_lags(X::AbstractMatrix{T}, p::Int) where T

Create matrix of lagged values for VAR estimation.

# Arguments
- `X::Matrix{T}`: Data matrix (T × n_vars)
- `p::Int`: Number of lags

# Returns
- Matrix of size (T × (1 + n_vars * p)) with intercept and lags
"""
function create_lags(X::AbstractMatrix{T}, p::Int) where T<:AbstractFloat
    n_obs, n_vars = size(X)
    n_cols = 1 + n_vars * p

    # Preallocate with concrete type
    lagged = Matrix{T}(undef, n_obs, n_cols)

    # Intercept
    lagged[:, 1] .= one(T)

    # Lags
    for lag_num in 1:p
        for var_idx in 1:n_vars
            col_idx = 1 + (lag_num - 1) * n_vars + var_idx
            lagged[:, col_idx] = lag(view(X, :, var_idx), lag_num; default=T(NaN))
        end
    end

    return lagged
end

"""
    create_lags!(dest::AbstractMatrix, X::AbstractMatrix, p::Int, include_intercept::Bool=true)

In-place creation of lagged matrix for VAR estimation.

# Arguments
- `dest::Matrix`: Pre-allocated destination matrix
- `X::Matrix`: Source data matrix
- `p::Int`: Number of lags
- `include_intercept::Bool`: Whether to include intercept column
"""
function create_lags!(dest::AbstractMatrix{T}, X::AbstractMatrix{T}, p::Int,
                     include_intercept::Bool=true) where T
    n_obs, n_vars = size(X)
    offset = include_intercept ? 1 : 0

    # Intercept
    if include_intercept
        fill!(view(dest, :, 1), one(T))
    end

    # Create lags
    for lag_num in 1:p
        for var_idx in 1:n_vars
            col_idx = offset + (lag_num - 1) * n_vars + var_idx
            for t in (lag_num + 1):n_obs
                dest[t, col_idx] = X[t - lag_num, var_idx]
            end
        end
    end

    # Set first p rows to NaN (unusable due to lags)
    dest[1:p, :] .= NaN

    return dest
end

# ============================================================================
# Matrix utilities
# ============================================================================

"""
    duplication_matrix(n::Int)

Magnus-Neudecker duplication matrix D_n that satisfies vec(A) = D_n * vech(A).
"""
function duplication_matrix(n::Int)
    n² = n * n
    n_vech = n * (n + 1) ÷ 2

    D = zeros(n², n_vech)

    vech_idx = 1
    for j in 1:n
        for i in j:n
            vec_idx_ij = (j - 1) * n + i
            vec_idx_ji = (i - 1) * n + j
            D[vec_idx_ij, vech_idx] = 1.0
            if i != j
                D[vec_idx_ji, vech_idx] = 1.0
            end
            vech_idx += 1
        end
    end

    return D
end

"""
    elimination_matrix(n::Int)

Magnus-Neudecker elimination matrix L_n that satisfies vech(A) = L_n * vec(A).
"""
function elimination_matrix(n::Int)
    n² = n * n
    n_vech = n * (n + 1) ÷ 2

    L = zeros(n_vech, n²)

    vech_idx = 1
    for j in 1:n
        for i in j:n
            vec_idx = (j - 1) * n + i
            L[vech_idx, vec_idx] = 1.0
            vech_idx += 1
        end
    end

    return L
end

"""
    commutation_matrix(m::Int, n::Int)

Commutation matrix K_{m,n} that satisfies vec(A') = K_{m,n} * vec(A) for A ∈ ℝ^{m×n}.
"""
function commutation_matrix(m::Int, n::Int)
    mn = m * n
    K = zeros(mn, mn)

    for i in 1:m
        for j in 1:n
            # Position in vec(A)
            vec_idx = (j - 1) * m + i
            # Position in vec(A')
            vec_t_idx = (i - 1) * n + j
            K[vec_t_idx, vec_idx] = 1.0
        end
    end

    return K
end

# ============================================================================
# Companion form utilities
# ============================================================================

"""
    companion_form(A::Array{T,3}) where T

Build companion form matrix from VAR lag coefficients.

# Arguments
- `A::Array{T,3}`: Lag coefficient array (n_vars, n_vars, n_lags)

# Returns
- `F::Matrix{T}`: Companion matrix (n_vars*n_lags × n_vars*n_lags)
"""
function companion_form(A::Array{T,3}) where T
    n_vars, _, n_lags = size(A)
    n = n_vars * n_lags

    F = zeros(T, n, n)

    # Top block: lag coefficients
    for lag in 1:n_lags
        row_range = 1:n_vars
        col_range = ((lag - 1) * n_vars + 1):(lag * n_vars)
        F[row_range, col_range] .= view(A, :, :, lag)
    end

    # Identity blocks below
    if n_lags > 1
        for i in 1:(n_lags - 1)
            row_start = n_vars * i + 1
            row_end = n_vars * (i + 1)
            col_start = n_vars * (i - 1) + 1
            col_end = n_vars * i
            F[row_start:row_end, col_start:col_end] .= I(n_vars)
        end
    end

    return F
end

# ============================================================================
# Information criteria
# ============================================================================

"""
    aic(model)

Akaike Information Criterion.
"""
function aic(model::VARModel{T,OLSVAR}) where T
    n, _, n_vars = size(model)
    n_lags = size(model.coefficients.lags, 3)
    k = n_vars^2 * n_lags + n_vars  # Total parameters
    ll = log_likelihood(model)
    return -2ll + 2k
end

"""
    bic(model)

Bayesian Information Criterion.
"""
function bic(model::VARModel{T,OLSVAR}) where T
    n, _, n_vars = size(model)
    n_lags = size(model.coefficients.lags, 3)
    k = n_vars^2 * n_lags + n_vars
    ll = log_likelihood(model)
    return -2ll + k * log(n)
end

"""
    hqic(model)

Hannan-Quinn Information Criterion.
"""
function hqic(model::VARModel{T,OLSVAR}) where T
    n, _, n_vars = size(model)
    n_lags = size(model.coefficients.lags, 3)
    k = n_vars^2 * n_lags + n_vars
    ll = log_likelihood(model)
    return -2ll + 2k * log(log(n))
end

# ============================================================================
# Accessor methods
# ============================================================================

"""
    n_vars(model)

Number of variables in the model.
"""
n_vars(model::VARModel) = size(model.Y, 2)

"""
    effective_obs(model)

Number of observations used in estimation (after accounting for lags).

This is a custom accessor specific to MacroEconometricTools.jl.
For StatsBase compatibility, use `nobs(model)` which returns the total observations.
"""
effective_obs(model::VARModel) = size(model.residuals, 1)

"""
    StatsBase.nobs(model::VARModel)

Total number of observations in the original data (before lag adjustment).

This follows the StatsBase.jl convention where `nobs` returns the total sample size.
For the effective observations used in estimation, use `effective_obs(model)`.
"""
StatsBase.nobs(model::VARModel) = size(model.Y, 1)

"""
    n_lags(model)

Number of lags in the model.
"""
n_lags(model::VARModel) = size(model.coefficients.lags, 3)

"""
    varnames(model)

Variable names in the model.
"""
varnames(model::VARModel) = model.names

"""
    intercept(model)

Intercept coefficients from the model.

Returns a vector of length `n_vars(model)`.
"""
intercept(model::VARModel) = model.coefficients.intercept

"""
    StatsBase.dof(model::VARModel)

Degrees of freedom in the model (number of estimated parameters).

For a VAR(p) model with n variables:
- Each equation has: 1 intercept + n × p lag coefficients
- Total: n × (1 + n × p) parameters
"""
StatsBase.dof(model::VARModel) = n_vars(model) * (1 + n_vars(model) * n_lags(model))

"""
    StatsBase.dof_residual(model::VARModel)

Residual degrees of freedom.

Calculated as: effective_obs - dof
"""
StatsBase.dof_residual(model::VARModel) = effective_obs(model) - dof(model)

"""
    StatsBase.modelmatrix(model::VARModel)

Design matrix used in VAR estimation.

Returns the matrix X (with intercept and lags) used in the regression Y = XB + ε.
Size: (T × (1 + n_vars × n_lags))
"""
StatsBase.modelmatrix(model::VARModel) = model.X

"""
    StatsBase.rss(model::VARModel)

Residual sum of squares.

Sum of squared residuals across all equations.
"""
StatsBase.rss(model::VARModel) = sum(abs2, model.residuals)

"""
    Base.size(model::VARModel)

Return (effective_obs, n_lags, n_vars).
"""
Base.size(model::VARModel) = (effective_obs(model), n_lags(model), n_vars(model))

# ============================================================================
# Pretty printing
# ============================================================================

function Base.show(io::IO, model::VARModel{T,S}) where {T,S}
    println(io, "VARModel{$T,$S}")
    println(io, "  Variables: ", join(model.names, ", "))
    println(io, "  Observations: ", effective_obs(model), " (", nobs(model), " total)")
    println(io, "  Lags: ", n_lags(model))
    if !isnothing(model.coefficients.constraints)
        println(io, "  Constraints: ", length(model.coefficients.constraints), " applied")
    end
end

function Base.show(io::IO, ::MIME"text/plain", model::VARModel{T,S}) where {T,S}
    show(io, model)
end
