# ============================================================================
# VAR Estimation with Constraints
# ============================================================================

"""
    StatsBase.fit(::Type{OLSVAR}, Y, n_lags; kwargs...)

Fit a Vector Autoregression using Ordinary Least Squares.

This follows the StatsBase.jl convention for model fitting.

# Arguments
- `::Type{OLSVAR}`: VAR specification type
- `Y::Matrix{T}`: Data matrix (T × n_vars) or named matrix with variable names
- `n_lags::Int`: Number of lags

# Keyword Arguments
- `constraints::Vector{<:AbstractConstraint}=AbstractConstraint[]`: Coefficient constraints
- `names::Vector{Symbol}=nothing`: Variable names (inferred if Y has names)
- `demean::Bool=false`: Whether to demean data before estimation
- `dates=nothing`: Observation dates, one per row of `Y` (length = number of
  observations). When supplied, the lag-trimmed dates are stored on the model and
  align with the residual rows, so date-based narrative restrictions resolve
  automatically. See [`dates`](@ref).

# Returns
- `VARModel{T,OLSVAR}`: Estimated VAR model

# Examples
```julia
# Basic estimation
var = fit(OLSVAR, Y, 4)

# With constraints
constraints = [BlockExogeneity([:Foreign], [:Domestic])]
var = fit(OLSVAR, Y, 4; constraints=constraints)
```
"""
function StatsBase.fit(
        ::Type{OLSVAR}, Y::AbstractMatrix{T}, n_lags::Int;
        constraints::Vector{<:AbstractConstraint} = AbstractConstraint[],
        names::Union{Nothing, Vector{Symbol}} = nothing,
        demean::Bool = false,
        dates = nothing
    ) where {T <: AbstractFloat}
    n_obs_total, n_vars = size(Y)
    n_lags > 0 || throw(ArgumentError("n_lags must be positive"))
    n_obs_total > n_lags || throw(ArgumentError("Not enough observations for $n_lags lags"))

    if dates !== nothing
        length(dates) == n_obs_total ||
            throw(ArgumentError("dates must have one entry per observation (got $(length(dates)) for $n_obs_total observations)"))
    end

    # Handle variable names
    if names === nothing
        # Try to extract names from array-like object
        names = try_extract_names(Y, n_vars)
    end
    length(names) == n_vars ||
        throw(ArgumentError("Length of names must match number of variables"))

    # Validate constraints
    if !isempty(constraints)
        check_constraints(constraints, names, n_lags)
    end

    # Demean if requested
    Y_work = demean ? Y .- mean(Y, dims = 1) : copy(Y)

    # Create lagged matrix
    X = create_lags(Y_work, n_lags)

    # Remove rows with missing values (first n_lags rows)
    valid_idx = (n_lags + 1):n_obs_total
    Y_est = Y_work[valid_idx, :]
    X_est = X[valid_idx, :]

    # Estimate coefficients
    if isempty(constraints)
        # Unconstrained OLS: A = (X'X)^{-1} X'Y
        A = X_est \ Y_est
    else
        # Constrained OLS
        A = constrained_ols(X_est, Y_est, constraints, names, n_lags, n_vars)
    end

    # Compute residuals
    residuals = Y_est - X_est * A

    # Residual covariance
    n_eff = size(residuals, 1)
    df = n_eff - n_vars * n_lags - 1  # Degrees of freedom
    Σ = Symmetric((residuals' * residuals) / df)

    # Build coefficient structure
    intercept = A[1, :]
    lags_matrix = Array(reshape(A[2:end, :]', (n_vars, n_vars, n_lags)))
    coefs = VARCoefficients(
        intercept, lags_matrix, isempty(constraints) ? nothing :
            constraints
    )

    # Companion form
    F = companion_form(lags_matrix)

    # Dates aligned with the residual rows (lag-trimmed, length n_eff)
    model_dates = dates === nothing ? nothing : dates[valid_idx]

    # Metadata
    metadata = (
        n_obs_total = n_obs_total,
        n_obs_used = n_eff,
        demean = demean,
        timestamp = now(),
    )

    return VARModel(
        OLSVAR(), Y_work, X, coefs, residuals, Σ, F, names, model_dates,
        metadata
    )
end

"""
    VAR(Y, n_lags; kwargs...)

Convenience wrapper for `fit(OLSVAR, Y, n_lags; kwargs...)` to preserve the
IRFs.jl API.
"""
function VAR(Y::AbstractMatrix{T}, n_lags::Int; kwargs...) where {T <: AbstractFloat}
    return fit(OLSVAR, Y, n_lags; kwargs...)
end

"""
    constrained_ols(X, Y, constraints, names, n_lags, n_vars)

Estimate VAR coefficients under zero and fixed-value constraints.

Restricted least squares equation by equation: in each equation the restricted
coefficients are set to their values, their contribution is subtracted from the
dependent variable, and the free coefficients are estimated by OLS on the
remaining regressors. Returns the `(1 + n_vars * n_lags) × n_vars` coefficient
matrix (intercept first; column = equation).
"""
function constrained_ols(
        X::AbstractMatrix{T}, Y::AbstractMatrix{T},
        constraints::AbstractVector{<:AbstractConstraint},
        names::Vector{Symbol}, n_lags::Int, n_vars::Int
    ) where {T}
    length(names) == n_vars ||
        throw(DimensionMismatch("expected $n_vars names, got $(length(names))"))
    n_coef = 1 + n_vars * n_lags
    size(X, 2) == n_coef && size(Y, 2) == n_vars && size(X, 1) == size(Y, 1) || throw(
        DimensionMismatch(
            "X must be T × $n_coef and Y T × $n_vars with the same T; got $(size(X)) and $(size(Y))"
        )
    )
    Base.require_one_based_indexing(X, Y)
    (; restricted, values) = _resolve_constraints(constraints, names, n_lags)

    A = zeros(T, n_coef, n_vars)
    for eq in 1:n_vars
        fixed = findall(restricted[:, eq])
        free = findall(.!restricted[:, eq])
        vals = T.(values[fixed, eq])
        y = Y[:, eq] - X[:, fixed] * vals
        A[fixed, eq] = vals
        if !isempty(free)
            A[free, eq] = X[:, free] \ y
        end
    end
    return A
end

"""
    try_extract_names(Y, n_vars)

Try to extract variable names from matrix-like object.
"""
function try_extract_names(Y, n_vars::Int)
    if hasproperty(Y, :colnames)
        return Symbol.(Y.colnames)
    elseif hasproperty(Y, :names)
        return Y.names
    else
        return [Symbol("Y_$i") for i in 1:n_vars]
    end
end

# ============================================================================
# Coefficient extraction
# ============================================================================

"""
    coef(model::VARModel)

Coefficient matrix `B` of the regression `Y = X B + U`, of size
`(1 + n_vars * n_lags) × n_vars`. Column `j` holds equation `j`; row 1 is the
intercept and rows `1 + (ℓ - 1) * n_vars .+ (1:n_vars)` are the coefficients on
lag `ℓ` of each variable, matching the columns of `modelmatrix(model)`.

Returns a new matrix. See [`intercept`](@ref) for the intercept alone and
`model.coefficients.lags` for the `(n_vars, n_vars, n_lags)` lag array.
"""
function StatsBase.coef(model::VARModel)
    (; intercept, lags) = model.coefficients
    n, _, p = size(lags)
    return vcat(transpose(intercept), transpose(reshape(lags, n, n * p)))
end

"""
    residuals(model::VARModel)

Extract model residuals.
"""
StatsBase.residuals(model::VARModel) = model.residuals

"""
    residual_covariance(model::VARModel)

Residual covariance matrix `Σ = U'U / dof_residual(model)` (`n_vars × n_vars`).
"""
residual_covariance(model::VARModel) = model.Σ

"""
    vcov(model::VARModel)

Covariance matrix of `vec(coef(model))`, `kron(Σ, inv(X'X))`, where `Σ` is
[`residual_covariance`](@ref) and `X` the design matrix of the estimation
sample. Size `n_vars * (1 + n_vars * n_lags)` square.

Coefficient constraints are not taken into account: the formula is that of
unrestricted OLS.
"""
function StatsBase.vcov(model::VARModel)
    X = _estimation_design(model)
    return kron(Matrix(residual_covariance(model)), inv(X' * X))
end

# Design matrix rows used in estimation (the first `n_lags` rows of `model.X`
# are presample).
_estimation_design(model::VARModel) = model.X[(n_lags(model) + 1):end, :]

"""
    dates(model::VARModel)

Return the observation dates aligned with the model's residual rows, or
`nothing` if no dates were supplied at fit time. The returned vector is
lag-trimmed: entry `i` corresponds to residual row `i`, which is what date-based
narrative restrictions resolve against.
"""
dates(model::VARModel) = model.dates

"""
    fitted(model::VARModel)

Fitted values `X B` on the estimation sample (`nobs(model) × n_vars`).
"""
StatsBase.fitted(model::VARModel) = _estimation_design(model) * coef(model)

"""
    loglikelihood(model::VARModel{T,OLSVAR})

Gaussian log-likelihood evaluated at the residual covariance `Σ`:
`-nobs/2 * (n_vars * log(2π) + log|Σ| + n_vars)`.
"""
function StatsBase.loglikelihood(model::VARModel{T, OLSVAR}) where {T}
    n_obs_val, n_vars_val = nobs(model), n_vars(model)
    logdet_Σ = logdet(residual_covariance(model))
    return -0.5 * n_obs_val * (n_vars_val * log(2π) + logdet_Σ + n_vars_val)
end

@deprecate log_likelihood(model::VARModel) loglikelihood(model) false

"""
    is_stable(model::VARModel)

Check if VAR model is stable (all eigenvalues of companion matrix inside unit circle).

# Returns
- `true` if all eigenvalues have modulus < 1, `false` otherwise
"""
function is_stable(model::VARModel)
    eigenvalues = eigvals(model.companion)
    return all(abs.(eigenvalues) .< 1.0)
end

"""
    long_run_effect(model::VARModel{T}) where T

Compute long-run multiplier matrix (I - A₁ - ... - Aₚ)⁻¹.

# Returns
- Matrix of long-run effects of shocks on variables
"""
function long_run_effect(model::VARModel{T}) where {T}
    A_sum = dropdims(sum(model.coefficients.lags, dims = 3), dims = 3)
    n = n_vars(model)
    return inv(Matrix{T}(I, n, n) - A_sum)
end

"""
    long_run_mean(model::VARModel{T}) where T

Compute long-run mean of the VAR process.

# Returns
- Vector of long-run means: (I - A₁ - ... - Aₚ)⁻¹ * c
"""
function long_run_mean(model::VARModel{T}) where {T}
    lr_effect = long_run_effect(model)
    return lr_effect * model.coefficients.intercept
end
