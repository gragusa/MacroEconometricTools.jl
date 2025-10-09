# ============================================================================
# VAR Estimation with Constraints
# ============================================================================

"""
    estimate(::Type{OLSVAR}, Y, n_lags; kwargs...)

Estimate a Vector Autoregression using Ordinary Least Squares.

# Arguments
- `::Type{OLSVAR}`: VAR specification type
- `Y::Matrix{T}`: Data matrix (T × n_vars) or named matrix with variable names
- `n_lags::Int`: Number of lags

# Keyword Arguments
- `constraints::Vector{<:AbstractConstraint}=AbstractConstraint[]`: Coefficient constraints
- `names::Vector{Symbol}=nothing`: Variable names (inferred if Y has names)
- `demean::Bool=false`: Whether to demean data before estimation

# Returns
- `VARModel{T,OLSVAR}`: Estimated VAR model

# Examples
```julia
# Basic estimation
var = estimate(OLSVAR, Y, 4)

# With constraints
constraints = [BlockExogeneity([:Foreign], [:Domestic])]
var = estimate(OLSVAR, Y, 4; constraints=constraints)
```
"""
function estimate(::Type{OLSVAR}, Y::AbstractMatrix{T}, n_lags::Int;
                  constraints::Vector{<:AbstractConstraint}=AbstractConstraint[],
                  names::Union{Nothing,Vector{Symbol}}=nothing,
                  demean::Bool=false) where T<:AbstractFloat

    n_obs_total, n_vars = size(Y)
    n_lags > 0 || throw(ArgumentError("n_lags must be positive"))
    n_obs_total > n_lags || throw(ArgumentError("Not enough observations for $n_lags lags"))

    # Handle variable names
    if names === nothing
        # Try to extract names from array-like object
        names = try_extract_names(Y, n_vars)
    end
    length(names) == n_vars || throw(ArgumentError("Length of names must match number of variables"))

    # Validate constraints
    if !isempty(constraints)
        check_constraints(constraints, names, n_lags)
    end

    # Demean if requested
    Y_work = demean ? Y .- mean(Y, dims=1) : copy(Y)

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
    coefs = VARCoefficients(intercept, lags_matrix, isempty(constraints) ? nothing : constraints)

    # Companion form
    F = companion_form(lags_matrix)

    # Metadata
    metadata = (
        n_obs_total = n_obs_total,
        n_obs_used = n_eff,
        demean = demean,
        timestamp = now()
    )

    return VARModel(OLSVAR(), Y_work, X, coefs, residuals, Σ, F, names, metadata)
end

"""
    VAR(Y, n_lags; kwargs...)

Convenience wrapper for `estimate(OLSVAR, Y, n_lags; kwargs...)` to preserve the
IRFs.jl API.
"""
function VAR(Y::AbstractMatrix{T}, n_lags::Int; kwargs...) where T<:AbstractFloat
    return estimate(OLSVAR, Y, n_lags; kwargs...)
end

"""
    constrained_ols(X, Y, constraints, names, n_lags, n_vars)

Estimate VAR coefficients under linear constraints.

Uses restricted least squares with selection matrix approach.
"""
function constrained_ols(X::AbstractMatrix{T}, Y::AbstractMatrix{T},
                         constraints::Vector{<:AbstractConstraint},
                         names::Vector{Symbol}, n_lags::Int, n_vars::Int) where T

    # Check if we only have zero/block constraints (easy case)
    only_zero = all(c -> c isa Union{ZeroConstraint,BlockExogeneity}, constraints)

    if only_zero
        # Use selection matrix approach
        return constrained_ols_selection(X, Y, constraints, names, n_lags, n_vars)
    else
        # General case with fixed constraints
        return constrained_ols_general(X, Y, constraints, names, n_lags, n_vars)
    end
end

"""
    constrained_ols_selection(X, Y, constraints, names, n_lags, n_vars)

Restricted OLS using selection matrix (for zero constraints only).

Estimates each equation separately, applying constraints.
"""
function constrained_ols_selection(X::AbstractMatrix{T}, Y::AbstractMatrix{T},
                                   constraints::Vector{<:AbstractConstraint},
                                   names::Vector{Symbol}, n_lags::Int, n_vars::Int) where T

    n_coef_per_eq = 1 + n_vars * n_lags
    A = zeros(T, n_coef_per_eq, n_vars)

    # Estimate each equation separately
    for eq_idx in 1:n_vars
        # Determine which coefficients are free in this equation
        is_free = trues(n_coef_per_eq)

        for c in constraints
            if c isa ZeroConstraint
                # Check if this constraint applies to current equation
                if names[eq_idx] == c.variable
                    lags_to_constrain = isempty(c.lags) ? (1:n_lags) : c.lags
                    for regressor in c.regressors
                        reg_idx = findfirst(==(regressor), names)
                        reg_idx === nothing && continue
                        for lag in lags_to_constrain
                            (lag < 1 || lag > n_lags) && continue
                            col_idx = 1 + (lag - 1) * n_vars + reg_idx
                            is_free[col_idx] = false
                        end
                    end
                end
            elseif c isa BlockExogeneity
                # Check if this equation is in the "to" list
                if names[eq_idx] in c.to
                    for from_var in c.from
                        reg_idx = findfirst(==(from_var), names)
                        reg_idx === nothing && continue
                        for lag in 1:n_lags
                            col_idx = 1 + (lag - 1) * n_vars + reg_idx
                            is_free[col_idx] = false
                        end
                    end
                end
            end
        end

        # Select free columns and estimate
        free_indices = findall(is_free)
        if !isempty(free_indices)
            X_free = X[:, free_indices]
            A[free_indices, eq_idx] = X_free \ Y[:, eq_idx]
        end
    end

    return A
end

"""
    constrained_ols_general(X, Y, constraints, names, n_lags, n_vars)

Restricted OLS with general linear equality constraints.

Handles both zero and fixed value constraints.
"""
function constrained_ols_general(X::AbstractMatrix{T}, Y::AbstractMatrix{T},
                                 constraints::Vector{<:AbstractConstraint},
                                names::Vector{Symbol}, n_lags::Int, n_vars::Int) where T

    # Start with unconstrained estimate
    A = X \ Y

    # Apply fixed constraints directly
    for c in constraints
        if c isa FixedConstraint
            row_idx = findfirst(==(c.variable), names)
            reg_idx = findfirst(==(c.regressor), names)
            col_idx = 1 + (c.lag - 1) * n_vars + reg_idx
            A[col_idx, row_idx] = c.value
        end
    end

    # Apply zero/block constraints
    zero_constraints = filter(c -> c isa Union{ZeroConstraint,BlockExogeneity}, constraints)
    if !isempty(zero_constraints)
        # Re-estimate with zero constraints holding fixed values constant
        A = reestimate_with_fixed(X, Y, A, constraints, names, n_lags, n_vars)
    end

    return A
end

"""
    reestimate_with_fixed(X, Y, A_init, constraints, names, n_lags, n_vars)

Re-estimate holding fixed constraints at their specified values.
"""
function reestimate_with_fixed(X::AbstractMatrix{T}, Y::AbstractMatrix{T},
                               A_init::AbstractMatrix{T},
                               constraints::Vector{<:AbstractConstraint},
                               names::Vector{Symbol}, n_lags::Int, n_vars::Int) where T

    # Build modified system: Y_adj = Y - X_fixed * A_fixed
    # Then estimate free parameters

    fixed_constraints = filter(c -> c isa FixedConstraint, constraints)
    if isempty(fixed_constraints)
        return constrained_ols_selection(X, Y, constraints, names, n_lags, n_vars)
    end

    # Adjust Y for fixed parameters
    Y_adj = copy(Y)
    for c in fixed_constraints
        row_idx = findfirst(==(c.variable), names)
        reg_idx = findfirst(==(c.regressor), names)
        col_idx = 1 + (c.lag - 1) * n_vars + reg_idx
        Y_adj[:, row_idx] .-= X[:, col_idx] .* c.value
    end

    # Now estimate on free parameters
    zero_constraints = filter(c -> c isa Union{ZeroConstraint,BlockExogeneity}, constraints)
    S, n_free = build_selection_matrix(zero_constraints, names, n_lags)

    # Solve for free parameters
    XS = X * S
    θ_free = XS \ Y_adj
    A = S * θ_free

    # Reshape and apply fixed constraints
    n_coef_per_eq = 1 + n_vars * n_lags
    A_full = reshape(A, (n_coef_per_eq, n_vars))
    apply_constraints!(A_full, fixed_constraints, names, n_lags)

    return A_full
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

Extract VAR coefficients.

# Returns
- `NamedTuple` with `intercept` and `lags` fields
"""
function StatsBase.coef(model::VARModel)
    return (intercept = model.coefficients.intercept,
            lags = model.coefficients.lags)
end

"""
    residuals(model::VARModel)

Extract model residuals.
"""
StatsBase.residuals(model::VARModel) = model.residuals

"""
    vcov(model::VARModel)

Extract residual covariance matrix.
"""
StatsBase.vcov(model::VARModel) = model.Σ

"""
    fitted(model::VARModel)

Compute fitted values.
"""
function StatsBase.fitted(model::VARModel)
    n_lags_val = n_lags(model)
    valid_idx = (n_lags_val + 1):size(model.Y, 1)
    X_est = model.X[valid_idx, :]

    # Reconstruct A matrix
    n_vars_val = n_vars(model)
    A = zeros(eltype(model.Y), 1 + n_vars_val * n_lags_val, n_vars_val)
    A[1, :] = model.coefficients.intercept
    A[2:end, :] = reshape(model.coefficients.lags, (n_vars_val * n_lags_val, n_vars_val))

    return X_est * A
end

"""
    log_likelihood(model::VARModel{T,OLSVAR})

Compute log-likelihood for OLS-VAR.
"""
function log_likelihood(model::VARModel{T,OLSVAR}) where T
    n_obs_val, n_vars_val = effective_obs(model), n_vars(model)
    Σ = vcov(model)

    # Log-likelihood: -0.5 * T * (n*log(2π) + log|Σ| + n)
    logdet_Σ = logdet(Σ)
    ll = -0.5 * n_obs_val * (n_vars_val * log(2π) + logdet_Σ + n_vars_val)

    return ll
end

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
function long_run_effect(model::VARModel{T}) where T
    A_sum = dropdims(sum(model.coefficients.lags, dims=3), dims=3)
    n = n_vars(model)
    return inv(Matrix{T}(I, n, n) - A_sum)
end

"""
    long_run_mean(model::VARModel{T}) where T

Compute long-run mean of the VAR process.

# Returns
- Vector of long-run means: (I - A₁ - ... - Aₚ)⁻¹ * c
"""
function long_run_mean(model::VARModel{T}) where T
    lr_effect = long_run_effect(model)
    return lr_effect * model.coefficients.intercept
end
