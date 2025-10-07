# ============================================================================
# VAR Inference
# ============================================================================

"""
    StatsBase.stderror(model::VARModel{T,OLSVAR}) where T

Compute standard errors for VAR coefficients.

Returns standard errors in same shape as coefficients.
"""
function StatsBase.stderror(model::VARModel{T,OLSVAR}) where T
    n_obs_val = n_obs(model)
    n_vars_val = n_vars(model)
    n_lags_val = n_lags(model)

    # Get X matrix used in estimation
    n_lags_val_check = n_lags(model)
    valid_idx = (n_lags_val_check + 1):size(model.Y, 1)
    X = model.X[valid_idx, :]

    # Variance-covariance of coefficients: (Σ ⊗ (X'X)^{-1})
    Σ = vcov(model)
    XXinv = inv(X' * X)

    # Kronecker product for coefficient variance
    V_coef = kron(Σ, XXinv)

    # Extract standard errors (sqrt of diagonal)
    stderr_vec = sqrt.(diag(V_coef))

    # Reshape to coefficient matrix
    n_coef_per_eq = 1 + n_vars_val * n_lags_val
    stderr_matrix = reshape(stderr_vec, (n_coef_per_eq, n_vars_val))

    return stderr_matrix
end

"""
    confint(model::VARModel; level=0.95)

Compute confidence intervals for VAR coefficients.

# Returns
- NamedTuple with `lower` and `upper` matrices
"""
function StatsBase.confint(model::VARModel{T,OLSVAR}; level::Float64=0.95) where T
    coef_est = coef(model)
    stderr_est = stderror(model)

    α = 1 - level
    z = norminvcdf(1 - α / 2)

    # Reconstruct full coefficient matrix
    n_vars_val = n_vars(model)
    n_lags_val = n_lags(model)
    A = zeros(T, 1 + n_vars_val * n_lags_val, n_vars_val)
    A[1, :] = coef_est.intercept
    A[2:end, :] = reshape(coef_est.lags, (n_vars_val * n_lags_val, n_vars_val))

    lower = A .- z .* stderr_est
    upper = A .+ z .* stderr_est

    return (lower=lower, upper=upper)
end
