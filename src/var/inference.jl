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

# ============================================================================
# Asymptotic variance computations for delta method
# ============================================================================

"""
    coefficient_covariance(model::VARModel)

Compute asymptotic covariance matrix of VAR coefficients (excluding intercept).

For a VAR model, the covariance is:
    Var(vec(B)) = Σ_ε ⊗ (X'X)^{-1}

where B are the lag coefficients (excluding intercept) and Σ_ε is the residual covariance.

# Returns
- Covariance matrix of size (n_vars * n_lags * n_vars, n_vars * n_lags * n_vars)
"""
function coefficient_covariance(model::VARModel{T,OLSVAR}) where T
    n_vars_val = n_vars(model)
    n_lags_val = n_lags(model)
    Σ = vcov(model)

    # Get design matrix (without first n_lags rows)
    valid_idx = (n_lags_val + 1):size(model.Y, 1)
    X = model.X[valid_idx, :]

    # (X'X)^{-1}
    XXinv = inv(X' * X)

    # Remove intercept from covariance computation
    # We only need the lag coefficient covariance
    XXinv_no_intercept = XXinv[2:end, 2:end]

    # Kronecker product: Σ ⊗ (X'X)^{-1}
    return kron(Σ, XXinv_no_intercept)
end

"""
    sigma_covariance(model::VARModel)

Compute asymptotic covariance matrix of vech(Σ_ε).

Uses the Magnus-Neudecker formula:
    Var(vech(Σ)) = 2 * D⁺ * (Σ ⊗ Σ) * (D⁺)' / n

where D is the duplication matrix and D⁺ is its Moore-Penrose pseudoinverse.

# Returns
- Covariance matrix of vech(Σ) elements
"""
function sigma_covariance(model::VARModel{T,OLSVAR}) where T
    n_obs_val = n_obs(model)
    n_vars_val = n_vars(model)
    Σ = Matrix(vcov(model))

    # Duplication matrix and its pseudoinverse
    D = duplication_matrix(n_vars_val)
    D_plus = pinv(D)

    # Magnus-Neudecker formula: 2 * D⁺ * (Σ ⊗ Σ) * (D⁺)'
    Σ_kron = kron(Σ, Σ)
    return 2 * D_plus * Σ_kron * D_plus' / n_obs_val
end

# ============================================================================
# IRF jacobian matrices for delta method
# ============================================================================

"""
    irf_jacobian_matrices(model::VARModel, irf_point::Array{T,3}, horizon::Int) where T

Compute Jacobian matrices G_h for IRF variance computation.

For each horizon h, G_h is the derivative of vec(Θ_h) with respect to vec(A),
where Θ_h is the IRF at horizon h and A are the VAR lag coefficients.

This is used in the delta method for computing IRF standard errors.

# Arguments
- `model::VARModel`: Estimated VAR model
- `irf_point::Array{T,3}`: Point estimate of IRF (horizon+1, n_vars, n_vars)
- `horizon::Int`: Maximum IRF horizon

# Returns
- Vector of G matrices, one for each horizon 1:horizon
"""
function irf_jacobian_matrices(model::VARModel{T,OLSVAR}, irf_point::Array{T,3},
                              horizon::Int) where T
    n_vars_val = n_vars(model)
    n_lags_val = n_lags(model)
    F = model.companion

    # Memoize F^h computations
    F_powers = Dict{Int,Matrix{T}}()
    for h in 0:horizon
        F_powers[h] = (F')^h
    end

    # Selection matrix J: extracts first n_vars rows from companion form
    J = zeros(T, n_vars_val, n_lags_val * n_vars_val)
    J[1:n_vars_val, 1:n_vars_val] .= I(n_vars_val)

    # Compute G_h for each horizon
    function compute_G_h(h::Int)
        G = zeros(T, n_vars_val^2, n_lags_val * n_vars_val^2)

        for j in 0:(h - 1)
            # F^(h-1-j)
            A_power = F_powers[h - 1 - j]

            # Φ_j = F^j[1:n_vars, 1:n_vars]
            Φ_j = (F^j)[1:n_vars_val, 1:n_vars_val]

            # G_h += J * F^(h-1-j) ⊗ Φ_j
            G .+= kron(J * A_power, Φ_j)
        end

        return G
    end

    return [compute_G_h(h) for h in 1:horizon]
end

"""
    irf_effect_covariance(model::VARModel, P::Matrix{T}, irf_point::Array{T,3}) where T

Compute asymptotic covariance of IRF using delta method.

Valid only for triangular (Cholesky) identification schemes.

Based on Lütkepohl (2005), Section 3.7.

# Arguments
- `model::VARModel`: Estimated VAR model
- `P::Matrix{T}`: Identification matrix (Cholesky factor)
- `irf_point::Array{T,3}`: Point estimate of IRF

# Returns
- Covariance array of size (horizon+1, n_vars^2, n_vars^2)
"""
function irf_effect_covariance(model::VARModel{T,OLSVAR}, P::Matrix{T},
                               irf_point::Array{T,3}) where T
    horizon = size(irf_point, 1) - 1
    n_vars_val = n_vars(model)
    n_lags_val = n_lags(model)

    # Compute Jacobians
    G_matrices = irf_jacobian_matrices(model, irf_point, horizon)

    # Variance matrices
    Σ_α = coefficient_covariance(model)
    Σ_σ = sigma_covariance(model)

    # Matrix utilities
    L = elimination_matrix(n_vars_val)
    K = commutation_matrix(n_vars_val, n_vars_val)
    I_m = Matrix{T}(I, n_vars_val, n_vars_val)
    P_inv = inv(P)

    # H matrix for variance contribution from Σ
    H = L' * inv(L * (I(n_vars_val^2) + K) * kron(P, I_m) * L')

    # Preallocate covariance array
    V = zeros(T, horizon + 1, n_vars_val^2, n_vars_val^2)

    # Horizon 0: only variance from Σ
    V[1, :, :] .= zero(T)

    # Horizons h ≥ 1
    for h in 1:horizon
        # C_h = (P' ⊗ I_m) * G_{h-1}
        C_h = kron(P', I_m) * G_matrices[h]

        # A_h = C_h * Σ_α * C_h'
        A_h = C_h * Σ_α * C_h'

        # C̄_h = (I_m ⊗ Θ_h * P^{-1}) * H
        Θ_h_P_inv = irf_point[h + 1, :, :] * P_inv
        C_bar_h = kron(I_m, Θ_h_P_inv) * H

        # B_h = C̄_h * Σ_σ * C̄_h'
        B_h = C_bar_h * Σ_σ * C_bar_h'

        # Total variance
        V[h + 1, :, :] = A_h + B_h
    end

    return V
end

"""
    irf_asymptotic_stderror(model::VARModel, P::Matrix{T}, irf_point::Array{T,3}) where T

Compute asymptotic standard errors for IRF using delta method.

Extracts the square root of diagonal elements of the covariance matrix.

# Arguments
- `model::VARModel`: Estimated VAR model
- `P::Matrix{T}`: Identification matrix (Cholesky factor)
- `irf_point::Array{T,3}`: Point estimate of IRF

# Returns
- Standard error array of size (horizon+1, n_vars, n_vars)
"""
function irf_asymptotic_stderror(model::VARModel{T,OLSVAR}, P::Matrix{T},
                               irf_point::Array{T,3}) where T
    n_vars_val = n_vars(model)

    # Compute covariance
    V = irf_effect_covariance(model, P, irf_point)

    # Extract standard errors as sqrt of diagonal
    horizon = size(V, 1) - 1
    stderror = zeros(T, horizon + 1, n_vars_val, n_vars_val)

    for h in 0:horizon
        v_h = V[h + 1, :, :]
        stderror_vec = sqrt.(max.(diag(v_h), zero(T)))  # Ensure non-negative
        stderror[h + 1, :, :] = reshape(stderror_vec, (n_vars_val, n_vars_val))
    end

    return stderror
end
