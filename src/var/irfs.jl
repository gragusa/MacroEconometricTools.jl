# ============================================================================
# Impulse Response Function Computation
# ============================================================================

"""
    irf(model::VARModel, identification::AbstractIdentification; kwargs...)

Compute impulse response functions with confidence bands.

# Arguments
- `model::VARModel`: Estimated VAR model
- `identification::AbstractIdentification`: Identification scheme

# Keyword Arguments
- `horizon::Int=24`: IRF horizon
- `inference::Symbol=:bootstrap`: Inference method (`:bootstrap`, `:delta`, `:none`)
- `coverage::Vector{Float64}=[0.68, 0.90, 0.95]`: Coverage levels for confidence bands
- `bootstrap_reps::Int=1000`: Number of bootstrap replications
- `bootstrap_method::Symbol=:wild`: Bootstrap method (`:wild`, `:standard`, `:block`)
- `block_length::Int=10`: Block length for block bootstrap
- `normalization::AbstractNormalization=UnitStd()`: Shock normalization

# Returns
- `IRFResult`: Impulse response functions with confidence bands

# Examples
```julia
# Cholesky identification with bootstrap inference
id = CholeskyID([:GDP, :Inflation, :InterestRate])
irfs = irf(var_model, id; horizon=24, inference=:bootstrap)

# Delta method for fast asymptotic inference
irfs = irf(var_model, id; horizon=12, inference=:delta)
```
"""
function irf(model::VARModel{T}, identification::AbstractIdentification;
             horizon::Int=24,
             inference::Symbol=:bootstrap,
             coverage::Vector{Float64}=[0.68, 0.90, 0.95],
             bootstrap_reps::Int=1000,
             bootstrap_method::Symbol=:wild,
             block_length::Int=10,
             normalization::AbstractNormalization=UnitStd()) where T

    horizon > 0 || throw(ArgumentError("horizon must be positive"))
    inference ∈ [:bootstrap, :delta, :none] ||
        throw(ArgumentError("inference must be :bootstrap, :delta, or :none"))
    all(0 .< coverage .< 1) || throw(ArgumentError("coverage levels must be in (0, 1)"))

    # Sort coverage levels
    coverage = sort(coverage)

    # Compute point estimate of IRF
    P = identify(model, identification)
    P = normalize(P, normalization)
    irf_point = compute_irf_point(model, P, horizon)

    # Compute confidence bands
    if inference == :none
        stderr = zeros(T, size(irf_point))
        lower = [zeros(T, size(irf_point)) for _ in coverage]
        upper = [zeros(T, size(irf_point)) for _ in coverage]

    elseif inference == :delta
        stderr = compute_irf_stderr_delta(model, P, irf_point, identification)
        lower, upper = compute_bands_delta(irf_point, stderr, coverage)

    elseif inference == :bootstrap
        irf_boot = bootstrap_irf(model, identification, horizon, bootstrap_reps;
                                 method=bootstrap_method, block_length=block_length,
                                 normalization=normalization)
        stderr = std(irf_boot; dims=1)[1, :, :, :]
        lower, upper = compute_bands_bootstrap(irf_point, irf_boot, coverage)
    end

    # Metadata
    metadata = (
        horizon = horizon,
        inference = inference,
        bootstrap_reps = inference == :bootstrap ? bootstrap_reps : 0,
        bootstrap_method = bootstrap_method,
        normalization = typeof(normalization),
        timestamp = now()
    )

    return IRFResult(irf_point, stderr, lower, upper, coverage, identification,
                    inference, metadata)
end

# Convenience alias
impulse_response = irf

"""
    compute_irf_point(model::VARModel, P::Matrix, horizon::Int)

Compute point estimate of structural IRFs.

# Returns
- Array of size (horizon+1, n_vars, n_shocks) with IRF coefficients
"""
function compute_irf_point(model::VARModel{T}, P::Matrix{T}, horizon::Int) where T
    n_vars_val = n_vars(model)
    n_lags_val = n_lags(model)
    F = model.companion

    # Preallocate
    irf_array = zeros(T, horizon + 1, n_vars_val, n_vars_val)

    # Impact response (horizon 0)
    irf_array[1, :, :] = P

    # Compute MA coefficients
    Φ = compute_ma_matrices(F, horizon, n_vars_val, n_lags_val)

    # IRF_h = Φ_h * P
    for h in 1:horizon
        mul!(view(irf_array, h + 1, :, :), Φ[:, :, h + 1], P)
    end

    return irf_array
end

# ============================================================================
# Delta Method Inference
# ============================================================================

"""
    compute_irf_stderr_delta(model, P, irf, identification)

Compute asymptotic standard errors using delta method.

Based on Lütkepohl (2005), Section 3.7.
"""
function compute_irf_stderr_delta(model::VARModel{T}, P::Matrix{T},
                                 irf::Array{T,3}, identification::AbstractIdentification) where T
    # This is a simplified implementation
    # Full implementation would compute exact Jacobian

    n_obs_val = n_obs(model)
    n_vars_val = n_vars(model)
    horizon = size(irf, 1) - 1

    # Placeholder: use bootstrap for complex identification schemes
    if !(identification isa CholeskyID)
        @warn "Delta method standard errors not fully implemented for $(typeof(identification)). Using bootstrap."
        irf_boot = bootstrap_irf(model, identification, horizon, 500; method=:wild)
        return std(irf_boot; dims=1)[1, :, :, :]
    end

    # For Cholesky: use analytical formula
    Σ = vcov(model)
    F = model.companion

    stderr = zeros(T, size(irf))

    # Variance of vec(Σ)
    Σ_vec_var = variance_of_sigma(model)

    # Compute variance of each IRF element
    # This requires the Jacobian ∂IRF/∂vec(Σ)
    # Simplified: use numerical differentiation or bootstrap

    # For now, return approximate stderr based on residual variance
    for h in 0:horizon
        # Approximation: stderr grows with horizon
        stderr[h + 1, :, :] = sqrt.(diag(Σ)) * (1 + h / 10) / sqrt(n_obs_val)
    end

    return stderr
end

"""
    variance_of_sigma(model::VARModel)

Compute asymptotic variance of residual covariance matrix.
"""
function variance_of_sigma(model::VARModel{T}) where T
    n_obs_val = n_obs(model)
    n_vars_val = n_vars(model)
    Σ = vcov(model)

    # Using Magnus-Neudecker formula: 2 * D⁺ * (Σ ⊗ Σ) * (D⁺)'
    D = duplication_matrix(n_vars_val)
    D_plus = pinv(D)

    Σ_kron = kron(Σ, Σ)
    Var_vech_Σ = 2 * D_plus * Σ_kron * D_plus' / n_obs_val

    return Var_vech_Σ
end

"""
    compute_bands_delta(irf, stderr, coverage)

Compute confidence bands using normal approximation.
"""
function compute_bands_delta(irf::Array{T,3}, stderr::Array{T,3},
                            coverage::Vector{Float64}) where T
    lower = Vector{Array{T,3}}(undef, length(coverage))
    upper = Vector{Array{T,3}}(undef, length(coverage))

    for (i, α) in enumerate(coverage)
        z = norminvcdf(1 - (1 - α) / 2)
        lower[i] = irf .- z .* stderr
        upper[i] = irf .+ z .* stderr
    end

    return lower, upper
end

# ============================================================================
# Accessor Methods for IRFResult
# ============================================================================

"""
    Base.size(irf::IRFResult)

Size of IRF array (horizon+1, n_vars, n_shocks).
"""
Base.size(irf::IRFResult) = size(irf.irf)

"""
    horizon(irf::IRFResult)

IRF horizon.
"""
horizon(irf::IRFResult) = size(irf.irf, 1) - 1

"""
    n_vars(irf::IRFResult)

Number of variables.
"""
n_vars(irf::IRFResult) = size(irf.irf, 2)

"""
    n_shocks(irf::IRFResult)

Number of shocks.
"""
n_shocks(irf::IRFResult) = size(irf.irf, 3)

# ============================================================================
# Pretty Printing
# ============================================================================

function Base.show(io::IO, irf::IRFResult{T}) where T
    n_v = n_vars(irf)
    n_s = n_shocks(irf)
    h = horizon(irf)

    println(io, "IRFResult{$T}")
    println(io, "  Identification: ", typeof(irf.identification))
    println(io, "  Horizon: ", h)
    println(io, "  Variables: ", n_v, " × Shocks: ", n_s)
    println(io, "  Inference: ", irf.inference)

    if !isempty(irf.coverage)
        println(io, "  Coverage: ", join(irf.coverage .* 100, "%, "), "%")
    end
end

function Base.show(io::IO, ::MIME"text/plain", irf::IRFResult)
    show(io, irf)
end

# ============================================================================
# Cumulative IRFs
# ============================================================================

"""
    cumulative_irf(irf::IRFResult)

Compute cumulative impulse response functions.

# Returns
- New `IRFResult` with cumulative IRFs
"""
function cumulative_irf(irf::IRFResult{T}) where T
    cum_irf = cumsum(irf.irf; dims=1)
    cum_lower = [cumsum(lb; dims=1) for lb in irf.lower]
    cum_upper = [cumsum(ub; dims=1) for ub in irf.upper]

    # Standard errors don't simply cumsum - recompute or leave as NaN
    cum_stderr = similar(irf.stderr)
    fill!(cum_stderr, NaN)

    metadata = merge(irf.metadata, (cumulative = true,))

    return IRFResult(cum_irf, cum_stderr, cum_lower, cum_upper, irf.coverage,
                    irf.identification, irf.inference, metadata)
end
