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
- `rng::AbstractRNG=Random.default_rng()`: Random number generator used when `inference == :bootstrap`

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
             normalization::AbstractNormalization=UnitStd(),
             rng::AbstractRNG=Random.default_rng()) where T

    horizon > 0 || throw(ArgumentError("horizon must be positive"))
    inference ∈ [:bootstrap, :delta, :none] ||
        throw(ArgumentError("inference must be :bootstrap, :delta, or :none"))
    all(0 .< coverage .< 1) || throw(ArgumentError("coverage levels must be in (0, 1)"))

    # Sort coverage levels
    coverage = sort(coverage)

    # Compute point estimate of IRF
    P = rotation_matrix(model, identification)
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
                                 normalization=normalization, rng=rng)
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
        names = model.names,
        timestamp = now()
    )

    return IRFResult(irf_point, stderr, lower, upper, coverage, identification,
                    inference, metadata)
end

# Convenience alias
impulse_response = irf

"""
    irf(model::VARModel, id::SignRestriction; kwargs...)

Compute impulse response functions for sign restriction identification.

For sign restrictions, this returns `SignRestrictedIRFResult` containing multiple
draws to represent set identification.

# Keyword Arguments
- `n_draws::Int=1000`: Number of valid rotation draws to compute
- `max_attempts::Int=10000`: Maximum attempts per draw
- `horizon::Int=24`: IRF horizon
- `coverage::Vector{Float64}=[0.68, 0.90, 0.95]`: Coverage levels for quantile bands
- `normalization::AbstractNormalization=UnitStd()`: Shock normalization
- `parallel::Symbol=:none`: Parallelization (`:none` or `:distributed`)
- `rng::AbstractRNG=Random.default_rng()`: Random number generator

# Returns
- `SignRestrictedIRFResult`: IRF result with multiple draws and quantile bands
"""
function irf(model::VARModel{T}, id::SignRestriction;
             n_draws::Int=1000,
             max_attempts::Int=10000,
             horizon::Int=24,
             coverage::Vector{Float64}=[0.68, 0.90, 0.95],
             normalization::AbstractNormalization=UnitStd(),
             parallel::Symbol=:none,
             rng::AbstractRNG=Random.default_rng()) where T

    # Compute multiple rotation matrices and IRFs
    rotation_matrices = Vector{Matrix{T}}(undef, n_draws)
    irf_draws = zeros(T, n_draws, horizon + 1, n_vars(model), n_vars(model))

    for i in 1:n_draws
        # Draw a rotation matrix satisfying restrictions
        P = rotation_matrix(model, id; max_draws=max_attempts, parallel=parallel,
                          verbose=false, rng=rng)
        P = normalize(P, normalization)
        rotation_matrices[i] = P

        # Compute IRF for this draw
        irf_draws[i, :, :, :] = compute_irf_point(model, P, horizon)
    end

    # Compute pointwise quantiles
    irf_median = dropdims(median(irf_draws; dims=1); dims=1)

    lower = Vector{Array{T,3}}(undef, length(coverage))
    upper = Vector{Array{T,3}}(undef, length(coverage))

    for (idx, cov) in enumerate(coverage)
        α = 1 - cov
        lower_q = α / 2
        upper_q = 1 - α / 2

        lower[idx] = dropdims(mapslices(x -> quantile(x, lower_q), irf_draws; dims=1); dims=1)
        upper[idx] = dropdims(mapslices(x -> quantile(x, upper_q), irf_draws; dims=1); dims=1)
    end

    metadata = (
        horizon = horizon,
        n_draws = n_draws,
        normalization = typeof(normalization),
        names = model.names,
        timestamp = now()
    )

    return SignRestrictedIRFResult(irf_median, irf_draws, lower, upper,
                                   coverage, rotation_matrices, id, metadata)
end

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

**Note**: Only valid for Cholesky (triangular) identification schemes.

Based on Lütkepohl (2005), Section 3.7.

# Arguments
- `model::VARModel`: Estimated VAR model
- `P::Matrix`: Identification matrix
- `irf::Array{T,3}`: Point estimate of IRF
- `identification::AbstractIdentification`: Identification scheme

# Returns
- Standard error array of size (horizon+1, n_vars, n_vars)
"""
function compute_irf_stderr_delta(model::VARModel{T}, P::Matrix{T},
                                 irf::Array{T,3}, identification::AbstractIdentification) where T
    # Delta method is only valid for Cholesky identification
    if !(identification isa CholeskyID)
        @warn "Delta method standard errors only implemented for Cholesky identification. " *
              "Falling back to bootstrap for $(typeof(identification))."
        horizon = size(irf, 1) - 1
        irf_boot = bootstrap_irf(model, identification, horizon, 500; method=:wild)
        return std(irf_boot; dims=1)[1, :, :, :]
    end

    # Use analytical delta method formulas
    return irf_asymptotic_stderror(model, P, irf)
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
    Base.size(irf::AbstractIRFResult)

Size of IRF array (horizon+1, n_vars, n_shocks).
"""
Base.size(irf::IRFResult) = size(irf.irf)
Base.size(irf::SignRestrictedIRFResult) = size(irf.irf_median)

"""
    horizon(irf::AbstractIRFResult)

IRF horizon.
"""
horizon(irf::IRFResult) = size(irf.irf, 1) - 1
horizon(irf::SignRestrictedIRFResult) = size(irf.irf_median, 1) - 1

"""
    n_vars(irf::AbstractIRFResult)

Number of variables.
"""
n_vars(irf::IRFResult) = size(irf.irf, 2)
n_vars(irf::SignRestrictedIRFResult) = size(irf.irf_median, 2)

"""
    n_shocks(irf::AbstractIRFResult)

Number of shocks.
"""
n_shocks(irf::IRFResult) = size(irf.irf, 3)
n_shocks(irf::SignRestrictedIRFResult) = size(irf.irf_median, 3)

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

function Base.show(io::IO, irf::SignRestrictedIRFResult{T}) where T
    n_v = n_vars(irf)
    n_s = n_shocks(irf)
    h = horizon(irf)
    n_draws = size(irf.irf_draws, 1)

    println(io, "SignRestrictedIRFResult{$T}")
    println(io, "  Identification: ", typeof(irf.identification))
    println(io, "  Horizon: ", h)
    println(io, "  Variables: ", n_v, " × Shocks: ", n_s)
    println(io, "  Draws: ", n_draws, " rotation matrices")

    if !isempty(irf.coverage)
        println(io, "  Coverage: ", join(irf.coverage .* 100, "%, "), "%")
    end
end

function Base.show(io::IO, ::MIME"text/plain", irf::SignRestrictedIRFResult)
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

"""
    cumulative_irf(irf::SignRestrictedIRFResult)

Compute cumulative impulse response functions for sign-restricted IRFs.

# Returns
- New `SignRestrictedIRFResult` with cumulative IRFs
"""
function cumulative_irf(irf::SignRestrictedIRFResult{T}) where T
    cum_median = cumsum(irf.irf_median; dims=1)
    cum_draws = cumsum(irf.irf_draws; dims=2)
    cum_lower = [cumsum(lb; dims=1) for lb in irf.lower]
    cum_upper = [cumsum(ub; dims=1) for ub in irf.upper]

    metadata = merge(irf.metadata, (cumulative = true,))

    return SignRestrictedIRFResult(cum_median, cum_draws, cum_lower, cum_upper,
                                   irf.coverage, irf.rotation_matrices,
                                   irf.identification, metadata)
end
