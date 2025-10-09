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
- `inference::Union{Nothing, InferenceType}=nothing`: Inference method
  - `nothing`: No inference (point estimates only)
  - `Analytic()`: Asymptotic delta method
  - `WildBootstrap(reps, save_draws)`: Wild bootstrap
  - `Bootstrap(reps, save_draws)`: Standard i.i.d. bootstrap
  - `BlockBootstrap(reps, block_length, save_draws)`: Moving block bootstrap
- `coverage::Vector{Float64}=[0.68, 0.90, 0.95]`: Coverage levels for confidence bands
- `normalization::AbstractNormalization=UnitStd()`: Shock normalization
- `rng::AbstractRNG=Random.default_rng()`: Random number generator

# Returns
- `IRFResult`: Impulse response functions with confidence bands

# Examples
```julia
# Point estimates only (no inference)
irfs = irf(var_model, CholeskyID(); inference=nothing)

# Wild bootstrap with saved draws
irfs = irf(var_model, CholeskyID();
          inference=WildBootstrap(reps=1000, save_draws=true))

# Block bootstrap for persistent series
irfs = irf(var_model, CholeskyID();
          inference=BlockBootstrap(reps=1000, block_length=20))

# Delta method for fast asymptotic inference
irfs = irf(var_model, CholeskyID(); inference=Analytic())
```
"""
function irf(model::VARModel{T}, identification::AbstractIdentification;
             horizon::Int=24,
             inference::Union{Nothing, InferenceType}=nothing,
             coverage::Vector{Float64}=[0.68, 0.90, 0.95],
             normalization::AbstractNormalization=UnitStd(),
             rng::AbstractRNG=Random.default_rng()) where T

    horizon > 0 || throw(ArgumentError("horizon must be positive"))
    all(0 .< coverage .< 1) || throw(ArgumentError("coverage levels must be in (0, 1)"))

    # Sort coverage levels
    coverage = sort(coverage)

    # Compute point estimate of IRF
    P = rotation_matrix(model, identification)
    P = normalize(P, normalization)
    irf_point = compute_irf_point(model, P, horizon)

    # Dispatch on inference type - NO if-statements!
    draws, stderr, lower, upper = compute_inference_bands(
        model, identification, irf_point, inference, coverage, rng
    )

    # Conditionally save draws based on inference type
    bootstrap_draws = should_save_draws(inference, draws)

    # Build metadata
    metadata = (
        horizon = horizon,
        inference_type = typeof(inference),
        normalization = typeof(normalization),
        names = model.names,
        timestamp = now()
    )

    return IRFResult(irf_point, stderr, bootstrap_draws, lower, upper, coverage,
                    identification, inference, metadata)
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

# ============================================================================
# Inference Bands Computation - Multiple Dispatch on InferenceType
# ============================================================================

"""
    should_save_draws(inference, draws)

Determine whether to save bootstrap draws based on inference type settings.
"""
should_save_draws(::Nothing, ::Nothing) = nothing
should_save_draws(::Analytic, ::Nothing) = nothing
should_save_draws(inf::Union{WildBootstrap, Bootstrap, BlockBootstrap}, draws) =
    inf.save_draws ? draws : nothing

"""
    compute_inference_bands(model, identification, irf_point, inference_type, coverage, rng)

Compute confidence bands using the specified inference method.

This function dispatches on the type of `inference_type`:
- `WildBootstrap`: Wild bootstrap with Rademacher weights
- `Bootstrap`: Standard i.i.d. bootstrap
- `BlockBootstrap`: Moving block bootstrap for time series
- `Analytic`: Asymptotic delta method (Lütkepohl)
- `Nothing`: No inference (empty bands)

# Returns
- `(draws, stderr, lower, upper)` tuple where:
  - `draws`: Bootstrap IRF draws (if bootstrap) or `nothing`
  - `stderr`: Standard errors array
  - `lower`: Vector of lower confidence bands (one per coverage level)
  - `upper`: Vector of upper confidence bands (one per coverage level)
"""
function compute_inference_bands end

# Method 1: Wild Bootstrap
function compute_inference_bands(
    model::VARModel{T},
    identification::AbstractIdentification,
    irf_point::Array{T,3},
    inf::WildBootstrap,
    coverage::Vector{Float64},
    rng::AbstractRNG
) where T

    horizon = size(irf_point, 1) - 1

    # Run wild bootstrap
    draws = bootstrap_irf_wild(model, identification, horizon, inf.reps, rng)

    # Compute bands from draws
    lower, upper = compute_bands_from_draws(irf_point, draws, coverage)

    # Compute stderr from draws
    stderr = dropdims(std(draws; dims=1); dims=1)

    return draws, stderr, lower, upper
end

# Method 2: Standard Bootstrap
function compute_inference_bands(
    model::VARModel{T},
    identification::AbstractIdentification,
    irf_point::Array{T,3},
    inf::Bootstrap,
    coverage::Vector{Float64},
    rng::AbstractRNG
) where T

    horizon = size(irf_point, 1) - 1
    draws = bootstrap_irf_standard(model, identification, horizon, inf.reps, rng)
    lower, upper = compute_bands_from_draws(irf_point, draws, coverage)
    stderr = dropdims(std(draws; dims=1); dims=1)

    return draws, stderr, lower, upper
end

# Method 3: Block Bootstrap
function compute_inference_bands(
    model::VARModel{T},
    identification::AbstractIdentification,
    irf_point::Array{T,3},
    inf::BlockBootstrap,
    coverage::Vector{Float64},
    rng::AbstractRNG
) where T

    horizon = size(irf_point, 1) - 1
    draws = bootstrap_irf_block(model, identification, horizon, inf.reps,
                                inf.block_length, rng)
    lower, upper = compute_bands_from_draws(irf_point, draws, coverage)
    stderr = dropdims(std(draws; dims=1); dims=1)

    return draws, stderr, lower, upper
end

# Method 4: Analytic (Delta Method)
function compute_inference_bands(
    model::VARModel{T},
    identification::AbstractIdentification,
    irf_point::Array{T,3},
    inf::Analytic,
    coverage::Vector{Float64},
    rng::AbstractRNG  # Not used, but keep signature consistent
) where T

    # Delta method doesn't produce draws
    draws = nothing

    # Compute asymptotic standard errors
    P = rotation_matrix(model, identification)
    P = normalize(P, UnitStd())
    stderr = irf_asymptotic_stderror(model, P, irf_point)

    # Compute bands from normal approximation
    lower, upper = compute_bands_from_stderr(irf_point, stderr, coverage)

    return draws, stderr, lower, upper
end

# Method 5: No Inference
function compute_inference_bands(
    model::VARModel{T},
    identification::AbstractIdentification,
    irf_point::Array{T,3},
    ::Nothing,
    coverage::Vector{Float64},
    rng::AbstractRNG
) where T

    draws = nothing
    stderr = zeros(T, size(irf_point))
    lower = [zeros(T, size(irf_point)) for _ in coverage]
    upper = [zeros(T, size(irf_point)) for _ in coverage]

    return draws, stderr, lower, upper
end

"""
    compute_bands_from_draws(irf_point, draws, coverage)

Compute confidence bands from bootstrap IRF draws using percentile method.

# Arguments
- `irf_point::Array{T,3}`: Point estimate IRF (horizon+1, n_vars, n_shocks)
- `draws::Array{T,4}`: Bootstrap draws (reps, horizon+1, n_vars, n_shocks)
- `coverage::Vector{Float64}`: Coverage levels (e.g., [0.68, 0.90, 0.95])

# Returns
- `(lower, upper)` tuple of vectors, one entry per coverage level
"""
function compute_bands_from_draws(irf_point::Array{T,3}, draws::Array{T,4},
                                 coverage::Vector{Float64}) where T
    lower = Vector{Array{T,3}}(undef, length(coverage))
    upper = Vector{Array{T,3}}(undef, length(coverage))

    # Compute centered bootstrap draws: draws - mean(draws) + irf_point
    draws_mean = dropdims(mean(draws; dims=1); dims=1)
    draws_centered = draws .- reshape(draws_mean, (1, size(draws_mean)...)) .+
                     reshape(irf_point, (1, size(irf_point)...))

    for (i, α) in enumerate(coverage)
        # Percentile method
        α_lower = (1 - α) / 2
        α_upper = 1 - α_lower

        lower[i] = dropdims(mapslices(x -> quantile(x, α_lower), draws_centered; dims=1); dims=1)
        upper[i] = dropdims(mapslices(x -> quantile(x, α_upper), draws_centered; dims=1); dims=1)
    end

    return lower, upper
end

"""
    compute_bands_from_stderr(irf_point, stderr, coverage)

Compute confidence bands from standard errors using normal approximation.

# Arguments
- `irf_point::Array{T,3}`: Point estimate IRF
- `stderr::Array{T,3}`: Standard errors
- `coverage::Vector{Float64}`: Coverage levels

# Returns
- `(lower, upper)` tuple of vectors, one entry per coverage level
"""
function compute_bands_from_stderr(irf_point::Array{T,3}, stderr::Array{T,3},
                                  coverage::Vector{Float64}) where T
    lower = Vector{Array{T,3}}(undef, length(coverage))
    upper = Vector{Array{T,3}}(undef, length(coverage))

    for (i, α) in enumerate(coverage)
        z = norminvcdf(1 - (1 - α) / 2)
        lower[i] = irf_point .- z .* stderr
        upper[i] = irf_point .+ z .* stderr
    end

    return lower, upper
end

# ============================================================================
# Post-Hoc Confidence Bands Computation
# ============================================================================

"""
    confidence_bands(irf::IRFResult, args...; kwargs...)

Compute confidence bands for IRF results using a specified inference method.

This function allows you to compute confidence bands after computing the IRF,
either by reusing saved bootstrap draws (fast) or by running new inference (requires model).

# Methods

1. **Recompute from saved draws** (fast, no model needed):
   ```julia
   confidence_bands(irf, inference_type; coverage=[...])
   ```
   Only works if `irf.bootstrap_draws` is not `nothing`.

2. **Recompute with new inference** (requires model):
   ```julia
   confidence_bands(irf, model, identification, inference_type; coverage=[...])
   ```
   Can compute any inference type, always works.

# Arguments
- `irf::IRFResult`: Computed IRF result
- `model::VARModel`: (Method 2 only) VAR model
- `identification::AbstractIdentification`: (Method 2 only) Identification scheme
- `inference_type::InferenceType`: Inference method to use

# Keyword Arguments
- `coverage::Vector{Float64}=[0.68, 0.90, 0.95]`: Coverage levels
- `rng::AbstractRNG=Random.default_rng()`: Random number generator (Method 2 only)

# Returns
- New `IRFResult` with same point estimates but updated confidence bands

# Examples
```julia
# Compute IRF once with saved draws
irf1 = irf(model, id; inference=WildBootstrap(save_draws=true))

# Fast: compute different coverage from saved draws
irf2 = confidence_bands(irf1, WildBootstrap(); coverage=[0.75, 0.85, 0.95])

# Slow: compute different inference method (needs model)
irf3 = confidence_bands(irf1, model, id, BlockBootstrap(block_length=20))

# Delta method (doesn't need draws)
irf4 = confidence_bands(irf1, model, id, Analytic())
```
"""
function confidence_bands end

# Method 1: Reuse saved bootstrap draws (fast path)
function confidence_bands(
    irf::IRFResult{T},
    ::Type{<:Union{WildBootstrap, Bootstrap, BlockBootstrap}};
    coverage::Vector{Float64}=[0.68, 0.90, 0.95]
) where T

    # Requires saved draws
    isnothing(irf.bootstrap_draws) &&
        error("Bootstrap draws not saved in IRFResult. " *
              "Pass model and identification to recompute, or use irf() with save_draws=true.")

    # Sort coverage
    coverage = sort(coverage)

    # Recompute bands from saved draws
    lower, upper = compute_bands_from_draws(irf.irf, irf.bootstrap_draws, coverage)

    # Return new IRFResult with same draws but updated bands
    return IRFResult(irf.irf, irf.stderr, irf.bootstrap_draws, lower, upper,
                    coverage, irf.identification, irf.inference, irf.metadata)
end

# Method 2: Recompute with new inference method (requires model)
function confidence_bands(
    irf::IRFResult{T},
    model::VARModel{T},
    identification::AbstractIdentification,
    inf::InferenceType;
    coverage::Vector{Float64}=[0.68, 0.90, 0.95],
    rng::AbstractRNG=Random.default_rng()
) where T

    # Sort coverage
    coverage = sort(coverage)

    # Dispatch to compute_inference_bands (reuses point estimates from irf.irf)
    draws, stderr, lower, upper = compute_inference_bands(
        model, identification, irf.irf, inf, coverage, rng
    )

    # Determine whether to save draws
    bootstrap_draws = should_save_draws(inf, draws)

    # Update metadata
    metadata = merge(irf.metadata, (
        inference_type = typeof(inf),
        timestamp = now()
    ))

    # Return new IRFResult
    return IRFResult(irf.irf, stderr, bootstrap_draws, lower, upper,
                    coverage, identification, inf, metadata)
end
