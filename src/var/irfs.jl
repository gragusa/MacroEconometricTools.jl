# ============================================================================
# Impulse Response Function Computation
# ============================================================================

using AxisArrays: AxisArrays, AxisArray, Axis

# ============================================================================
# AxisArray Wrapping / Unwrapping Helpers
# ============================================================================

"""
    _wrap_irf_3d(raw, names) -> AxisArray

Wrap a raw (horizon+1, n_vars, n_shocks) array into a 3D AxisArray
with axes (:variable, :shock, :horizon).
"""
function _wrap_irf_3d(raw::Array{T, 3}, names::Vector{Symbol}) where {T}
    H1 = size(raw, 1)
    data = permutedims(raw, (2, 3, 1))  # (variable, shock, horizon)
    return AxisArray(data,
        Axis{:variable}(names),
        Axis{:shock}(names),
        Axis{:horizon}(0:(H1 - 1)))
end

"""
    _wrap_irf_4d(raw, names) -> AxisArray

Wrap a raw (reps, horizon+1, n_vars, n_shocks) array into a 4D AxisArray
with axes (:draw, :variable, :shock, :horizon).
"""
function _wrap_irf_4d(raw::Array{T, 4}, names::Vector{Symbol}) where {T}
    R, H1 = size(raw, 1), size(raw, 2)
    data = permutedims(raw, (1, 3, 4, 2))  # (draw, variable, shock, horizon)
    return AxisArray(data,
        Axis{:draw}(1:R),
        Axis{:variable}(names),
        Axis{:shock}(names),
        Axis{:horizon}(0:(H1 - 1)))
end

"""
    _unwrap_irf_3d(ax) -> Array{T,3}

Unwrap a 3D AxisArray (variable, shock, horizon) back to raw (horizon+1, n_vars, n_shocks).
"""
function _unwrap_irf_3d(ax::AxisArray)
    permutedims(Array(ax), (3, 1, 2))
end

"""
    _unwrap_irf_4d(ax) -> Array{T,4}

Unwrap a 4D AxisArray (draw, variable, shock, horizon) back to raw (draw, horizon+1, n_vars, n_shocks).
"""
function _unwrap_irf_4d(ax::AxisArray)
    permutedims(Array(ax), (1, 4, 2, 3))
end

# ============================================================================
# Fast Quantile Helpers (avoids mapslices overhead)
# ============================================================================

"""
    _quantile_along_dim1!(lo::Array{T,3}, hi::Array{T,3},
                          draws::Array{T,4}, α_lower, α_upper) where T

Compute elementwise quantiles of `draws` along dimension 1 (the bootstrap
draw dimension) and store results in pre-allocated `lo` and `hi`.

Uses partial sorting via `partialsort!` for efficiency — O(n) per element
instead of O(n log n) for a full sort.
"""
function _quantile_along_dim1!(lo::Array{T, 3}, hi::Array{T, 3},
        draws::Array{T, 4}, α_lower::Float64, α_upper::Float64) where {T}
    reps = size(draws, 1)
    buf = Vector{T}(undef, reps)
    idx_lo = clamp(round(Int, α_lower * reps), 1, reps)
    idx_hi = clamp(round(Int, α_upper * reps), 1, reps)

    d2, d3, d4 = size(draws, 2), size(draws, 3), size(draws, 4)
    @inbounds for k in 1:d4, j in 1:d3, h in 1:d2
        for r in 1:reps
            buf[r] = draws[r, h, j, k]
        end
        sort!(buf)
        lo[h, j, k] = buf[idx_lo]
        hi[h, j, k] = buf[idx_hi]
    end
end

# ============================================================================
# Cumulation Helpers
# ============================================================================

"""
    _resolve_cumulate(cumulate, names) -> Union{Nothing, Vector{Int}}

Resolve cumulation specification to variable indices.
Accepts `nothing`, `Vector{Symbol}`, or `Vector{Int}`.
"""
function _resolve_cumulate(cumulate, names::Vector{Symbol})
    cumulate === nothing && return nothing
    isempty(cumulate) && return nothing
    if cumulate isa Vector{Symbol}
        idx = Int[]
        for s in cumulate
            i = findfirst(==(s), names)
            i === nothing &&
                throw(ArgumentError("Variable :$s not found in model. Available: $names"))
            push!(idx, i)
        end
        return idx
    elseif cumulate isa Vector{Int}
        all(1 .<= cumulate .<= length(names)) ||
            throw(ArgumentError("Cumulate indices out of range [1, $(length(names))]"))
        return collect(cumulate)
    else
        throw(ArgumentError("cumulate must be nothing, Vector{Symbol}, or Vector{Int}"))
    end
end

"""
    _cumulate_point!(irf, idx)

In-place cumulative sum along the horizon dimension (dim 1) for selected variables.
Operates on raw arrays with shape (horizon+1, n_vars, n_shocks).
"""
function _cumulate_point!(irf::AbstractArray{T, 3}, idx::Vector{Int}) where {T}
    for v in idx
        for s in axes(irf, 3)
            for h in 2:size(irf, 1)
                irf[h, v, s] += irf[h - 1, v, s]
            end
        end
    end
end

"""
    _apply_cumulation(irf_point, draws, coverage, idx)

Apply per-variable cumulation to point estimate and bootstrap draws,
then recompute confidence bands from the cumulated draws.

Returns `(cum_point, cum_draws, stderr, lower, upper)`.
"""
function _apply_cumulation(
        irf_point::Array{T, 3},
        draws::Union{Nothing, Array{T, 4}},
        coverage::Vector{Float64},
        idx::Vector{Int}
) where {T}
    draws === nothing &&
        error("Cumulation requires bootstrap draws. " *
              "Use a bootstrap inference method (e.g., WildBootstrap(reps=1000)).")

    # Cumulate point estimate
    cum_point = copy(irf_point)
    _cumulate_point!(cum_point, idx)

    # Cumulate each bootstrap draw
    cum_draws = copy(draws)
    for r in axes(cum_draws, 1)
        _cumulate_point!(view(cum_draws,r,:,:,:), idx)
    end

    # Recompute bands from cumulated draws
    lower, upper = compute_bands_from_draws(cum_point, cum_draws, coverage)

    # Stderr from cumulated draws
    stderr = dropdims(std(cum_draws; dims = 1); dims = 1)

    return cum_point, cum_draws, stderr, lower, upper
end

# ============================================================================

"""
    irf(model::VARModel, identification::AbstractIdentification; kwargs...)

Compute impulse response functions with confidence bands.

Returns an `IRFResult` with AxisArray fields indexed by variable name, shock name,
and horizon (e.g., `result.irf[:GDP, :MonetaryShock, 0:12]`).

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
- `cumulate::Union{Nothing, Vector{Symbol}, Vector{Int}}=nothing`: Variables to cumulate.
  When specified, the selected variables are cumulated (cumulative sum along horizon).
  Requires a bootstrap inference method — incompatible with `Analytic()`.
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

# Cumulate GDP_growth to get level response
irfs = irf(var_model, CholeskyID();
          inference=WildBootstrap(reps=1000),
          cumulate=[:GDP_growth])

# Delta method for fast asymptotic inference
irfs = irf(var_model, CholeskyID(); inference=Analytic())
```
"""
function irf(model::VARModel{T}, identification::AbstractIdentification;
        horizon::Int = 24,
        inference::Union{Nothing, InferenceType} = nothing,
        coverage::Vector{Float64} = [0.68, 0.90, 0.95],
        normalization::AbstractNormalization = UnitStd(),
        cumulate::Union{Nothing, Vector{Symbol}, Vector{Int}} = nothing,
        scale::Real = 1,
        rng::AbstractRNG = Random.default_rng()) where {T}
    horizon > 0 || throw(ArgumentError("horizon must be positive"))
    all(0 .< coverage .< 1) || throw(ArgumentError("coverage levels must be in (0, 1)"))

    # Resolve cumulation indices
    cumulate_idx = _resolve_cumulate(cumulate, model.names)

    # Validate: analytic inference is incompatible with cumulation
    if cumulate_idx !== nothing && inference isa Analytic
        throw(ArgumentError(
            "Cumulation of IRFs is incompatible with Analytic() inference. " *
            "The delta method does not apply to cumulated IRFs. " *
            "Use a bootstrap method (e.g., WildBootstrap(reps=1000)) instead."))
    end

    # Sort coverage levels
    coverage = sort(coverage)

    # Compute point estimate of IRF
    P = rotation_matrix(model, identification)
    P = normalize(P, normalization)
    irf_point = compute_irf_point(model, P, horizon)

    # Dispatch on inference type - NO if-statements!
    draws, stderr,
    lower,
    upper = compute_inference_bands(
        model, identification, irf_point, inference, coverage, normalization, rng
    )

    # Apply cumulation if requested (BEFORE scaling and wrapping)
    if cumulate_idx !== nothing
        irf_point, draws, stderr,
        lower, upper = _apply_cumulation(
            irf_point, draws, coverage, cumulate_idx)
    end

    # Apply scaling (e.g., scale=0.25 for a 25bp shock under unit-effect normalization)
    if scale != 1
        s = T(scale)
        irf_point .*= s
        stderr .*= abs(s)
        for i in eachindex(lower)
            lower[i] .*= s
            upper[i] .*= s
        end
        if draws !== nothing
            draws .*= s
        end
    end

    # Conditionally save draws based on inference type
    bootstrap_draws = should_save_draws(inference, draws)

    # Wrap raw arrays into AxisArrays
    names = Symbol.(model.names)
    irf_ax = _wrap_irf_3d(irf_point, names)
    stderr_ax = _wrap_irf_3d(stderr, names)
    lower_ax = [_wrap_irf_3d(lb, names) for lb in lower]
    upper_ax = [_wrap_irf_3d(ub, names) for ub in upper]
    draws_ax = bootstrap_draws === nothing ? nothing : _wrap_irf_4d(bootstrap_draws, names)

    # Build metadata
    cumulate_syms = cumulate_idx === nothing ? nothing : Symbol.(model.names[cumulate_idx])
    metadata = (
        horizon = horizon,
        inference_type = typeof(inference),
        normalization = typeof(normalization),
        names = model.names,
        cumulated_vars = cumulate_syms,
        scale = scale,
        timestamp = now()
    )

    return IRFResult(irf_ax, stderr_ax, draws_ax, lower_ax, upper_ax, coverage,
        identification, inference, metadata)
end

# Convenience alias
impulse_response = irf

"""
    irf(model::VARModel{T, <:IVSVAR}; kwargs...)

Convenience method for IVSVAR models — defaults to `IVIdentification()` using
the instrument stored in the model. Equivalent to `irf(model, IVIdentification(); kwargs...)`.
"""
function irf(model::VARModel{T, <:IVSVAR}; kwargs...) where {T}
    irf(model, IVIdentification(); kwargs...)
end

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
- `cumulate::Union{Nothing, Vector{Symbol}, Vector{Int}}=nothing`: Variables to cumulate
- `parallel::Symbol=:none`: Parallelization (`:none` or `:distributed`)
- `rng::AbstractRNG=Random.default_rng()`: Random number generator

# Returns
- `SignRestrictedIRFResult`: IRF result with multiple draws and quantile bands
"""
function irf(model::VARModel{T}, id::SignRestriction;
        n_draws::Int = 1000,
        max_attempts::Int = 10000,
        horizon::Int = 24,
        coverage::Vector{Float64} = [0.68, 0.90, 0.95],
        normalization::AbstractNormalization = UnitStd(),
        cumulate::Union{Nothing, Vector{Symbol}, Vector{Int}} = nothing,
        scale::Real = 1,
        parallel::Symbol = :none,
        rng::AbstractRNG = Random.default_rng()) where {T}

    # Resolve cumulation indices
    cumulate_idx = _resolve_cumulate(cumulate, model.names)

    # Compute multiple rotation matrices and IRFs
    rotation_matrices = Vector{Matrix{T}}(undef, n_draws)
    irf_draws_raw = zeros(T, n_draws, horizon + 1, n_vars(model), n_vars(model))

    for i in 1:n_draws
        # Draw a rotation matrix satisfying restrictions
        P = rotation_matrix(model, id; max_draws = max_attempts, parallel = parallel,
            verbose = false, rng = rng)
        P = normalize(P, normalization)
        rotation_matrices[i] = P

        # Compute IRF for this draw
        irf_draws_raw[i, :, :, :] = compute_irf_point(model, P, horizon)
    end

    # Apply cumulation to each draw if requested
    if cumulate_idx !== nothing
        for i in 1:n_draws
            _cumulate_point!(view(irf_draws_raw,i,:,:,:), cumulate_idx)
        end
    end

    # Apply scaling
    if scale != 1
        irf_draws_raw .*= T(scale)
    end

    # Compute pointwise quantiles (from possibly cumulated/scaled draws)
    irf_median_raw = dropdims(median(irf_draws_raw; dims = 1); dims = 1)

    lower_raw = Vector{Array{T, 3}}(undef, length(coverage))
    upper_raw = Vector{Array{T, 3}}(undef, length(coverage))

    sz3 = size(irf_draws_raw)[2:4]
    for (idx, cov) in enumerate(coverage)
        α = 1 - cov
        lo = zeros(T, sz3...)
        hi = zeros(T, sz3...)
        _quantile_along_dim1!(lo, hi, irf_draws_raw, α / 2, 1 - α / 2)
        lower_raw[idx] = lo
        upper_raw[idx] = hi
    end

    # Wrap in AxisArrays
    names = Symbol.(model.names)
    irf_median_ax = _wrap_irf_3d(irf_median_raw, names)
    irf_draws_ax = _wrap_irf_4d(irf_draws_raw, names)
    lower_ax = [_wrap_irf_3d(lb, names) for lb in lower_raw]
    upper_ax = [_wrap_irf_3d(ub, names) for ub in upper_raw]

    cumulate_syms = cumulate_idx === nothing ? nothing : Symbol.(model.names[cumulate_idx])
    metadata = (
        horizon = horizon,
        n_draws = n_draws,
        normalization = typeof(normalization),
        names = model.names,
        cumulated_vars = cumulate_syms,
        scale = scale,
        timestamp = now()
    )

    return SignRestrictedIRFResult(irf_median_ax, irf_draws_ax, lower_ax, upper_ax,
        coverage, rotation_matrices, id, metadata)
end

"""
    compute_irf_point(model::VARModel, P::Matrix, horizon::Int)

Compute point estimate of structural IRFs.

# Returns
- Array of size (horizon+1, n_vars, n_shocks) with IRF coefficients
"""
function compute_irf_point(model::VARModel{T}, P::Matrix{T}, horizon::Int) where {T}
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
        mul!(view(irf_array,(h + 1),:,:), Φ[:, :, h + 1], P)
    end

    return irf_array
end

# ============================================================================
# Accessor Methods for IRFResult
# ============================================================================

"""
    Base.size(irf::AbstractIRFResult)

Size of IRF AxisArray (n_vars, n_shocks, horizon+1).
"""
Base.size(irf::IRFResult) = size(irf.irf)
Base.size(irf::SignRestrictedIRFResult) = size(irf.irf_median)

# Note: horizon() and n_vars() for IRFResult and SignRestrictedIRFResult are defined in types.jl

"""
    n_shocks(irf::AbstractIRFResult)

Number of shocks.
"""
function n_shocks(irf::IRFResult)
    ax = AxisArrays.axes(irf.irf, Axis{:shock})
    return length(AxisArrays.axisvalues(ax)[1])
end

function n_shocks(irf::SignRestrictedIRFResult)
    ax = AxisArrays.axes(irf.irf_median, Axis{:shock})
    return length(AxisArrays.axisvalues(ax)[1])
end

# ============================================================================
# Pretty Printing
# ============================================================================

function Base.show(io::IO, irf::IRFResult{T}) where {T}
    n_v = n_vars(irf)
    n_s = n_shocks(irf)
    h = horizon(irf)

    println(io, "IRFResult{$T}")
    println(io, "  Identification: ", typeof(irf.identification))
    println(io, "  Horizon: ", h)
    println(
        io, "  Variables: ", join(varnames(irf), ", "), " (", n_v, " × Shocks: ", n_s, ")")
    println(io, "  Inference: ", irf.inference)

    if !isempty(irf.coverage)
        println(io, "  Coverage: ", join(irf.coverage .* 100, "%, "), "%")
    end
    if haskey(irf.metadata, :cumulated_vars) && irf.metadata.cumulated_vars !== nothing
        println(io, "  Cumulated: ", join(irf.metadata.cumulated_vars, ", "))
    end
end

function Base.show(io::IO, ::MIME"text/plain", irf::IRFResult)
    show(io, irf)
end

function Base.show(io::IO, irf::SignRestrictedIRFResult{T}) where {T}
    n_v = n_vars(irf)
    n_s = n_shocks(irf)
    h = horizon(irf)
    nd = MacroEconometricTools.n_draws(irf)

    println(io, "SignRestrictedIRFResult{$T}")
    println(io, "  Identification: ", typeof(irf.identification))
    println(io, "  Horizon: ", h)
    println(
        io, "  Variables: ", join(varnames(irf), ", "), " (", n_v, " × Shocks: ", n_s, ")")
    println(io, "  Draws: ", nd, " rotation matrices")

    if !isempty(irf.coverage)
        println(io, "  Coverage: ", join(irf.coverage .* 100, "%, "), "%")
    end
    if haskey(irf.metadata, :cumulated_vars) && irf.metadata.cumulated_vars !== nothing
        println(io, "  Cumulated: ", join(irf.metadata.cumulated_vars, ", "))
    end
end

function Base.show(io::IO, ::MIME"text/plain", irf::SignRestrictedIRFResult)
    show(io, irf)
end

# ============================================================================
# Cumulative IRFs
# ============================================================================

"""
    cumulative_irf(irf::IRFResult; vars=nothing)

Compute cumulative impulse response functions.

# Keyword Arguments
- `vars::Union{Nothing, Vector{Symbol}, Vector{Int}}=nothing`: Variables to cumulate.
  When `nothing`, all variables are cumulated (backward compatible).

When bootstrap draws are available, confidence bands are correctly recomputed
from cumulated draws. Without draws, bands are approximately cumulated with a warning.

# Returns
- New `IRFResult` with cumulative IRFs
"""
function cumulative_irf(irf::IRFResult; vars = nothing)
    names = varnames(irf)
    all_idx = collect(1:length(names))

    # Resolve which variables to cumulate
    if vars === nothing
        cumulate_idx = all_idx
    else
        cumulate_idx = _resolve_cumulate(vars, names)
        cumulate_idx === nothing && return irf  # empty cumulate
    end

    # Unwrap AxisArrays to raw arrays
    raw_point = _unwrap_irf_3d(irf.irf)

    # Cumulate point estimate
    cum_point = copy(raw_point)
    _cumulate_point!(cum_point, cumulate_idx)

    if irf.bootstrap_draws !== nothing
        # Correct path: cumulate draws, recompute bands
        raw_draws = _unwrap_irf_4d(irf.bootstrap_draws)
        cum_draws = copy(raw_draws)
        for r in axes(cum_draws, 1)
            _cumulate_point!(view(cum_draws,r,:,:,:), cumulate_idx)
        end

        lower_raw, upper_raw = compute_bands_from_draws(cum_point, cum_draws, irf.coverage)
        cum_stderr = dropdims(std(cum_draws; dims = 1); dims = 1)
    else
        @warn "No bootstrap draws available. Cumulated confidence bands are approximate. " *
              "For correct bands, use irf() with a bootstrap method and save_draws=true, " *
              "or use the cumulate keyword in irf() directly."

        # Approximate: cumsum the bands directly (known to be incorrect but best available)
        cum_draws = nothing
        raw_lower = [_unwrap_irf_3d(lb) for lb in irf.lower]
        raw_upper = [_unwrap_irf_3d(ub) for ub in irf.upper]
        lower_raw = similar(raw_lower)
        upper_raw = similar(raw_upper)
        for i in eachindex(raw_lower)
            lb = copy(raw_lower[i])
            ub = copy(raw_upper[i])
            _cumulate_point!(lb, cumulate_idx)
            _cumulate_point!(ub, cumulate_idx)
            lower_raw[i] = lb
            upper_raw[i] = ub
        end
        raw_stderr = _unwrap_irf_3d(irf.stderr)
        cum_stderr = similar(raw_stderr)
        fill!(cum_stderr, NaN)
    end

    # Re-wrap in AxisArrays
    irf_ax = _wrap_irf_3d(cum_point, names)
    stderr_ax = _wrap_irf_3d(cum_stderr, names)
    lower_ax = [_wrap_irf_3d(lb, names) for lb in lower_raw]
    upper_ax = [_wrap_irf_3d(ub, names) for ub in upper_raw]
    draws_ax = cum_draws === nothing ? nothing : _wrap_irf_4d(cum_draws, names)

    cumulate_syms = Symbol.(names[cumulate_idx])
    metadata = merge(irf.metadata, (cumulative = true, cumulated_vars = cumulate_syms))

    return IRFResult(irf_ax, stderr_ax, draws_ax, lower_ax, upper_ax, irf.coverage,
        irf.identification, irf.inference, metadata)
end

"""
    cumulative_irf(irf::SignRestrictedIRFResult; vars=nothing)

Compute cumulative impulse response functions for sign-restricted IRFs.

Cumulates draws first, then recomputes median and quantile bands correctly.

# Keyword Arguments
- `vars::Union{Nothing, Vector{Symbol}, Vector{Int}}=nothing`: Variables to cumulate.

# Returns
- New `SignRestrictedIRFResult` with cumulative IRFs
"""
function cumulative_irf(irf::SignRestrictedIRFResult; vars = nothing)
    names = varnames(irf)
    all_idx = collect(1:length(names))

    if vars === nothing
        cumulate_idx = all_idx
    else
        cumulate_idx = _resolve_cumulate(vars, names)
        cumulate_idx === nothing && return irf
    end

    # Unwrap, cumulate draws, recompute median and bands
    raw_draws = _unwrap_irf_4d(irf.irf_draws)
    cum_draws = copy(raw_draws)
    for r in axes(cum_draws, 1)
        _cumulate_point!(view(cum_draws,r,:,:,:), cumulate_idx)
    end

    cum_median = dropdims(median(cum_draws; dims = 1); dims = 1)

    Te = eltype(cum_draws)
    sz3 = size(cum_draws)[2:4]
    lower_raw = Vector{Array{Te, 3}}(undef, length(irf.coverage))
    upper_raw = Vector{Array{Te, 3}}(undef, length(irf.coverage))
    for (i, cov) in enumerate(irf.coverage)
        α = 1 - cov
        lo = zeros(Te, sz3...)
        hi = zeros(Te, sz3...)
        _quantile_along_dim1!(lo, hi, cum_draws, α / 2, 1 - α / 2)
        lower_raw[i] = lo
        upper_raw[i] = hi
    end

    # Re-wrap in AxisArrays
    cum_median_ax = _wrap_irf_3d(cum_median, names)
    cum_draws_ax = _wrap_irf_4d(cum_draws, names)
    lower_ax = [_wrap_irf_3d(lb, names) for lb in lower_raw]
    upper_ax = [_wrap_irf_3d(ub, names) for ub in upper_raw]

    cumulate_syms = Symbol.(names[cumulate_idx])
    metadata = merge(irf.metadata, (cumulative = true, cumulated_vars = cumulate_syms))

    return SignRestrictedIRFResult(cum_median_ax, cum_draws_ax, lower_ax, upper_ax,
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
function should_save_draws(
        inf::Union{
            WildBootstrap, Bootstrap, BlockBootstrap, ProxySVARMBB}, draws)
    inf.save_draws ? draws : nothing
end

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
        irf_point::Array{T, 3},
        inf::WildBootstrap,
        coverage::Vector{Float64},
        normalization::AbstractNormalization,
        rng::AbstractRNG
) where {T}
    horizon = size(irf_point, 1) - 1

    # Run wild bootstrap
    draws = bootstrap_irf_wild(model, identification, horizon, inf.reps, rng;
        normalization)

    # Compute bands from draws
    lower, upper = compute_bands_from_draws(irf_point, draws, coverage)

    # Compute stderr from draws
    stderr = dropdims(std(draws; dims = 1); dims = 1)

    return draws, stderr, lower, upper
end

# Method 2: Standard Bootstrap
function compute_inference_bands(
        model::VARModel{T},
        identification::AbstractIdentification,
        irf_point::Array{T, 3},
        inf::Bootstrap,
        coverage::Vector{Float64},
        normalization::AbstractNormalization,
        rng::AbstractRNG
) where {T}
    horizon = size(irf_point, 1) - 1
    draws = bootstrap_irf_standard(model, identification, horizon, inf.reps, rng;
        normalization)
    lower, upper = compute_bands_from_draws(irf_point, draws, coverage)
    stderr = dropdims(std(draws; dims = 1); dims = 1)

    return draws, stderr, lower, upper
end

# Method 3: Block Bootstrap
function compute_inference_bands(
        model::VARModel{T},
        identification::AbstractIdentification,
        irf_point::Array{T, 3},
        inf::BlockBootstrap,
        coverage::Vector{Float64},
        normalization::AbstractNormalization,
        rng::AbstractRNG
) where {T}
    horizon = size(irf_point, 1) - 1
    draws = bootstrap_irf_block(model, identification, horizon, inf.reps,
        inf.block_length, rng; normalization)
    lower, upper = compute_bands_from_draws(irf_point, draws, coverage)
    stderr = dropdims(std(draws; dims = 1); dims = 1)

    return draws, stderr, lower, upper
end

# Method 4: Analytic (Delta Method)
function compute_inference_bands(
        model::VARModel{T},
        identification::AbstractIdentification,
        irf_point::Array{T, 3},
        inf::Analytic,
        coverage::Vector{Float64},
        normalization::AbstractNormalization,
        rng::AbstractRNG  # Not used, but keep signature consistent
) where {T}

    # Delta method doesn't produce draws
    draws = nothing

    # Compute asymptotic standard errors
    P = rotation_matrix(model, identification)
    P = normalize(P, normalization)
    stderr = irf_asymptotic_stderror(model, P, irf_point)

    # Compute bands from normal approximation
    lower, upper = compute_bands_from_stderr(irf_point, stderr, coverage)

    return draws, stderr, lower, upper
end

# Method 5: No Inference
function compute_inference_bands(
        model::VARModel{T},
        identification::AbstractIdentification,
        irf_point::Array{T, 3},
        ::Nothing,
        coverage::Vector{Float64},
        ::AbstractNormalization,
        rng::AbstractRNG
) where {T}
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
function compute_bands_from_draws(irf_point::Array{T, 3}, draws::Array{T, 4},
        coverage::Vector{Float64}) where {T}
    lower = Vector{Array{T, 3}}(undef, length(coverage))
    upper = Vector{Array{T, 3}}(undef, length(coverage))

    # Compute centered bootstrap draws: draws - mean(draws) + irf_point
    draws_mean = dropdims(mean(draws; dims = 1); dims = 1)
    draws_centered = draws .- reshape(draws_mean, (1, size(draws_mean)...)) .+
                     reshape(irf_point, (1, size(irf_point)...))

    sz3 = size(draws_centered)[2:4]
    for (i, α) in enumerate(coverage)
        α_lower = (1 - α) / 2
        α_upper = 1 - α_lower

        lo = zeros(T, sz3...)
        hi = zeros(T, sz3...)
        _quantile_along_dim1!(lo, hi, draws_centered, α_lower, α_upper)
        lower[i] = lo
        upper[i] = hi
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
function compute_bands_from_stderr(irf_point::Array{T, 3}, stderr::Array{T, 3},
        coverage::Vector{Float64}) where {T}
    lower = Vector{Array{T, 3}}(undef, length(coverage))
    upper = Vector{Array{T, 3}}(undef, length(coverage))

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
        coverage::Vector{Float64} = [0.68, 0.90, 0.95]
) where {T}

    # Requires saved draws
    isnothing(irf.bootstrap_draws) &&
        error("Bootstrap draws not saved in IRFResult. " *
              "Pass model and identification to recompute, or use irf() with save_draws=true.")

    # Sort coverage
    coverage = sort(coverage)

    # Unwrap to raw arrays for band computation
    raw_point = _unwrap_irf_3d(irf.irf)
    raw_draws = _unwrap_irf_4d(irf.bootstrap_draws)

    # Recompute bands from saved draws
    lower_raw, upper_raw = compute_bands_from_draws(raw_point, raw_draws, coverage)

    # Re-wrap in AxisArrays
    names = varnames(irf)
    lower_ax = [_wrap_irf_3d(lb, names) for lb in lower_raw]
    upper_ax = [_wrap_irf_3d(ub, names) for ub in upper_raw]

    # Return new IRFResult with same draws but updated bands
    return IRFResult(irf.irf, irf.stderr, irf.bootstrap_draws, lower_ax, upper_ax,
        coverage, irf.identification, irf.inference, irf.metadata)
end

# Method 2: Recompute with new inference method (requires model)
function confidence_bands(
        irf::IRFResult{T},
        model::VARModel{T},
        identification::AbstractIdentification,
        inf::InferenceType;
        coverage::Vector{Float64} = [0.68, 0.90, 0.95],
        rng::AbstractRNG = Random.default_rng()
) where {T}

    # Sort coverage
    coverage = sort(coverage)

    # Unwrap point estimate to raw array for compute_inference_bands
    raw_point = _unwrap_irf_3d(irf.irf)

    # Dispatch to compute_inference_bands (works on raw arrays)
    draws, stderr_raw,
    lower_raw,
    upper_raw = compute_inference_bands(
        model, identification, raw_point, inf, coverage, rng
    )

    # Determine whether to save draws
    bootstrap_draws_raw = should_save_draws(inf, draws)

    # Re-wrap in AxisArrays
    names = Symbol.(model.names)
    stderr_ax = _wrap_irf_3d(stderr_raw, names)
    lower_ax = [_wrap_irf_3d(lb, names) for lb in lower_raw]
    upper_ax = [_wrap_irf_3d(ub, names) for ub in upper_raw]
    draws_ax = bootstrap_draws_raw === nothing ? nothing :
               _wrap_irf_4d(bootstrap_draws_raw, names)

    # Update metadata
    metadata = merge(irf.metadata, (
        inference_type = typeof(inf),
        timestamp = now()
    ))

    # Return new IRFResult
    return IRFResult(irf.irf, stderr_ax, draws_ax, lower_ax, upper_ax,
        coverage, identification, inf, metadata)
end
