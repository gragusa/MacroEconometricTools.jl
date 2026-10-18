# ============================================================================
# Impulse Response Function Computation
# ============================================================================

using AxisArrays: AxisArrays, AxisArray, Axis

# ============================================================================
# AxisArray Wrapping / Unwrapping Helpers
# ============================================================================

"""
    _wrap_irf_3d(raw, names) -> AxisArray

Wrap a raw `(n_vars, n_shocks, horizon+1)` array, without copying, into a 3D
AxisArray with axes `(:variable, :shock, :horizon)`.
"""
function _wrap_irf_3d(raw::Array{T, 3}, names::Vector{Symbol}) where {T}
    return AxisArray(
        raw,
        Axis{:variable}(names),
        Axis{:shock}(names),
        Axis{:horizon}(0:(size(raw, 3) - 1))
    )
end

"""
    _wrap_irf_4d(raw, names) -> AxisArray

Wrap a raw `(reps, n_vars, n_shocks, horizon+1)` array, without copying, into a
4D AxisArray with axes `(:draw, :variable, :shock, :horizon)`.
"""
function _wrap_irf_4d(raw::Array{T, 4}, names::Vector{Symbol}) where {T}
    return AxisArray(
        raw,
        Axis{:draw}(1:size(raw, 1)),
        Axis{:variable}(names),
        Axis{:shock}(names),
        Axis{:horizon}(0:(size(raw, 4) - 1))
    )
end

"""
    _unwrap_irf_3d(ax) -> Array{T,3}

Copy a 3D AxisArray `(variable, shock, horizon)` into a plain array of the same layout.
"""
_unwrap_irf_3d(ax::AxisArray) = Array(ax)

"""
    _unwrap_irf_4d(ax) -> Array{T,4}

Copy a 4D AxisArray `(draw, variable, shock, horizon)` into a plain array of the same layout.
"""
_unwrap_irf_4d(ax::AxisArray) = Array(ax)

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
function _quantile_along_dim1!(
        lo::Array{T, 3}, hi::Array{T, 3},
        draws::Array{T, 4}, α_lower::Float64, α_upper::Float64
    ) where {T}
    reps = size(draws, 1)
    buf = Vector{T}(undef, reps)
    idx_lo = clamp(round(Int, α_lower * reps), 1, reps)
    idx_hi = clamp(round(Int, α_upper * reps), 1, reps)

    d2, d3, d4 = size(draws, 2), size(draws, 3), size(draws, 4)
    return @inbounds for h in 1:d4, j in 1:d3, i in 1:d2
        for r in 1:reps
            buf[r] = draws[r, i, j, h]
        end
        sort!(buf)
        lo[i, j, h] = buf[idx_lo]
        hi[i, j, h] = buf[idx_hi]
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

In-place cumulative sum along the horizon dimension (dim 3) for selected variables.
Operates on raw arrays with shape (n_vars, n_shocks, horizon+1).
"""
function _cumulate_point!(irf::AbstractArray{T, 3}, idx::Vector{Int}) where {T}
    for v in idx
        for s in axes(irf, 2)
            for h in 2:size(irf, 3)
                irf[v, s, h] += irf[v, s, h - 1]
            end
        end
    end
    return
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
        error(
        "Cumulation requires bootstrap draws. " *
            "Use a bootstrap inference method (e.g., WildBootstrap(reps=1000))."
    )

    # Cumulate point estimate
    cum_point = copy(irf_point)
    _cumulate_point!(cum_point, idx)

    # Cumulate each bootstrap draw
    cum_draws = copy(draws)
    for r in axes(cum_draws, 1)
        _cumulate_point!(view(cum_draws, r, :, :, :), idx)
    end

    # Recompute bands from cumulated draws
    lower, upper = compute_bands_from_draws(cum_point, cum_draws, coverage)

    # Stderr from cumulated draws
    stderr = dropdims(std(cum_draws; dims = 1); dims = 1)

    return cum_point, cum_draws, stderr, lower, upper
end

# ============================================================================

# Checks shared by every `irf` method; returns the coverage levels sorted.
function _validate_irf_args(horizon::Integer, coverage::AbstractVector{<:Real})
    horizon > 0 || throw(ArgumentError("horizon must be positive"))
    all(0 .< coverage .< 1) || throw(ArgumentError("coverage levels must be in (0, 1)"))
    return sort(coverage)
end

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
function irf(
        model::VARModel{T}, identification::AbstractIdentification;
        horizon::Int = 24,
        inference::Union{Nothing, InferenceType} = nothing,
        coverage::Vector{Float64} = [0.68, 0.9, 0.95],
        normalization::AbstractNormalization = UnitStd(),
        cumulate::Union{Nothing, Vector{Symbol}, Vector{Int}} = nothing,
        scale::Real = 1,
        rng::AbstractRNG = Random.default_rng()
    ) where {T}
    coverage = _validate_irf_args(horizon, coverage)

    # Resolve cumulation indices
    cumulate_idx = _resolve_cumulate(cumulate, model.names)

    # Validate: analytic inference is incompatible with cumulation
    if cumulate_idx !== nothing && inference isa Analytic
        throw(
            ArgumentError(
                "Cumulation of IRFs is incompatible with Analytic() inference. " *
                    "The delta method does not apply to cumulated IRFs. " *
                    "Use a bootstrap method (e.g., WildBootstrap(reps=1000)) instead."
            )
        )
    end

    # Compute point estimate of IRF
    P = rotation_matrix(model, identification)
    P = normalize(P, normalization)
    impact_diagonal = T.(diag(P))
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
            irf_point, draws, coverage, cumulate_idx
        )
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
    n = n_vars(model)
    cumulate_syms = cumulate_idx === nothing ? nothing : Symbol.(model.names[cumulate_idx])
    metadata = (
        horizon = horizon,
        inference_type = typeof(inference),
        normalization = typeof(normalization),
        names = model.names,
        cumulated_vars = cumulate_syms,
        scale = fill(T(scale), n),
        impact_diagonal = impact_diagonal,
        timestamp = now(),
    )

    return IRFResult(
        irf_ax, stderr_ax, draws_ax, lower_ax, upper_ax, coverage,
        identification, inference, metadata
    )
end

# Convenience alias
impulse_response = irf

"""
    irf(model::VARModel{T, <:IVSVAR}; kwargs...)

IRFs of a model from the deprecated `fit(IVSVAR, ...)`, identified with the
instrument stored in the model. Equivalent to
`irf(model, IVIdentification(model.spec.instrument); kwargs...)`.
"""
function irf(model::VARModel{T, <:IVSVAR}; kwargs...) where {T}
    return irf(model, IVIdentification(model.spec.instrument); kwargs...)
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
- `scale::Real=1`: Scaling applied to the responses
- `parallel::Symbol=:none`: Parallelization of the rotation search (`:none` or `:distributed`)
- `rng::AbstractRNG=Random.default_rng()`: Random number generator

# Returns
- `SignRestrictedIRFResult`: IRF result with multiple draws and quantile bands
"""
function irf(
        model::VARModel, id::SignRestriction;
        max_attempts::Int = 10000,
        parallel::Symbol = :none,
        rng::AbstractRNG = Random.default_rng(),
        kwargs...
    )
    draw_rotation() = rotation_matrix(
        model, id; max_draws = max_attempts, parallel, verbose = false, rng
    )
    return _set_identified_irf(draw_rotation, model, id; max_attempts, kwargs...)
end

"""
    irf(model::VARModel, id::NarrativeRestriction; kwargs...)

Compute impulse response functions for narrative sign restriction identification.

Narrative restrictions are set-identifying, so this returns a
`SignRestrictedIRFResult` collecting `n_draws` rotation matrices that satisfy both
the sign restrictions on the responses and the narrative sign constraints on the
structural shocks.

# Keyword Arguments
- `n_draws::Int=1000`: Number of valid rotation draws to compute
- `max_attempts::Int=10000`: Maximum rotation draws per accepted draw
- `horizon::Int=24`: IRF horizon
- `coverage::Vector{Float64}=[0.68, 0.90, 0.95]`: Coverage levels for quantile bands
- `normalization::AbstractNormalization=UnitStd()`: Shock normalization
- `cumulate::Union{Nothing, Vector{Symbol}, Vector{Int}}=nothing`: Variables to cumulate
- `scale::Real=1`: Scaling applied to the responses
- `parallel::Symbol=:none`: Only `:none` is supported; other values throw
- `rng::AbstractRNG=Random.default_rng()`: Random number generator

Date-based narrative restrictions resolve against the dates stored on the model
(supplied via the `dates` keyword of `fit`/`VAR`); without stored dates the
restrictions are interpreted as integer residual-row indices.

# Returns
- `SignRestrictedIRFResult`: IRF result with multiple draws and quantile bands
"""
function irf(
        model::VARModel, id::NarrativeRestriction;
        max_attempts::Int = 10000,
        parallel::Symbol = :none,
        rng::AbstractRNG = Random.default_rng(),
        kwargs...
    )
    parallel === :none || throw(
        ArgumentError(
            "parallel = $(repr(parallel)) is not supported for NarrativeRestriction; use parallel = :none"
        )
    )
    draw_rotation() = rotation_matrix(model, id; max_draws = max_attempts, rng)
    return _set_identified_irf(draw_rotation, model, id; max_attempts, kwargs...)
end

"""
    _set_identified_irf(draw_rotation, model, id; n_draws, max_attempts, horizon,
                        coverage, normalization, cumulate, scale)

Shared implementation of `irf` for set-identifying schemes. `draw_rotation()`
returns one impact matrix accepted by the identification `id`; it is called
`n_draws` times. Returns a `SignRestrictedIRFResult` with the pointwise median
and quantile bands of the accepted draws.
"""
function _set_identified_irf(
        draw_rotation, model::VARModel{T}, id::AbstractIdentification;
        n_draws::Int = 1000,
        max_attempts::Int,
        horizon::Int = 24,
        coverage::Vector{Float64} = [0.68, 0.9, 0.95],
        normalization::AbstractNormalization = UnitStd(),
        cumulate::Union{Nothing, Vector{Symbol}, Vector{Int}} = nothing,
        scale::Real = 1
    ) where {T}
    coverage = _validate_irf_args(horizon, coverage)
    n_draws > 0 || throw(ArgumentError("n_draws must be positive"))
    max_attempts > 0 || throw(ArgumentError("max_attempts must be positive"))
    cumulate_idx = _resolve_cumulate(cumulate, model.names)

    n = n_vars(model)
    rotation_matrices = Vector{Matrix{T}}(undef, n_draws)
    irf_draws_raw = zeros(T, n_draws, n, n, horizon + 1)

    for i in 1:n_draws
        P = normalize(draw_rotation(), normalization)
        rotation_matrices[i] = P
        irf_draws_raw[i, :, :, :] = compute_irf_point(model, P, horizon)
    end

    if cumulate_idx !== nothing
        for i in 1:n_draws
            _cumulate_point!(view(irf_draws_raw, i, :, :, :), cumulate_idx)
        end
    end

    if scale != 1
        irf_draws_raw .*= T(scale)
    end

    # Pointwise quantiles of the (possibly cumulated and scaled) draws
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

    impact_diags = hcat([T.(diag(rotation_matrices[i])) for i in 1:n_draws]...)
    impact_diagonal = vec(median(impact_diags; dims = 2))

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
        scale = fill(T(scale), n),
        impact_diagonal = impact_diagonal,
        timestamp = now(),
    )

    return SignRestrictedIRFResult(
        irf_median_ax, irf_draws_ax, lower_ax, upper_ax,
        coverage, rotation_matrices, id, metadata
    )
end

"""
    compute_irf_point(model::VARModel, P::Matrix, horizon::Int)

Compute point estimate of structural IRFs.

# Returns
- Array of size (n_vars, n_shocks, horizon+1); `[i, j, h+1]` is the response
  of variable `i` to shock `j` at horizon `h`
"""
function compute_irf_point(model::VARModel{T}, P::Matrix{T}, horizon::Int) where {T}
    n_vars_val = n_vars(model)
    Φ = compute_ma_matrices(model.companion, horizon, n_vars_val, n_lags(model))

    # IRF_h = Φ_h * P, with Φ_0 = I
    irf_array = similar(Φ)
    for h in axes(Φ, 3)
        mul!(view(irf_array, :, :, h), view(Φ, :, :, h), P)
    end

    return irf_array
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
        io, "  Variables: ", join(varnames(irf), ", "), " (", n_v, " × Shocks: ", n_s, ")"
    )
    println(io, "  Inference: ", irf.inference)

    if !isempty(irf.coverage)
        println(io, "  Coverage: ", join(irf.coverage .* 100, "%, "), "%")
    end
    return if haskey(irf.metadata, :cumulated_vars) && irf.metadata.cumulated_vars !== nothing
        println(io, "  Cumulated: ", join(irf.metadata.cumulated_vars, ", "))
    end
end

function Base.show(io::IO, ::MIME"text/plain", irf::IRFResult)
    return show(io, irf)
end

function Base.show(io::IO, irf::SignRestrictedIRFResult{T}) where {T}
    n_v = n_vars(irf)
    n_s = n_shocks(irf)
    h = horizon(irf)
    nd = n_draws(irf)

    println(io, "SignRestrictedIRFResult{$T}")
    println(io, "  Identification: ", typeof(irf.identification))
    println(io, "  Horizon: ", h)
    println(
        io, "  Variables: ", join(varnames(irf), ", "), " (", n_v, " × Shocks: ", n_s, ")"
    )
    println(io, "  Draws: ", nd, " rotation matrices")

    if !isempty(irf.coverage)
        println(io, "  Coverage: ", join(irf.coverage .* 100, "%, "), "%")
    end
    return if haskey(irf.metadata, :cumulated_vars) && irf.metadata.cumulated_vars !== nothing
        println(io, "  Cumulated: ", join(irf.metadata.cumulated_vars, ", "))
    end
end

function Base.show(io::IO, ::MIME"text/plain", irf::SignRestrictedIRFResult)
    return show(io, irf)
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
            _cumulate_point!(view(cum_draws, r, :, :, :), cumulate_idx)
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

    return IRFResult(
        irf_ax, stderr_ax, draws_ax, lower_ax, upper_ax, irf.coverage,
        irf.identification, irf.inference, metadata
    )
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
        _cumulate_point!(view(cum_draws, r, :, :, :), cumulate_idx)
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

    return SignRestrictedIRFResult(
        cum_median_ax, cum_draws_ax, lower_ax, upper_ax,
        irf.coverage, irf.rotation_matrices,
        irf.identification, metadata
    )
end

# ============================================================================
# Rescaling — per-shock API
# ============================================================================

"""
    _build_shock_factors(names, T, pairs) -> Vector{T}

Build a per-shock scale vector from `Symbol => Real` pairs.
Shocks not mentioned get factor 1.
"""
function _build_shock_factors(
        names::Vector{Symbol}, ::Type{T},
        pairs::Tuple{Vararg{Pair{Symbol, <:Real}}}
    ) where {T}
    factors = ones(T, length(names))
    for (nm, val) in pairs
        idx = findfirst(==(nm), names)
        idx === nothing && error("Shock :$nm not found. Available shocks: $names")
        factors[idx] = T(val)
    end
    return factors
end

"""
    _apply_shock_factors_3d!(raw, factors)

Multiply raw `(variable, shock, horizon)` array by per-shock factors along dim 2.
"""
function _apply_shock_factors_3d!(raw::Array{T, 3}, factors::Vector{T}) where {T}
    for j in eachindex(factors)
        factors[j] == one(T) && continue
        raw[:, j, :] .*= factors[j]
    end
    return
end

"""
    _apply_shock_factors_4d!(raw, factors)

Multiply raw `(draw, variable, shock, horizon)` array by per-shock factors along dim 3.
"""
function _apply_shock_factors_4d!(raw::Array{T, 4}, factors::Vector{T}) where {T}
    for j in eachindex(factors)
        factors[j] == one(T) && continue
        raw[:, :, j, :] .*= factors[j]
    end
    return
end

function _update_scale_metadata(metadata::NamedTuple, factors::Vector{T}) where {T}
    old = haskey(metadata, :scale) ? metadata.scale : ones(T, length(factors))
    old_vec = old isa AbstractVector ? T.(old) : fill(T(old), length(factors))
    return merge(metadata, (scale = old_vec .* factors,))
end

"""
    rescale(irf::IRFResult, pairs::Pair{Symbol, <:Real}...) -> IRFResult

Return a new `IRFResult` with per-shock rescaling.  Each pair `shock => factor`
multiplies the specified shock column (all variables, all horizons) by `factor`.
Shocks not mentioned are left unchanged.

When bootstrap draws are available, confidence bands are **recomputed from
rescaled draws** (statistically correct).  Without draws, bands are rescaled
directly and a warning is emitted.

# Examples
```julia
# Flip the sign of the monetary policy shock
result2 = rescale(result, :MP => -1)

# Flip MP and convert supply shock to percentage points
result2 = rescale(result, :MP => -1, :Supply => 100)
```
"""
function rescale(irf::IRFResult, pairs::Pair{Symbol, <:Real}...)
    T = eltype(irf.irf)
    names = varnames(irf)
    factors = _build_shock_factors(names, T, pairs)

    raw_point = _unwrap_irf_3d(irf.irf) |> copy
    raw_stderr = _unwrap_irf_3d(irf.stderr) |> copy
    _apply_shock_factors_3d!(raw_point, factors)
    # stderr scales by |factor| per shock
    _apply_shock_factors_3d!(raw_stderr, abs.(factors))

    if irf.bootstrap_draws !== nothing
        raw_draws = _unwrap_irf_4d(irf.bootstrap_draws) |> copy
        _apply_shock_factors_4d!(raw_draws, factors)
        lower_raw, upper_raw = compute_bands_from_draws(raw_point, raw_draws, irf.coverage)
    else
        @warn "No bootstrap draws available. Rescaled confidence bands are approximate. " *
            "For correct bands, use irf() with save_draws=true."
        raw_draws = nothing
        lower_raw = [
            let lb = copy(_unwrap_irf_3d(l))
                _apply_shock_factors_3d!(lb, factors)
                lb
            end
                for l in irf.lower
        ]
        upper_raw = [
            let ub = copy(_unwrap_irf_3d(u))
                _apply_shock_factors_3d!(ub, factors)
                ub
            end
                for u in irf.upper
        ]
        # Negative factors swap lower/upper — fix with elementwise min/max
        if any(f -> f < 0, factors)
            for k in eachindex(lower_raw)
                lo, hi = lower_raw[k], upper_raw[k]
                lower_raw[k] = min.(lo, hi)
                upper_raw[k] = max.(lo, hi)
            end
        end
    end

    irf_ax = _wrap_irf_3d(raw_point, names)
    stderr_ax = _wrap_irf_3d(raw_stderr, names)
    lower_ax = [_wrap_irf_3d(lb, names) for lb in lower_raw]
    upper_ax = [_wrap_irf_3d(ub, names) for ub in upper_raw]
    draws_ax = raw_draws === nothing ? nothing : _wrap_irf_4d(raw_draws, names)

    metadata = _update_scale_metadata(irf.metadata, factors)

    return IRFResult(
        irf_ax, stderr_ax, draws_ax, lower_ax, upper_ax, irf.coverage,
        irf.identification, irf.inference, metadata
    )
end

"""
    rescale(irf::SignRestrictedIRFResult, pairs::Pair{Symbol, <:Real}...) -> SignRestrictedIRFResult

Per-shock rescaling for sign-restricted IRFs.  Draws are rescaled, then median
and quantile bands are recomputed.
"""
function rescale(irf::SignRestrictedIRFResult, pairs::Pair{Symbol, <:Real}...)
    T = eltype(irf.irf_median)
    names = varnames(irf)
    factors = _build_shock_factors(names, T, pairs)

    raw_draws = _unwrap_irf_4d(irf.irf_draws) |> copy
    _apply_shock_factors_4d!(raw_draws, factors)
    raw_median = dropdims(median(raw_draws; dims = 1); dims = 1)

    sz3 = size(raw_draws)[2:4]
    lower_raw = Vector{Array{T, 3}}(undef, length(irf.coverage))
    upper_raw = Vector{Array{T, 3}}(undef, length(irf.coverage))
    for (i, cov) in enumerate(irf.coverage)
        α = 1 - cov
        lo = zeros(T, sz3...)
        hi = zeros(T, sz3...)
        _quantile_along_dim1!(lo, hi, raw_draws, α / 2, 1 - α / 2)
        lower_raw[i] = lo
        upper_raw[i] = hi
    end

    median_ax = _wrap_irf_3d(raw_median, names)
    draws_ax = _wrap_irf_4d(raw_draws, names)
    lower_ax = [_wrap_irf_3d(lb, names) for lb in lower_raw]
    upper_ax = [_wrap_irf_3d(ub, names) for ub in upper_raw]

    metadata = _update_scale_metadata(irf.metadata, factors)

    return SignRestrictedIRFResult(
        median_ax, draws_ax, lower_ax, upper_ax,
        irf.coverage, irf.rotation_matrices, irf.identification, metadata
    )
end

"""
    rescale!(irf::IRFResult, pairs::Pair{Symbol, <:Real}...) -> IRFResult

In-place per-shock rescaling.  Mutates AxisArray contents and returns a new
`IRFResult` with updated metadata (the returned object must be used).
"""
function rescale!(irf::IRFResult, pairs::Pair{Symbol, <:Real}...)
    T = eltype(irf.irf)
    names = varnames(irf)
    factors = _build_shock_factors(names, T, pairs)

    # Mutate point estimate and stderr via raw views
    raw_point = _unwrap_irf_3d(irf.irf)
    raw_stderr = _unwrap_irf_3d(irf.stderr)
    for j in eachindex(factors)
        factors[j] == one(T) && continue
        # AxisArray shares memory with the unwrapped view after permutedims?
        # No — permutedims creates a copy.  Must write back via AxisArray indexing.
    end
    # Safer: operate on the AxisArrays directly (variable, shock, horizon)
    for j in eachindex(factors)
        factors[j] == one(T) && continue
        irf.irf[:, j, :] .*= factors[j]
        irf.stderr[:, j, :] .*= abs(factors[j])
    end

    if irf.bootstrap_draws !== nothing
        for j in eachindex(factors)
            factors[j] == one(T) && continue
            irf.bootstrap_draws[:, :, j, :] .*= factors[j]
        end
        raw_point_new = _unwrap_irf_3d(irf.irf)
        raw_draws_new = _unwrap_irf_4d(irf.bootstrap_draws)
        lower_raw,
            upper_raw = compute_bands_from_draws(raw_point_new, raw_draws_new, irf.coverage)
        lower_ax = [_wrap_irf_3d(lb, names) for lb in lower_raw]
        upper_ax = [_wrap_irf_3d(ub, names) for ub in upper_raw]
    else
        @warn "No bootstrap draws available. Rescaled confidence bands are approximate. " *
            "For correct bands, use irf() with save_draws=true."
        for j in eachindex(factors)
            factors[j] == one(T) && continue
            for lb in irf.lower
                lb[:, j, :] .*= factors[j]
            end
            for ub in irf.upper
                ub[:, j, :] .*= factors[j]
            end
        end
        # Negative factors swap lower/upper — fix with elementwise min/max
        if any(f -> f < 0, factors)
            for k in eachindex(irf.lower)
                lo = Array(irf.lower[k])
                hi = Array(irf.upper[k])
                irf.lower[k] .= min.(lo, hi)
                irf.upper[k] .= max.(lo, hi)
            end
        end
        lower_ax = irf.lower
        upper_ax = irf.upper
    end

    metadata = _update_scale_metadata(irf.metadata, factors)

    return IRFResult(
        irf.irf, irf.stderr, irf.bootstrap_draws,
        lower_ax, upper_ax, irf.coverage,
        irf.identification, irf.inference, metadata
    )
end

"""
    rescale!(irf::SignRestrictedIRFResult, pairs::Pair{Symbol, <:Real}...) -> SignRestrictedIRFResult

In-place per-shock rescaling for sign-restricted IRFs.
"""
function rescale!(irf::SignRestrictedIRFResult, pairs::Pair{Symbol, <:Real}...)
    T = eltype(irf.irf_median)
    names = varnames(irf)
    factors = _build_shock_factors(names, T, pairs)

    # Mutate draws via AxisArray (draw, variable, shock, horizon)
    for j in eachindex(factors)
        factors[j] == one(T) && continue
        irf.irf_draws[:, :, j, :] .*= factors[j]
    end

    # Recompute median and bands
    raw_draws = _unwrap_irf_4d(irf.irf_draws)
    raw_median = dropdims(median(raw_draws; dims = 1); dims = 1)

    sz3 = size(raw_draws)[2:4]
    lower_raw = Vector{Array{T, 3}}(undef, length(irf.coverage))
    upper_raw = Vector{Array{T, 3}}(undef, length(irf.coverage))
    for (i, cov) in enumerate(irf.coverage)
        α = 1 - cov
        lo = zeros(T, sz3...)
        hi = zeros(T, sz3...)
        _quantile_along_dim1!(lo, hi, raw_draws, α / 2, 1 - α / 2)
        lower_raw[i] = lo
        upper_raw[i] = hi
    end

    irf.irf_median .= _wrap_irf_3d(raw_median, names)
    lower_ax = [_wrap_irf_3d(lb, names) for lb in lower_raw]
    upper_ax = [_wrap_irf_3d(ub, names) for ub in upper_raw]

    metadata = _update_scale_metadata(irf.metadata, factors)

    return SignRestrictedIRFResult(
        irf.irf_median, irf.irf_draws, lower_ax, upper_ax,
        irf.coverage, irf.rotation_matrices, irf.identification, metadata
    )
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
            WildBootstrap, Bootstrap, BlockBootstrap, ProxySVARMBB,
        }, draws
    )
    return inf.save_draws ? draws : nothing
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
    horizon = size(irf_point, 3) - 1

    # Run wild bootstrap
    draws = bootstrap_irf_wild(
        model, identification, horizon, inf.reps, rng;
        normalization, max_failure_rate = inf.max_failure_rate
    )

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
    horizon = size(irf_point, 3) - 1
    draws = bootstrap_irf_standard(
        model, identification, horizon, inf.reps, rng;
        normalization, max_failure_rate = inf.max_failure_rate
    )
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
    horizon = size(irf_point, 3) - 1
    draws = bootstrap_irf_block(
        model, identification, horizon, inf.reps,
        inf.block_length, rng; normalization, max_failure_rate = inf.max_failure_rate
    )
    lower, upper = compute_bands_from_draws(irf_point, draws, coverage)
    stderr = dropdims(std(draws; dims = 1); dims = 1)

    return draws, stderr, lower, upper
end

# The delta method differentiates the lower-triangular Cholesky factor of Σ, so it
# is valid only for `CholeskyID` in the model's variable order.
function _require_default_cholesky(identification::AbstractIdentification)
    identification isa CholeskyID && identification.ordering === nothing && return nothing
    got = identification isa CholeskyID ?
        "CholeskyID with ordering $(identification.ordering)" :
        string(nameof(typeof(identification)))
    throw(
        ArgumentError(
            "Analytic() inference is valid only for CholeskyID() with the model's " *
                "variable order (got $got). Use a bootstrap " *
                "(WildBootstrap, Bootstrap, BlockBootstrap), or reorder the variables " *
                "when fitting the model."
        )
    )
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

    _require_default_cholesky(identification)

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
- `irf_point::Array{T,3}`: Point estimate IRF (n_vars, n_shocks, horizon+1)
- `draws::Array{T,4}`: Bootstrap draws (reps, n_vars, n_shocks, horizon+1)
- `coverage::Vector{Float64}`: Coverage levels (e.g., [0.68, 0.90, 0.95])

# Returns
- `(lower, upper)` tuple of vectors, one entry per coverage level
"""
function compute_bands_from_draws(
        irf_point::Array{T, 3}, draws::Array{T, 4},
        coverage::Vector{Float64}
    ) where {T}
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
function compute_bands_from_stderr(
        irf_point::Array{T, 3}, stderr::Array{T, 3},
        coverage::Vector{Float64}
    ) where {T}
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
        coverage::Vector{Float64} = [0.68, 0.9, 0.95]
    ) where {T}

    # Requires saved draws
    isnothing(irf.bootstrap_draws) &&
        error(
        "Bootstrap draws not saved in IRFResult. " *
            "Pass model and identification to recompute, or use irf() with save_draws=true."
    )

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
    return IRFResult(
        irf.irf, irf.stderr, irf.bootstrap_draws, lower_ax, upper_ax,
        coverage, irf.identification, irf.inference, irf.metadata
    )
end

# Method 2: Recompute with new inference method (requires model)
function confidence_bands(
        irf::IRFResult{T},
        model::VARModel{T},
        identification::AbstractIdentification,
        inf::InferenceType;
        coverage::Vector{Float64} = [0.68, 0.9, 0.95],
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
    metadata = merge(
        irf.metadata, (
            inference_type = typeof(inf),
            timestamp = now(),
        )
    )

    # Return new IRFResult
    return IRFResult(
        irf.irf, stderr_ax, draws_ax, lower_ax, upper_ax,
        coverage, identification, inf, metadata
    )
end
