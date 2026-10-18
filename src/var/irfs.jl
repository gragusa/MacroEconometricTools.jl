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
    draws === nothing && throw(
        ArgumentError(
            "Cumulation requires bootstrap draws. " *
                "Use a bootstrap inference method (e.g., WildBootstrap(reps=1000))."
        )
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
  Requires a bootstrap inference method — incompatible with `Analytic()` and with
  Anderson-Rubin bands (`ProxySVARMBB(compute_ar = true)`).
- `scale`: deprecated; use [`rescale`](@ref) on the result
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
        scale::Union{Nothing, Real} = nothing,
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
    cumulate_idx === nothing || _check_cumulable(inference)

    # Compute point estimate of IRF
    P = rotation_matrix(model, identification)
    P = normalize_impact(P, normalization)
    impact_diagonal = T.(diag(P))
    irf_point = compute_irf_point(model, P, horizon)

    # Dispatch on inference type - NO if-statements!
    draws, stderr,
        lower,
        upper = compute_inference_bands(
        model, identification, irf_point, inference, coverage, normalization, rng
    )

    # Apply cumulation if requested (before wrapping)
    if cumulate_idx !== nothing
        irf_point, draws, stderr,
            lower, upper = _apply_cumulation(
            irf_point, draws, coverage, cumulate_idx
        )
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
        scale = ones(T, n),
        impact_diagonal = impact_diagonal,
        timestamp = now(),
    )

    result = IRFResult(
        irf_ax, stderr_ax, draws_ax, lower_ax, upper_ax, coverage,
        identification, inference, metadata
    )
    return _deprecated_scale(result, scale)
end

@deprecate impulse_response(args...; kwargs...) irf(args...; kwargs...) false

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
- `scale`: deprecated; use [`rescale`](@ref) on the result
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
- `scale`: deprecated; use [`rescale`](@ref) on the result
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
                        coverage, normalization, cumulate, scale = nothing)

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
        scale::Union{Nothing, Real} = nothing,
        inference = nothing
    ) where {T}
    inference === nothing || throw(
        ArgumentError(
            "$(nameof(typeof(id))) is set-identifying and takes no `inference` keyword; " *
                "the bands of the returned SignRestrictedIRFResult are quantiles across " *
                "accepted rotations (control them with `n_draws` and `coverage`)"
        )
    )
    coverage = _validate_irf_args(horizon, coverage)
    n_draws > 0 || throw(ArgumentError("n_draws must be positive"))
    max_attempts > 0 || throw(ArgumentError("max_attempts must be positive"))
    cumulate_idx = _resolve_cumulate(cumulate, model.names)

    n = n_vars(model)
    rotation_matrices = Vector{Matrix{T}}(undef, n_draws)
    irf_draws_raw = zeros(T, n_draws, n, n, horizon + 1)

    for i in 1:n_draws
        P = normalize_impact(draw_rotation(), normalization)
        rotation_matrices[i] = P
        irf_draws_raw[i, :, :, :] = compute_irf_point(model, P, horizon)
    end

    if cumulate_idx !== nothing
        for i in 1:n_draws
            _cumulate_point!(view(irf_draws_raw, i, :, :, :), cumulate_idx)
        end
    end

    # Pointwise median and quantiles of the (possibly cumulated) draws
    irf_median_raw, lower_raw, upper_raw = _median_and_quantile_bands(irf_draws_raw, coverage)

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
        scale = ones(T, n),
        impact_diagonal = impact_diagonal,
        timestamp = now(),
    )

    result = SignRestrictedIRFResult(
        irf_median_ax, irf_draws_ax, lower_ax, upper_ax,
        coverage, rotation_matrices, id, metadata
    )
    return _deprecated_scale(result, scale)
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

# Anderson-Rubin bands are confidence sets for the uncumulated responses; a
# cumulated response would need its own test statistic.
function _check_cumulable(inference)
    inference isa ProxySVARMBB && inference.compute_ar && throw(
        ArgumentError(
            "Cumulation is not supported with Anderson-Rubin bands " *
                "(ProxySVARMBB(compute_ar = true)); use ProxySVARMBB(reps) for " *
                "percentile bands of the cumulated responses."
        )
    )
    return nothing
end

"""
    _post_hoc_cumulate(irf, vars) -> (idx, cumulated_vars)

Positions of the variables `vars` to cumulate (all variables when `vars === nothing`)
and the names of every cumulated variable afterwards. `idx` is `nothing` when `vars`
is empty. Throws when a variable in `vars` is already cumulated.
"""
function _post_hoc_cumulate(irf::AbstractIRFResult, vars)
    names = varnames(irf)
    idx = vars === nothing ? collect(eachindex(names)) : _resolve_cumulate(vars, names)
    previous = get(irf.metadata, :cumulated_vars, nothing)
    previous = previous === nothing ? Symbol[] : Symbol.(previous)
    idx === nothing && return nothing, previous
    again = intersect(Symbol.(names[idx]), previous)
    isempty(again) || throw(ArgumentError("Variables $again are already cumulated"))
    cumulated = Symbol.(names[sort(union(idx, findall(in(previous), Symbol.(names))))])
    return idx, cumulated
end

"""
    cumulative_irf(irf::IRFResult; vars=nothing)

Cumulate the responses of the variables `vars` (names or positions; all variables
when `nothing`) along the horizon. Bands and standard errors are recomputed from the
cumulated draws, so the result agrees with `irf(model, id; cumulate = vars, ...)`.

Requires stored draws (an inference method with `save_draws = true`); throws
otherwise, and when the result has Anderson-Rubin bands or a variable in `vars` is
already cumulated.

# Returns
- New `IRFResult` with cumulative IRFs
"""
function cumulative_irf(irf::IRFResult; vars = nothing)
    has_draws(irf) || throw(
        ArgumentError(
            "cumulative_irf requires stored bootstrap draws to recompute the bands; use " *
                "an inference method with save_draws = true, or irf(...; cumulate = vars)."
        )
    )
    _check_cumulable(irf.inference)
    cumulate_idx, cumulated_vars = _post_hoc_cumulate(irf, vars)
    cumulate_idx === nothing && return irf

    cum_point, cum_draws, cum_stderr, lower_raw,
        upper_raw = _apply_cumulation(
        _unwrap_irf_3d(irf.irf), _unwrap_irf_4d(irf.bootstrap_draws),
        irf.coverage, cumulate_idx
    )

    names = varnames(irf)
    irf_ax = _wrap_irf_3d(cum_point, names)
    stderr_ax = _wrap_irf_3d(cum_stderr, names)
    lower_ax = [_wrap_irf_3d(lb, names) for lb in lower_raw]
    upper_ax = [_wrap_irf_3d(ub, names) for ub in upper_raw]
    draws_ax = _wrap_irf_4d(cum_draws, names)
    metadata = merge(irf.metadata, (; cumulated_vars))

    return IRFResult(
        irf_ax, stderr_ax, draws_ax, lower_ax, upper_ax, irf.coverage,
        irf.identification, irf.inference, metadata
    )
end

"""
    cumulative_irf(irf::SignRestrictedIRFResult; vars=nothing)

Cumulate the responses of the variables `vars` (names or positions; all variables
when `nothing`) in every accepted draw, then recompute the median and quantile bands.
Throws when a variable in `vars` is already cumulated.

# Returns
- New `SignRestrictedIRFResult` with cumulative IRFs
"""
function cumulative_irf(irf::SignRestrictedIRFResult; vars = nothing)
    cumulate_idx, cumulated_vars = _post_hoc_cumulate(irf, vars)
    cumulate_idx === nothing && return irf

    cum_draws = _unwrap_irf_4d(irf.irf_draws)
    for r in axes(cum_draws, 1)
        _cumulate_point!(view(cum_draws, r, :, :, :), cumulate_idx)
    end
    cum_median, lower_raw, upper_raw = _median_and_quantile_bands(cum_draws, irf.coverage)

    names = varnames(irf)
    cum_median_ax = _wrap_irf_3d(cum_median, names)
    cum_draws_ax = _wrap_irf_4d(cum_draws, names)
    lower_ax = [_wrap_irf_3d(lb, names) for lb in lower_raw]
    upper_ax = [_wrap_irf_3d(ub, names) for ub in upper_raw]
    metadata = merge(irf.metadata, (; cumulated_vars))

    return SignRestrictedIRFResult(
        cum_median_ax, cum_draws_ax, lower_ax, upper_ax,
        irf.coverage, irf.rotation_matrices,
        irf.identification, metadata
    )
end

"""
    _median_and_quantile_bands(draws, coverage) -> (median, lower, upper)

Pointwise median and equal-tailed quantile bands at each level in `coverage` of
`draws` indexed `(draw, variable, shock, horizon)`.
"""
function _median_and_quantile_bands(draws::Array{T, 4}, coverage::Vector{Float64}) where {T}
    med = dropdims(median(draws; dims = 1); dims = 1)
    sz3 = size(draws)[2:4]
    lower = Vector{Array{T, 3}}(undef, length(coverage))
    upper = Vector{Array{T, 3}}(undef, length(coverage))
    for (i, cov) in enumerate(coverage)
        α = 1 - cov
        lo = zeros(T, sz3...)
        hi = zeros(T, sz3...)
        _quantile_along_dim1!(lo, hi, draws, α / 2, 1 - α / 2)
        lower[i] = lo
        upper[i] = hi
    end
    return med, lower, upper
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
        idx === nothing &&
            throw(ArgumentError("Shock :$nm not found. Available shocks: $names"))
        factors[idx] = T(val)
    end
    return factors
end

# Multiply the shock dimension `dim` of `x` by per-shock `factors`.
function _scale_shocks(x::AxisArray, factors::AbstractVector, dim::Integer)
    raw = Array(x)
    for j in eachindex(factors)
        selectdim(raw, dim, j) .*= factors[j]
    end
    return AxisArray(raw, AxisArrays.axes(x)...)
end

# Scaling a shock by `c` maps every band this package computes (percentiles of
# centered draws, normal bands from standard errors, Anderson-Rubin hulls) to the
# band of the scaled responses; a negative `c` swaps lower and upper.
function _scale_bands(lower::Vector, upper::Vector, factors::AbstractVector)
    lo = [_scale_shocks(l, factors, 2) for l in lower]
    hi = [_scale_shocks(u, factors, 2) for u in upper]
    for k in eachindex(lo, hi), j in eachindex(factors)
        factors[j] < 0 || continue
        l, u = copy(lo[k][:, j, :]), copy(hi[k][:, j, :])
        lo[k][:, j, :] = u
        hi[k][:, j, :] = l
    end
    return lo, hi
end

function _update_scale_metadata(metadata::NamedTuple, factors::Vector{T}) where {T}
    old = haskey(metadata, :scale) ? metadata.scale : ones(T, length(factors))
    old_vec = old isa AbstractVector ? T.(old) : fill(T(old), length(factors))
    return merge(metadata, (scale = old_vec .* factors,))
end

"""
    rescale(irf, shock => factor, ...) -> typeof(irf)
    rescale(irf, factor::Real) -> typeof(irf)

Return a copy of `irf` (an `IRFResult` or `SignRestrictedIRFResult`) with the
responses to each named shock multiplied by `factor`; the second form multiplies the
responses to every shock. Shocks not mentioned are unchanged. Point estimates, draws,
and bands are multiplied by the factor (lower and upper swap when it is negative),
standard errors by its absolute value. The cumulative factors are recorded in the
result's metadata and reported by [`get_scale`](@ref).

# Examples
```julia
# Flip the sign of the monetary policy shock
result2 = rescale(result, :MP => -1)

# Flip MP and convert supply shock to percentage points
result2 = rescale(result, :MP => -1, :Supply => 100)

# Express every response in percent
result_pct = rescale(result, 100)
```
"""
function rescale(
        irf::Union{IRFResult, SignRestrictedIRFResult}, pairs::Pair{Symbol, <:Real}...
    )
    T = eltype(point_estimate(irf))
    return _rescale(irf, _build_shock_factors(shocknames(irf), T, pairs))
end

function rescale(irf::Union{IRFResult, SignRestrictedIRFResult}, factor::Real)
    T = eltype(point_estimate(irf))
    return _rescale(irf, fill(T(factor), n_shocks(irf)))
end

function _rescale(irf::IRFResult, factors::Vector)
    all(isfinite, factors) || throw(ArgumentError("scale factors must be finite, got $factors"))
    lower, upper = _scale_bands(irf.lower, irf.upper, factors)
    draws = irf.bootstrap_draws === nothing ? nothing :
        _scale_shocks(irf.bootstrap_draws, factors, 3)
    return IRFResult(
        _scale_shocks(irf.irf, factors, 2), _scale_shocks(irf.stderr, abs.(factors), 2),
        draws, lower, upper, irf.coverage, irf.identification, irf.inference,
        _update_scale_metadata(irf.metadata, factors)
    )
end

function _rescale(irf::SignRestrictedIRFResult, factors::Vector)
    all(isfinite, factors) || throw(ArgumentError("scale factors must be finite, got $factors"))
    lower, upper = _scale_bands(irf.lower, irf.upper, factors)
    return SignRestrictedIRFResult(
        _scale_shocks(irf.irf_median, factors, 2), _scale_shocks(irf.irf_draws, factors, 3),
        lower, upper, irf.coverage, irf.rotation_matrices, irf.identification,
        _update_scale_metadata(irf.metadata, factors)
    )
end

# `scale` keyword of `irf`, deprecated in favor of `rescale`.
_deprecated_scale(result, ::Nothing) = result
function _deprecated_scale(result, scale::Real)
    Base.depwarn(
        "The `scale` keyword of `irf` is deprecated; use `rescale(irf(...), $scale)`.",
        :irf
    )
    return rescale(result, scale)
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

# Methods 1-3: residual bootstraps (wild, standard, block)
function compute_inference_bands(
        model::VARModel{T},
        identification::AbstractIdentification,
        irf_point::Array{T, 3},
        inf::_ResidualBootstrap,
        coverage::Vector{Float64},
        normalization::AbstractNormalization,
        rng::AbstractRNG
    ) where {T}
    horizon = size(irf_point, 3) - 1
    draws = _bootstrap_irf_draws(model, identification, inf, horizon, normalization, rng)
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
    P = normalize_impact(P, normalization)
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
    confidence_bands(irf::IRFResult; coverage=[0.68, 0.90, 0.95])

Recompute the percentile bands of `irf` at new coverage levels from its saved
bootstrap draws. The point estimate, standard errors, draws, identification, and
inference settings are unchanged. Rescaled or cumulated results work as well,
because `rescale` and `cumulative_irf` transform the saved draws.

Throws an `ArgumentError` when `irf` has no saved draws (compute it with
`save_draws = true`) or when its bands are Anderson-Rubin confidence sets
(`ProxySVARMBB(compute_ar = true)`), which are not percentile bands of the draws.
To change the inference method, call [`irf`](@ref) again.

# Examples
```julia
irf1 = irf(model, CholeskyID(); inference = WildBootstrap(reps = 1000, save_draws = true))
irf2 = confidence_bands(irf1; coverage = [0.75, 0.85, 0.95])
```
"""
function confidence_bands(
        irf::IRFResult{T};
        coverage::Vector{Float64} = [0.68, 0.9, 0.95]
    ) where {T}
    isnothing(irf.bootstrap_draws) && throw(
        ArgumentError(
            "the IRFResult has no saved bootstrap draws; compute it with " *
                "save_draws = true (e.g. inference = WildBootstrap(reps; save_draws = true))."
        )
    )
    irf.inference isa ProxySVARMBB && irf.inference.compute_ar && throw(
        ArgumentError(
            "the bands of this IRFResult are Anderson-Rubin confidence sets " *
                "(ProxySVARMBB(compute_ar = true)), not percentile bands of the draws; " *
                "call irf(model, id; inference = ProxySVARMBB(...; compute_ar = true), " *
                "coverage) to compute them at other levels."
        )
    )
    all(0 .< coverage .< 1) || throw(ArgumentError("coverage levels must be in (0, 1)"))
    coverage = sort(coverage)

    raw_point = _unwrap_irf_3d(irf.irf)
    raw_draws = _unwrap_irf_4d(irf.bootstrap_draws)
    lower_raw, upper_raw = compute_bands_from_draws(raw_point, raw_draws, coverage)

    names = varnames(irf)
    lower_ax = [_wrap_irf_3d(lb, names) for lb in lower_raw]
    upper_ax = [_wrap_irf_3d(ub, names) for ub in upper_raw]

    return IRFResult(
        irf.irf, irf.stderr, irf.bootstrap_draws, lower_ax, upper_ax,
        coverage, irf.identification, irf.inference, irf.metadata
    )
end

function confidence_bands(
        irf::IRFResult, ::Type{<:_ResidualBootstrap};
        coverage::Vector{Float64} = [0.68, 0.9, 0.95]
    )
    Base.depwarn(
        "`confidence_bands(irf, InferenceType; coverage)` is deprecated; " *
            "use `confidence_bands(irf; coverage)`.", :confidence_bands
    )
    return confidence_bands(irf; coverage)
end

function confidence_bands(
        result::IRFResult, model::VARModel, identification::AbstractIdentification,
        inf::InferenceType;
        coverage::Vector{Float64} = [0.68, 0.9, 0.95],
        rng::AbstractRNG = Random.default_rng()
    )
    Base.depwarn(
        "`confidence_bands(irf, model, identification, inference)` is deprecated; " *
            "use `irf(model, identification; horizon, inference, coverage, normalization, rng)`.",
        :confidence_bands
    )
    # `irf` recomputes the result from the model; it cannot reproduce a later
    # `rescale` or `cumulative_irf`.
    cumulated = get(result.metadata, :cumulated_vars, nothing)
    scale = get(result.metadata, :scale, nothing)
    cumulated === nothing && (scale === nothing || all(isone, scale)) || throw(
        ArgumentError(
            "the IRFResult was rescaled or cumulated; call irf(model, identification; " *
                "inference, coverage, ...) and apply rescale/cumulative_irf to the new result."
        )
    )
    return irf(
        model, identification;
        horizon = horizon(result), inference = inf, coverage,
        normalization = result.metadata.normalization(), rng
    )
end
