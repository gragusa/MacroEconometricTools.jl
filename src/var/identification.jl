# ============================================================================
# Structural Identification Schemes
# ============================================================================

"""
    rotation_matrix(model::VARModel, id::AbstractIdentification; kwargs...)

Compute structural rotation/impact matrix from identification scheme.

The rotation matrix `P` relates structural shocks ε to reduced-form shocks u via u = P*ε.

# Arguments
- `model::VARModel`: Estimated VAR model
- `id::AbstractIdentification`: Identification scheme

# Keyword Arguments
(For SignRestriction only)
- `max_draws::Int=10000`: Maximum number of rotation draws
- `parallel::Symbol=:none`: Parallelization (`:none`, `:distributed`)
- `verbose::Bool=false`: Print progress information
- `rng::AbstractRNG=Random.default_rng()`: Random number generator

# Returns
- Matrix `P`: Structural rotation/impact matrix

# Examples
```julia
# Cholesky identification
P = rotation_matrix(var_model, CholeskyID())

# Sign restrictions with reproducible draws
P = rotation_matrix(var_model, sign_id; max_draws=5000, rng=StableRNG(42))
```
"""
function rotation_matrix end

# ============================================================================
# Cholesky Identification
# ============================================================================

"""
    rotation_matrix(model::VARModel, id::CholeskyID)

Recursive (Cholesky) identification.

Computes lower-triangular Cholesky factor: Σ = P*P'

# Arguments
- `model::VARModel`: Estimated VAR model
- `id::CholeskyID`: Cholesky identification with optional variable ordering

# Returns
- Rotation matrix `P` with `P * P' ≈ Σ`; lower triangular in the data ordering
  when `id.ordering` is `nothing`. Computed by [`structural_impact`](@ref).
"""
function rotation_matrix(model::VARModel{T}, id::CholeskyID) where {T}
    Σ_L = Matrix(cholesky(residual_covariance(model)).L)
    ordering_idx = id.ordering === nothing ? nothing :
        get_permutation(model.names, id.ordering)
    return structural_impact(Σ_L, id; ordering_idx)
end

"""
    get_permutation(names::Vector{Symbol}, ordering::Vector{Int})

Validate an integer ordering against the number of variables and return it.
"""
function get_permutation(names::Vector{Symbol}, ordering::Vector{Int})
    sort(ordering) == collect(1:length(names)) ||
        throw(ArgumentError("Integer ordering must be a permutation of 1:$(length(names))"))
    return ordering
end

"""
    get_permutation(names::Vector{Symbol}, ordering::Vector{Symbol})

Get permutation vector to reorder variables according to `ordering`.
"""
function get_permutation(names::Vector{Symbol}, ordering::Vector{Symbol})
    length(names) == length(ordering) ||
        throw(ArgumentError("Ordering must include all variables"))

    Set(names) == Set(ordering) ||
        throw(ArgumentError("Ordering must contain same variables as data"))

    return [findfirst(==(v), names) for v in ordering]
end

# ============================================================================
# Sign Restrictions
# ============================================================================

"""
    rotation_matrix(model::VARModel, id::SignRestriction; kwargs...)

Identification via sign restrictions.

Uses algorithm of Rubio-Ramírez, Waggoner, and Zha (2010).

# Arguments
- `model::VARModel`: Estimated VAR model
- `id::SignRestriction`: Sign restriction specification

# Keyword Arguments
- `max_draws::Int=10000`: Maximum number of rotation draws
- `parallel::Symbol=:none`: Parallelization (`:none`, `:distributed`)
- `verbose::Bool=false`: Print progress information
- `rng::AbstractRNG=Random.default_rng()`: Random number generator for reproducible draws

# Returns
- Rotation matrix satisfying sign restrictions

# Examples
```julia
# Serial search
P = rotation_matrix(model, id; max_draws=10000)

# Distributed search (requires Distributed + workers)
using Distributed
addprocs(4)
@everywhere using MacroEconometricTools
P = rotation_matrix(model, id; max_draws=50000, parallel=:distributed)
```
"""
function rotation_matrix(
        model::VARModel{T}, id::SignRestriction;
        max_draws::Int = 10000,
        parallel::Symbol = :none,
        verbose::Bool = false,
        rng::AbstractRNG = Random.default_rng()
    ) where {T}
    parallel ∈ [:none, :distributed] ||
        throw(ArgumentError("parallel must be :none or :distributed"))

    if parallel == :distributed
        return identify_sign_distributed(model, id, max_draws, verbose, rng)
    else
        return identify_sign_serial(model, id, max_draws, verbose, rng)
    end
end

"""
    identify_sign_serial(model, id, max_draws, verbose)

Serial sign restriction search.
"""
function identify_sign_serial(
        model::VARModel{T}, id::SignRestriction,
        max_draws::Int, verbose::Bool,
        rng::AbstractRNG
    ) where {T}
    n_vars_val = n_vars(model)
    Σ = residual_covariance(model)

    # Cholesky as starting point
    P_chol = Matrix(cholesky(Σ).L)
    Φ = compute_ma_matrices(model.companion, id.horizon, n_vars_val, n_lags(model))

    # Search for valid rotation
    for iter in 1:max_draws
        # Random orthogonal matrix via QR decomposition
        Q = generate_random_orthogonal(n_vars_val, rng)

        # Candidate impact matrix
        P_candidate = P_chol * Q

        # Check sign restrictions on impact or IRFs
        if check_sign_restrictions(P_candidate, id.restrictions, Φ, id.horizon)
            if verbose
                println("Found valid rotation at attempt $iter")
            end
            return P_candidate
        end

        if verbose && iter % 1000 == 0
            println("Completed $iter draws, no valid rotation found yet...")
        end
    end

    throw(ErrorException("Could not find impact matrix satisfying sign restrictions after $max_draws attempts"))
end

"""
    identify_sign_distributed(model, id, max_draws, verbose)

Distributed sign restriction search using multiple processes.

Divides search across workers and stops when first valid rotation is found.
"""
function identify_sign_distributed(
        model::VARModel{T}, id::SignRestriction,
        max_draws::Int, verbose::Bool,
        rng::AbstractRNG
    ) where {T}

    # Check if Distributed is available
    # if !isdefined(Main, :Distributed)
    #     @warn "Distributed package not loaded. Falling back to serial execution."
    #     return identify_sign_serial(model, id, max_draws, verbose, rng)
    # end

    #dist = Base.require(Main, :Distributed)

    if Distributed.nworkers() == 1
        @warn "No worker processes available. Falling back to serial execution."
        return identify_sign_serial(model, id, max_draws, verbose, rng)
    end

    n_vars_val = n_vars(model)
    Σ = residual_covariance(model)
    P_chol = Matrix(cholesky(Σ).L)
    Φ = compute_ma_matrices(model.companion, id.horizon, n_vars_val, n_lags(model))

    # Function to search for valid rotation (returns nothing if not found)
    # Takes (n_attempts, worker_id, base_seed) for independent streams
    function search_rotations(work_info::Tuple{Int, Int, UInt64})
        n_attempts, worker_id, base_seed = work_info

        # Create worker-specific seed using base seed and worker ID
        worker_seed = base_seed + UInt64(worker_id) * 0x9e3779b97f4a7c15
        rng_local = Random.Xoshiro(worker_seed)

        for attempt in 1:n_attempts
            Q = generate_random_orthogonal(n_vars_val, rng_local)
            P_candidate = P_chol * Q

            if check_sign_restrictions(P_candidate, id.restrictions, Φ, id.horizon)
                return (found = true, P = P_candidate, attempt = attempt)
            end
        end

        return (found = false, P = nothing, attempt = n_attempts)
    end

    # Divide work across workers
    n_w = Distributed.nworkers()
    draws_per_worker = fill(div(max_draws, n_w), n_w)
    remainder = rem(max_draws, n_w)
    for i in 1:remainder
        draws_per_worker[i] += 1
    end

    base_seed = rand(rng, UInt64)
    work_specs = [
        (draws_per_worker[i], i, base_seed)
            for i in 1:n_w if draws_per_worker[i] > 0
    ]
    active_workers = length(work_specs)

    if verbose
        println("Searching for valid rotation using $active_workers workers (draw allocations: $(join(draws_per_worker, ", ")))...")
    end

    # Parallel search - returns as soon as first worker finds valid rotation
    results = Distributed.pmap(search_rotations, work_specs)

    # Find first successful result
    for (idx, result) in enumerate(results)
        worker_id = work_specs[idx][2]
        if result.found
            if verbose
                println("Worker $worker_id found valid rotation at attempt $(result.attempt)")
            end
            return result.P
        end
    end

    # If no worker found valid rotation
    total_attempts = sum(draws_per_worker)
    throw(ErrorException("Could not find impact matrix satisfying sign restrictions after $total_attempts attempts across $active_workers workers"))
end

"""
    generate_random_orthogonal(n::Int, rng::AbstractRNG)

Allocating form of [`generate_random_orthogonal!`](@ref): an `n × n` random
orthogonal `Float64` matrix.
"""
function generate_random_orthogonal(n::Int, rng::AbstractRNG)
    return generate_random_orthogonal!(
        Matrix{Float64}(undef, n, n), Matrix{Float64}(undef, n, n), rng
    )
end

generate_random_orthogonal(n::Int) = generate_random_orthogonal(n, Random.default_rng())

"""
    check_sign_restrictions(P, restrictions, model::VARModel, horizon) -> Bool

Check whether impact matrix `P` satisfies sign restrictions, computing the MA
matrices from `model`. Delegates to the matrix-level method
`check_sign_restrictions(P, restrictions, Φ, horizon)`; when checking many
candidates against one model, compute `Φ` once and call that method directly.
"""
function check_sign_restrictions(
        P::AbstractMatrix, restrictions::Matrix{Int},
        model::VARModel, horizon::Int
    )
    Φ = compute_ma_matrices(model.companion, horizon, n_vars(model), n_lags(model))
    return check_sign_restrictions(P, restrictions, Φ, horizon)
end

# ============================================================================
# IV Identification (placeholder - full implementation in ivsvar/)
# ============================================================================

"""
    rotation_matrix(model::VARModel, id::IVIdentification)

SVAR-IV identification via external instruments (Stock & Watson 2018).

The impact column of the target shock is estimated by 2SLS on the model's
residuals. The deprecated `IVIdentification()` (no instrument) takes the
instrument from a model fitted with `fit(IVSVAR, ...)`.
"""
function rotation_matrix(model::VARModel{T}, id::IVIdentification) where {T}
    resolved = _resolve_iv(model, id)
    inst = resolved.instrument

    # Perform 2SLS identification
    ν = residuals(model)
    Z, target = _extract_instrument(inst, size(ν, 1), n_lags(model), model.names)
    β_iv, _, _ = _iv_identify(ν, Z, target)
    return _build_full_impact_matrix(β_iv, target, Matrix(residual_covariance(model)))
end

# ============================================================================
# Narrative Restrictions
# ============================================================================

"""
    rotation_matrix(model::VARModel, id::NarrativeRestriction; kwargs...)

Identification via narrative sign restrictions (Antolín-Díaz & Rubio-Ramírez 2018).

Rejection search: draw a random orthogonal `Q`, form the candidate impact matrix
`P = Σ_L Q` from the residual Cholesky factor, and accept it when it satisfies both
the sign restrictions on the impulse responses and the narrative sign constraints on
the structural shocks `ε = P⁻¹ U` at the restricted dates.

# Keyword Arguments
- `max_draws::Int=10000`: Maximum number of rotation draws before giving up
- `rng::AbstractRNG=Random.default_rng()`: Random number generator

Date-based narrative restrictions resolve against the dates stored on the model
(`dates(model)`, supplied via the `dates` keyword of `fit`/`VAR`). When the model
carries no dates, narrative restrictions are interpreted as integer residual-row
indices; a `Dates.TimeType` restriction then fails fast.

# Returns
- Rotation matrix `P` satisfying the sign and narrative restrictions
"""
function rotation_matrix(
        model::VARModel{T}, id::NarrativeRestriction;
        max_draws::Int = 10000,
        rng::AbstractRNG = Random.default_rng()
    ) where {T}
    n_vars_val = n_vars(model)
    n_lags_val = n_lags(model)

    Σ_L = Matrix(cholesky(residual_covariance(model)).L)
    U = residuals(model)
    Φ = compute_ma_matrices(model.companion, id.horizon, n_vars_val, n_lags_val)
    model_dates = dates(model)

    for _ in 1:max_draws
        Q = generate_random_orthogonal(n_vars_val, rng)
        P = Σ_L * Q

        check_sign_restrictions(P, id.sign_restrictions, Φ, id.horizon) || continue

        ε = compute_structural_shocks(U, P)
        if check_narrative_restrictions(ε, id.narrative_shocks, model_dates)
            return P
        end
    end

    throw(ErrorException("Could not find impact matrix satisfying narrative restrictions after $max_draws attempts"))
end

# ============================================================================
# Normalization functions
# (AbstractNormalization, UnitStd, UnitEffect are defined in types.jl)
# ============================================================================

"""
    normalize_impact!(P::AbstractMatrix, norm::AbstractNormalization)

Rescale the columns of the impact matrix `P` in place according to `norm` and
return `P`.

- `UnitStd()`: no change (each column is the response to a one-standard-deviation
  shock).
- `UnitEffect()`: divide column `j` by `P[j, j]`, so shock `j` moves variable `j`
  by one unit on impact. Columns with `|P[j, j]| ≤ eps` are left unchanged.
"""
normalize_impact!(P::AbstractMatrix, ::UnitStd) = P

function normalize_impact!(P::AbstractMatrix, ::UnitEffect)
    for j in axes(P, 2)
        scale = P[j, j]
        if abs(scale) > eps(real(eltype(P)))
            P[:, j] ./= scale
        end
    end
    return P
end

"""
    normalize_impact(P::AbstractMatrix, norm::AbstractNormalization)

Non-mutating [`normalize_impact!`](@ref).
"""
normalize_impact(P::AbstractMatrix, norm::AbstractNormalization) = normalize_impact!(copy(P), norm)

@deprecate normalize!(P::AbstractMatrix, norm::AbstractNormalization) normalize_impact!(P, norm) false
@deprecate normalize(P::AbstractMatrix, norm::AbstractNormalization) normalize_impact(P, norm) false

# ============================================================================
# MA Representation
# ============================================================================

"""
    compute_ma_matrices(F::AbstractMatrix, horizon::Int, n_vars::Int, n_lags::Int)

Moving-average matrices `Φ_h` (the top-left `n_vars × n_vars` block of `F^h`)
from the companion matrix `F`, for `h = 0, …, horizon`. Only the first
`n_vars` rows of `F` (`[A_1 … A_p]`) are read; the recursion is
[`compute_ma_matrices_from_B!`](@ref).

# Returns
- Array of size (n_vars, n_vars, horizon+1) with `Φ[:, :, 1] = I`
"""
function compute_ma_matrices(
        F::AbstractMatrix{T}, horizon::Int, n_vars::Int, n_lags::Int
    ) where {T}
    Base.require_one_based_indexing(F)
    n_comp = n_vars * n_lags
    size(F) == (n_comp, n_comp) || throw(
        DimensionMismatch(
            "companion matrix must be $(n_comp)×$(n_comp) for n_vars=$n_vars, n_lags=$n_lags; got $(size(F))"
        )
    )
    B = zeros(T, 1 + n_comp, n_vars)
    if n_comp > 0
        B[2:end, :] .= transpose(view(F, 1:n_vars, :))
    end
    Φ = Array{T, 3}(undef, n_vars, n_vars, horizon + 1)
    return compute_ma_matrices_from_B!(
        Φ, B, Matrix{T}(undef, n_vars, n_vars), horizon, n_vars, n_lags
    )
end
