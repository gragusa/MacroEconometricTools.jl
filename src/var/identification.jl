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
- Lower-triangular rotation matrix
"""
function rotation_matrix(model::VARModel{T}, id::CholeskyID) where {T}
    Σ = vcov(model)

    # Handle variable ordering
    if id.ordering !== nothing
        perm = get_permutation(model.names, id.ordering)
        Σ_ordered = Σ[perm, perm]
        P_ordered = Matrix(cholesky(Σ_ordered).L)

        # Permute back to original ordering
        inv_perm = invperm(perm)
        P = P_ordered[inv_perm, inv_perm]
    else
        # Use data ordering
        P = Matrix(cholesky(Σ).L)
    end

    return P
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
function rotation_matrix(model::VARModel{T}, id::SignRestriction;
        max_draws::Int = 10000,
        parallel::Symbol = :none,
        verbose::Bool = false,
        rng::AbstractRNG = Random.default_rng()) where {T}
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
function identify_sign_serial(model::VARModel{T}, id::SignRestriction,
        max_draws::Int, verbose::Bool,
        rng::AbstractRNG) where {T}
    n_vars_val = n_vars(model)
    Σ = vcov(model)

    # Cholesky as starting point
    P_chol = Matrix(cholesky(Σ).L)

    # Search for valid rotation
    for iter in 1:max_draws
        # Random orthogonal matrix via QR decomposition
        Q = generate_random_orthogonal(n_vars_val, rng)

        # Candidate impact matrix
        P_candidate = P_chol * Q

        # Check sign restrictions on impact or IRFs
        if check_sign_restrictions(P_candidate, id.restrictions, model, id.horizon)
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
function identify_sign_distributed(model::VARModel{T}, id::SignRestriction,
        max_draws::Int, verbose::Bool,
        rng::AbstractRNG) where {T}

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
    Σ = vcov(model)
    P_chol = Matrix(cholesky(Σ).L)

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

            if check_sign_restrictions(P_candidate, id.restrictions, model, id.horizon)
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
    work_specs = [(draws_per_worker[i], i, base_seed)
                  for i in 1:n_w if draws_per_worker[i] > 0]
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

Generate random orthogonal matrix via QR decomposition of random normal matrix.
"""
function generate_random_orthogonal(n::Int, rng::AbstractRNG)
    X = randn(rng, n, n)
    Q, R = qr(X)
    # Normalize to ensure det(Q) = 1
    D = Diagonal(sign.(diag(R)))
    return Matrix(Q * D)
end

generate_random_orthogonal(n::Int) = generate_random_orthogonal(n, Random.default_rng())

"""
    check_sign_restrictions(P, restrictions, model, horizon)

Check if impact matrix satisfies sign restrictions.

# Arguments
- `P::Matrix`: Candidate impact matrix
- `restrictions::Matrix{Int}`: Sign restriction matrix (+1, -1, 0 for no restriction)
- `model::VARModel`: VAR model
- `horizon::Int`: Horizon over which restrictions apply

# Returns
- `true` if restrictions are satisfied
"""
function check_sign_restrictions(P::Matrix{T}, restrictions::Matrix{Int},
        model::VARModel, horizon::Int) where {T}
    n_vars_val = size(P, 1)

    # Check impact (horizon 0)
    for i in 1:n_vars_val
        for j in 1:n_vars_val
            if restrictions[i, j] == 1 && P[i, j] < 0
                return false
            elseif restrictions[i, j] == -1 && P[i, j] > 0
                return false
            end
        end
    end

    # Check restrictions on IRFs up to horizon if horizon > 0
    if horizon > 0
        # Compute IRFs
        F = model.companion
        n_lags_val = n_lags(model)
        Φ = compute_ma_matrices(F, horizon, n_vars_val, n_lags_val)

        for h in 1:horizon
            IRF_h = Φ[:, :, h + 1] * P
            for i in 1:n_vars_val
                for j in 1:n_vars_val
                    if restrictions[i, j] == 1 && IRF_h[i, j] < 0
                        return false
                    elseif restrictions[i, j] == -1 && IRF_h[i, j] > 0
                        return false
                    end
                end
            end
        end
    end

    return true
end

# ============================================================================
# IV Identification (placeholder - full implementation in ivsvar/)
# ============================================================================

"""
    rotation_matrix(model::VARModel, id::IVIdentification)

SVAR-IV identification via external instruments (Stock & Watson 2018).

When `id` carries an instrument, 2SLS identification is performed on the model's
residuals. When `id` has no instrument (backward compat `IVIdentification()`),
the structural impact matrix is extracted from an IVSVAR model's metadata.
"""
function rotation_matrix(model::VARModel{T}, id::IVIdentification) where {T}
    resolved = _resolve_iv(model, id)
    inst = resolved.instrument

    # Perform 2SLS identification
    ν = residuals(model)
    Z, target = _extract_instrument(inst, size(ν, 1), n_lags(model), model.names)
    β_iv, _, _ = _iv_identify(ν, Z, target)
    return _build_full_impact_matrix(β_iv, target, Matrix(vcov(model)))
end

# ============================================================================
# Normalization functions
# (AbstractNormalization, UnitStd, UnitEffect are defined in types.jl)
# ============================================================================

"""
    normalize!(P::Matrix, norm::AbstractNormalization)

Normalize impact matrix according to normalization scheme.

# Arguments
- `P::Matrix`: Impact matrix
- `norm::AbstractNormalization`: Normalization scheme

# Returns
- Normalized impact matrix (modifies in place)
"""
function normalize!(P::Matrix{T}, ::UnitStd) where {T}
    # Already normalized with Cholesky: P*P' = Σ implies structural shocks have unit variance
    return P
end

function normalize!(P::Matrix{T}, ::UnitEffect) where {T}
    # Normalize so diagonal elements are 1 (unit effect on impact)
    n = size(P, 1)
    for j in 1:n
        scale = P[j, j]
        if abs(scale) > eps(T)
            P[:, j] ./= scale
        end
    end
    return P
end

"""
    normalize(P::Matrix, norm::AbstractNormalization)

Non-mutating version of normalize!
"""
normalize(P::Matrix, norm::AbstractNormalization) = normalize!(copy(P), norm)

# ============================================================================
# MA Representation
# ============================================================================

"""
    compute_ma_matrices(F::Matrix, horizon::Int, n_vars::Int, n_lags::Int)

Compute moving average (MA) representation matrices Φ_h = F^h.

# Returns
- Array of size (n_vars, n_vars, horizon+1) with MA coefficients
"""
function compute_ma_matrices(F::Matrix{T}, horizon::Int, n_vars::Int, n_lags::Int) where {T}
    n_comp = n_vars * n_lags

    # Preallocate
    Φ = zeros(T, n_vars, n_vars, horizon + 1)

    # Φ_0 = I
    for i in 1:n_vars
        Φ[i, i, 1] = one(T)
    end

    # Since J selects the first n_vars rows and J' selects the first n_vars columns,
    # Φ_h = F_power[1:n_vars, 1:n_vars]. No need to form J explicitly.
    F_power = copy(F)
    F_power_buf = similar(F)

    for h in 1:horizon
        # Extract Φ_h = F^h[1:n_vars, 1:n_vars]
        @inbounds for j in 1:n_vars, i in 1:n_vars

            Φ[i, j, h + 1] = F_power[i, j]
        end
        # F_power <- F_power * F  (in-place via buffer swap)
        mul!(F_power_buf, F_power, F)
        F_power, F_power_buf = F_power_buf, F_power
    end

    return Φ
end
