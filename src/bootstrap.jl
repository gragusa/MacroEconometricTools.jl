# ============================================================================
# Bootstrap Methods for Inference
# ============================================================================

"""
    bootstrap_irf(model::VARModel, identification, horizon, reps; kwargs...)

Bootstrap impulse response functions for inference.

# Arguments
- `model::VARModel`: Estimated VAR model
- `identification::AbstractIdentification`: Identification scheme
- `horizon::Int`: IRF horizon
- `reps::Int`: Number of bootstrap replications

# Keyword Arguments
- `method::Symbol=:wild`: Bootstrap method (`:wild`, `:standard`, `:block`)
- `block_length::Int=10`: Block length for block bootstrap
- `normalization::AbstractNormalization=UnitStd()`: Shock normalization
- `parallel::Symbol=:none`: Parallelization (`:none`, `:distributed`)
- `rng::AbstractRNG=Random.default_rng()`: Random number generator for reproducible draws

# Returns
- Array of size (reps, horizon+1, n_vars, n_shocks) with bootstrap IRFs

# Examples
```julia
# Serial execution
irf_boot = bootstrap_irf(model, id, 24, 1000)

# Distributed execution (requires Distributed package and workers)
using Distributed
addprocs(4)
@everywhere using MacroEconometricTools
irf_boot = bootstrap_irf(model, id, 24, 1000; parallel=:distributed)
```
"""
function bootstrap_irf(model::VARModel{T}, identification::AbstractIdentification,
        horizon::Int, reps::Int;
        method::Symbol = :wild,
        block_length::Int = 10,
        normalization::AbstractNormalization = UnitStd(),
        parallel::Symbol = :none,
        rng::AbstractRNG = Random.default_rng()) where {T}
    method ∈ [:wild, :standard, :block] ||
        throw(ArgumentError("method must be :wild, :standard, or :block"))
    parallel ∈ [:none, :distributed] ||
        throw(ArgumentError("parallel must be :none or :distributed"))

    n_vars_val = n_vars(model)

    # Choose execution method
    if parallel == :distributed
        irf_boot = bootstrap_irf_distributed(model, identification, horizon, reps,
            method, block_length, normalization, rng)
    else
        # Preallocate bootstrap IRF storage
        irf_boot = zeros(T, reps, horizon + 1, n_vars_val, n_vars_val)
        bootstrap_irf_serial!(irf_boot, model, identification, horizon,
            method, block_length, normalization, rng)
    end

    return irf_boot
end

"""
    bootstrap_irf_serial!(irf_boot, model, identification, horizon, method, block_length, normalization, rng)

Serial bootstrap computation.
"""
function bootstrap_irf_serial!(irf_boot::Array{T, 4}, model::VARModel{T},
        identification::AbstractIdentification, horizon::Int,
        method::Symbol, block_length::Int,
        normalization::AbstractNormalization,
        rng::AbstractRNG) where {T}
    reps = size(irf_boot, 1)
    residuals = model.residuals
    n_obs_val = size(residuals, 1)
    n_vars_val = n_vars(model)

    # Original data for initialization
    Y_original = model.Y

    for r in 1:reps
        # Generate bootstrap residuals
        if method == :wild
            u_boot = wild_bootstrap_residuals(rng, residuals)
        elseif method == :standard
            u_boot = standard_bootstrap_residuals(rng, residuals)
        elseif method == :block
            u_boot = block_bootstrap_residuals(rng, residuals, block_length)
        end

        # Simulate bootstrap data
        Y_boot = simulate_var(model, u_boot, Y_original)

        # Re-estimate VAR on bootstrap data
        try
            constraints_arg = model.coefficients.constraints
            constraints_arg = constraints_arg === nothing ? AbstractConstraint[] :
                              constraints_arg
            var_boot = fit(typeof(model.spec), Y_boot, n_lags(model);
                constraints = constraints_arg,
                names = model.names)

            # Compute IRF for bootstrap sample
            P_boot = rotation_matrix(var_boot, identification)
            P_boot = normalize(P_boot, normalization)
            irf_boot[r, :, :, :] = compute_irf_point(var_boot, P_boot, horizon)
        catch e
            # If estimation fails, use previous valid draw (or zeros)
            if r > 1
                irf_boot[r, :, :, :] = irf_boot[r - 1, :, :, :]
            end
            @warn "Bootstrap iteration $r failed: $e"
        end
    end
end

# ============================================================================
# Bootstrap Residual Generation
# ============================================================================

"""
    wild_bootstrap_residuals(rng::AbstractRNG, residuals::Matrix)

Wild bootstrap: multiply residuals by random ±1.
"""
function wild_bootstrap_residuals(rng::AbstractRNG, residuals::Matrix{T}) where {T}
    n_obs, n_vars = size(residuals)
    u_boot = similar(residuals)

    # Rademacher weights: ±1 with equal probability
    weights = rand(rng, [-one(T), one(T)], n_obs)

    for j in 1:n_vars
        u_boot[:, j] = weights .* residuals[:, j]
    end

    return u_boot
end

"""
    standard_bootstrap_residuals(rng::AbstractRNG, residuals::Matrix)

Standard bootstrap: resample residuals with replacement.
"""
function standard_bootstrap_residuals(rng::AbstractRNG, residuals::Matrix{T}) where {T}
    n_obs = size(residuals, 1)
    indices = rand(rng, 1:n_obs, n_obs)
    return residuals[indices, :]
end

"""
    block_bootstrap_residuals(rng::AbstractRNG, residuals::Matrix, block_length::Int)

Moving block bootstrap for time series.
"""
function block_bootstrap_residuals(rng::AbstractRNG, residuals::Matrix{T}, block_length::Int) where {T}
    n_obs, n_vars = size(residuals)
    u_boot = similar(residuals)

    n_blocks = ceil(Int, n_obs / block_length)
    max_start = n_obs - block_length + 1

    pos = 1
    for block in 1:n_blocks
        # Random starting point
        start_idx = rand(rng, 1:max_start)
        end_idx = min(start_idx + block_length - 1, n_obs)
        block_size = end_idx - start_idx + 1

        # Copy block
        copy_end = min(pos + block_size - 1, n_obs)
        u_boot[pos:copy_end, :] = residuals[start_idx:(start_idx + copy_end - pos), :]

        pos += block_size
        if pos > n_obs
            break
        end
    end

    # Demean blocks to preserve zero mean (Carlstein, 1986)
    for j in 1:n_vars
        u_boot[:, j] .-= mean(u_boot[:, j])
    end

    return u_boot
end

# ============================================================================
# Confidence Band Computation
# ============================================================================

"""
    compute_bands_bootstrap(irf_point, irf_boot, coverage)

Compute percentile bootstrap confidence bands.
"""
function compute_bands_bootstrap(irf_point::Array{T, 3}, irf_boot::Array{T, 4},
        coverage::Vector{Float64}) where {T}
    reps = size(irf_boot, 1)

    lower = Vector{Array{T, 3}}(undef, length(coverage))
    upper = Vector{Array{T, 3}}(undef, length(coverage))

    # Compute centered bootstrap draws: irf_boot - mean(irf_boot) + irf_point
    irf_boot_mean = mean(irf_boot; dims = 1)[1, :, :, :]
    irf_boot_centered = irf_boot .- reshape(irf_boot_mean, (1, size(irf_boot_mean)...)) .+
                        reshape(irf_point, (1, size(irf_point)...))

    for (i, α) in enumerate(coverage)
        # Percentile method
        α_lower = (1 - α) / 2
        α_upper = 1 - α_lower

        lower[i] = mapslices(x -> quantile(x, α_lower), irf_boot_centered; dims = 1)[1, :, :, :]
        upper[i] = mapslices(x -> quantile(x, α_upper), irf_boot_centered; dims = 1)[1, :, :, :]
    end

    return lower, upper
end

"""
    compute_bands_percentile(irf_boot, coverage)

Compute simple percentile confidence bands (non-centered).
"""
function compute_bands_percentile(irf_boot::Array{T, 4}, coverage::Vector{Float64}) where {T}
    lower = Vector{Array{T, 3}}(undef, length(coverage))
    upper = Vector{Array{T, 3}}(undef, length(coverage))

    for (i, α) in enumerate(coverage)
        α_lower = (1 - α) / 2
        α_upper = 1 - α_lower

        lower[i] = mapslices(x -> quantile(x, α_lower), irf_boot; dims = 1)[1, :, :, :]
        upper[i] = mapslices(x -> quantile(x, α_upper), irf_boot; dims = 1)[1, :, :, :]
    end

    return lower, upper
end

# ============================================================================
# Bootstrap Diagnostics
# ============================================================================

"""
    bootstrap_diagnostics(irf_boot::Array)

Compute bootstrap diagnostics (bias, variance, etc.).

# Returns
- NamedTuple with diagnostic statistics
"""
function bootstrap_diagnostics(irf_boot::Array{T, 4}) where {T}
    boot_mean = mean(irf_boot; dims = 1)[1, :, :, :]
    boot_std = std(irf_boot; dims = 1)[1, :, :, :]
    boot_median = mapslices(median, irf_boot; dims = 1)[1, :, :, :]

    return (
        mean = boot_mean,
        std = boot_std,
        median = boot_median,
        n_reps = size(irf_boot, 1)
    )
end

# ============================================================================
# Distributed Bootstrap
# ============================================================================

"""
    bootstrap_irf_distributed(model, identification, horizon, reps, method, block_length, normalization, rng)

Distributed bootstrap using multiple processes with batching for efficiency.

Batches replications across workers to minimize communication overhead.
Each worker processes batch_size replications using independent RNG streams.

Requires Distributed package and worker processes to be set up:
```julia
using Distributed
addprocs(4)
@everywhere using MacroEconometricTools
```
"""
function bootstrap_irf_distributed(
        model::VARModel{T}, identification::AbstractIdentification,
        horizon::Int, reps::Int,
        method::Symbol, block_length::Int,
        normalization::AbstractNormalization,
        rng::AbstractRNG) where {T}

    # Check if Distributed is available
    if !isdefined(Main, :Distributed)
        @warn "Distributed package not loaded. Falling back to serial execution."
        irf_boot = zeros(T, reps, horizon + 1, n_vars(model), n_vars(model))
        bootstrap_irf_serial!(irf_boot, model, identification, horizon,
            method, block_length, normalization, rng)
        return irf_boot
    end

    dist = Base.require(Main, :Distributed)

    if dist.nworkers() == 1
        @warn "No worker processes available. Use addprocs() to add workers. Falling back to serial."
        irf_boot = zeros(T, reps, horizon + 1, n_vars(model), n_vars(model))
        bootstrap_irf_serial!(irf_boot, model, identification, horizon,
            method, block_length, normalization, rng)
        return irf_boot
    end

    # Batch bootstrap iterations to minimize pmap overhead
    # Each worker processes a batch of replications
    function batched_bootstrap_rep(batch_info::Tuple{UnitRange{Int}, UInt64})
        batch_range, base_seed = batch_info
        n_batch = length(batch_range)
        n_vars_val = n_vars(model)
        n_lags_val = n_lags(model)

        # Preallocate output for this batch
        batch_irfs = zeros(T, n_batch, horizon + 1, n_vars_val, n_vars_val)

        # Extract model components (avoid repeated access)
        residuals = model.residuals
        Y_original = model.Y[1:n_lags_val, :]
        constraints_arg = model.coefficients.constraints
        constraints_arg = constraints_arg === nothing ? AbstractConstraint[] :
                          constraints_arg

        # Process each replication in the batch
        for (batch_idx, rep_idx) in enumerate(batch_range)
            # Create replication-specific seed
            # This ensures deterministic, independent streams
            worker_seed = base_seed + UInt64(rep_idx) * 0x9e3779b97f4a7c15
            rng_local = Random.Xoshiro(worker_seed)

            # Resample residuals
            u_boot = if method == :wild
                wild_bootstrap_residuals(rng_local, residuals)
            elseif method == :standard
                standard_bootstrap_residuals(rng_local, residuals)
            else  # :block
                block_bootstrap_residuals(rng_local, residuals, block_length)
            end

            # Simulate VAR
            Y_boot = simulate_var(model, u_boot, Y_original)

            # Re-estimate
            try
                var_boot = fit(typeof(model.spec), Y_boot, n_lags_val;
                    constraints = constraints_arg,
                    names = model.names)

                # Re-identify and compute IRF
                P_boot = rotation_matrix(var_boot, identification)
                P_boot = normalize(P_boot, normalization)
                batch_irfs[batch_idx, :, :, :] = compute_irf_point(var_boot, P_boot, horizon)
            catch e
                # If estimation fails, use zeros (or could use previous valid draw)
                @warn "Bootstrap replication $rep_idx failed: $e"
            end
        end

        return batch_irfs
    end

    # Create batches: distribute replications across workers
    n_workers = dist.nworkers()
    # Aim for ~10-20 batches per worker for good load balancing
    n_batches = min(reps, max(n_workers * 10, 20))
    batch_size = ceil(Int, reps / n_batches)

    # Generate base seed for reproducibility
    base_seed = rand(rng, UInt64)

    # Create batch ranges
    batches = Tuple{UnitRange{Int}, UInt64}[]
    for start_idx in 1:batch_size:reps
        end_idx = min(start_idx + batch_size - 1, reps)
        push!(batches, (start_idx:end_idx, base_seed))
    end

    # Distributed map over batches
    println("Running $reps bootstrap replications in $(length(batches)) batches on $n_workers workers...")
    batch_results = dist.pmap(batched_bootstrap_rep, batches)

    # Concatenate batch results
    n_vars_val = n_vars(model)
    irf_boot = zeros(T, reps, horizon + 1, n_vars_val, n_vars_val)

    current_idx = 1
    for batch_result in batch_results
        n_in_batch = size(batch_result, 1)
        irf_boot[current_idx:(current_idx + n_in_batch - 1), :, :, :] = batch_result
        current_idx += n_in_batch
    end

    return irf_boot
end

# ============================================================================
# Bootstrap Implementations for InferenceType System
# ============================================================================

"""
    bootstrap_irf_wild(model, identification, horizon, reps, rng)

Wild bootstrap for VAR impulse responses.

The wild bootstrap resamples residuals by multiplying them with random
Rademacher weights (±1 with equal probability). This preserves the
conditional heteroskedasticity structure in the residuals while maintaining
independence across equations.

# Algorithm
1. For each bootstrap replication:
   - Draw Rademacher weights: ω[t] ∼ Uniform({-1, +1})
   - Create bootstrap residuals: ū[t] = ω[t] * u[t]
   - Simulate VAR with bootstrap residuals starting from initial observations
   - Re-estimate VAR and compute IRF
2. Return all bootstrap IRF draws

# References
- Liu (1988): "Bootstrap Procedures under Some Non-I.I.D. Models"
- Gonçalves and Kilian (2004): "Bootstrapping autoregressions with
  conditional heteroskedasticity of unknown form"

# Note
The wild bootstrap is robust to conditional heteroskedasticity and preserves
the cross-equation correlation structure in the residuals.
"""
function bootstrap_irf_wild(
        model::VARModel{T},
        identification::AbstractIdentification,
        horizon::Int,
        reps::Int,
        rng::AbstractRNG
) where {T}
    m = n_vars(model)
    n_lags_val = n_lags(model)

    # Preallocate output: (reps, horizon+1, n_vars, n_shocks)
    irf_boot = zeros(T, reps, horizon + 1, m, m)

    # Get residuals and original data
    u = residuals(model)
    Y_original = model.Y

    # Get normalization from identification (if applicable)
    # We'll compute this once outside the loop
    for r in 1:reps
        # Wild bootstrap: Rademacher weights (±1 with equal probability)
        ω = rand(rng, T[-one(T), one(T)], size(u, 1))

        # Multiply residuals by weights (broadcasts across columns)
        ū = ω .* u

        # Simulate bootstrap VAR
        Y_boot = simulate_var(model, ū, Y_original)

        # Re-estimate VAR on bootstrap data
        try
            constraints_arg = model.coefficients.constraints
            constraints_arg = isnothing(constraints_arg) ? AbstractConstraint[] :
                              constraints_arg
            var_boot = fit(typeof(model.spec), Y_boot, n_lags_val;
                constraints = constraints_arg,
                names = model.names)

            # Compute bootstrap IRF
            P_boot = rotation_matrix(var_boot, identification)
            P_boot = normalize(P_boot, UnitStd())
            irf_boot[r, :, :, :] = compute_irf_point(var_boot, P_boot, horizon)
        catch e
            # If estimation fails, fill with NaN
            @warn "Wild bootstrap replication $r failed: $e"
            fill!(view(irf_boot,r,:,:,:), NaN)
        end
    end

    return irf_boot
end

"""
    bootstrap_irf_standard(model, identification, horizon, reps, rng)

Standard i.i.d. bootstrap for VAR impulse responses.

The standard bootstrap resamples residuals with replacement, treating them
as independent draws from an unknown distribution. This is appropriate when
residuals can be assumed i.i.d. (homoskedastic and uncorrelated).

# Algorithm
1. For each bootstrap replication:
   - Sample row indices with replacement: i[t] ∼ Uniform(1, ..., T)
   - Create bootstrap residuals: ū[t] = u[i[t], :]
   - Simulate VAR with bootstrap residuals
   - Re-estimate VAR and compute IRF
2. Return all bootstrap IRF draws

# References
- Efron (1979): "Bootstrap methods: Another look at the jackknife"
- Freedman (1981): "Bootstrapping regression models"

# Note
For time series with temporal dependence or heteroskedasticity, wild bootstrap or
block bootstrap may be more appropriate. The standard bootstrap assumes i.i.d. errors.
"""
function bootstrap_irf_standard(
        model::VARModel{T},
        identification::AbstractIdentification,
        horizon::Int,
        reps::Int,
        rng::AbstractRNG
) where {T}
    m = n_vars(model)
    n_lags_val = n_lags(model)
    n_obs = size(model.residuals, 1)

    # Preallocate output
    irf_boot = zeros(T, reps, horizon + 1, m, m)

    u = residuals(model)
    Y_original = model.Y

    for r in 1:reps
        # Standard bootstrap: resample rows with replacement
        indices = rand(rng, 1:n_obs, n_obs)
        ū = u[indices, :]

        # Simulate bootstrap VAR
        Y_boot = simulate_var(model, ū, Y_original)

        # Re-estimate and compute IRF
        try
            constraints_arg = model.coefficients.constraints
            constraints_arg = isnothing(constraints_arg) ? AbstractConstraint[] :
                              constraints_arg
            var_boot = fit(typeof(model.spec), Y_boot, n_lags_val;
                constraints = constraints_arg,
                names = model.names)

            P_boot = rotation_matrix(var_boot, identification)
            P_boot = normalize(P_boot, UnitStd())
            irf_boot[r, :, :, :] = compute_irf_point(var_boot, P_boot, horizon)
        catch e
            @warn "Standard bootstrap replication $r failed: $e"
            fill!(view(irf_boot,r,:,:,:), NaN)
        end
    end

    return irf_boot
end

"""
    bootstrap_irf_block(model, identification, horizon, reps, block_length, rng)

Moving block bootstrap for VAR impulse responses.

The block bootstrap resamples blocks of consecutive residuals to preserve
the temporal dependence structure in the data. This is appropriate for time
series with serial correlation that violates the i.i.d. assumption.

# Algorithm
1. Divide the sample into overlapping blocks of length ℓ
2. For each bootstrap replication:
   - Randomly select N = ⌈T/ℓ⌉ blocks (with replacement)
   - Concatenate blocks to form bootstrap residual series of length N*ℓ
   - Apply position-specific centering: for position s within a block,
     subtract the mean of all residuals at position s, s+ℓ, s+2ℓ, ...
   - Trim bootstrap series to original sample size T
   - Simulate VAR with bootstrap residuals
   - Re-estimate VAR and compute IRF
3. Return all bootstrap IRF draws

# Position-Specific Centering
The key innovation (following Carlstein 1986, Künsch 1989) is centering
residuals based on their position within the block cycle:

    ū[j * ℓ + s] -= mean(u[s : ℓ : end])

This preserves the block structure while ensuring the resampled residuals
have (approximately) zero mean, which is critical for VAR simulation. Without
this adjustment, the bootstrap VAR would drift away from the true mean.

# Why Position-Specific?
Simply subtracting the global mean would break the temporal dependence structure
within blocks. The position-specific approach respects that observations at
position s may systematically differ from those at position s+1, preserving
intra-block patterns.

# References
- Künsch (1989): "The jackknife and the bootstrap for general stationary
  observations"
- Carlstein (1986): "The use of subseries values for estimating the variance
  of a general statistic from a stationary sequence"
- Paparoditis and Politis (2001): "Tapered block bootstrap"

# Block Length Selection
Rule of thumb: ℓ ≈ T^(1/3) for moderate dependence
For stronger persistence, use larger blocks (e.g., ℓ = 10-20 for quarterly data)

# Example
```julia
# For T=100 observations with moderate dependence
irf_boot = bootstrap_irf_block(model, id, 24, 1000, 10, rng)

# For stronger persistence
irf_boot = bootstrap_irf_block(model, id, 24, 1000, 20, rng)
```
"""
function bootstrap_irf_block(
        model::VARModel{T},
        identification::AbstractIdentification,
        horizon::Int,
        reps::Int,
        block_length::Int,
        rng::AbstractRNG
) where {T}
    m = n_vars(model)
    n_lags_val = n_lags(model)
    n_obs = size(model.residuals, 1)  # T
    ℓ = block_length

    # Number of blocks needed to cover T observations
    N = ceil(Int, n_obs / ℓ)

    # Preallocate
    irf_boot = zeros(T, reps, horizon + 1, m, m)
    u = residuals(model)
    Y_original = model.Y

    # Preallocate bootstrap residuals (may be longer than T)
    # We create N*ℓ observations, then trim to T
    ū_full = zeros(T, N * ℓ, m)

    # Maximum valid starting index for a block
    max_start = n_obs - ℓ + 1

    for r in 1:reps
        # Step 1: Sample N random blocks
        # Each block has length ℓ and starts at a random position
        for j in 1:N
            # Random starting index for this block
            start_idx = rand(rng, 1:max_start)

            # Copy block into position j of bootstrap sample
            block_start = 1 + ℓ * (j - 1)
            block_end = block_start + ℓ - 1
            ū_full[block_start:block_end, :] = u[start_idx:(start_idx + ℓ - 1), :]
        end

        # Step 2: Position-specific centering
        # For each position s within a block (s = 1, ..., ℓ):
        #   Subtract mean of all residuals at positions s, s+ℓ, s+2ℓ, ..., s+(T-ℓ)
        # This preserves the block structure while ensuring zero mean
        for s in 1:ℓ
            # Indices at this position in the ORIGINAL residuals
            # These are positions s, s+ℓ, s+2ℓ, ..., s+k*ℓ where s+k*ℓ ≤ T
            positions_in_original = s:ℓ:(n_obs - ℓ + s)

            # Compute mean at this position from original residuals
            mean_at_s = vec(mean(u[positions_in_original, :]; dims = 1))

            # Subtract from all occurrences of this position in bootstrap sample
            for j in 0:(N - 1)
                ū_full[j * ℓ + s, :] .-= mean_at_s
            end
        end

        # Step 3: Trim to original sample size
        ū = ū_full[1:n_obs, :]

        # Step 4: Simulate bootstrap VAR
        Y_boot = simulate_var(model, ū, Y_original)

        # Step 5: Re-estimate and compute IRF
        try
            constraints_arg = model.coefficients.constraints
            constraints_arg = isnothing(constraints_arg) ? AbstractConstraint[] :
                              constraints_arg
            var_boot = fit(typeof(model.spec), Y_boot, n_lags_val;
                constraints = constraints_arg,
                names = model.names)

            P_boot = rotation_matrix(var_boot, identification)
            P_boot = normalize(P_boot, UnitStd())
            irf_boot[r, :, :, :] = compute_irf_point(var_boot, P_boot, horizon)
        catch e
            @warn "Block bootstrap replication $r failed: $e"
            fill!(view(irf_boot,r,:,:,:), NaN)
        end
    end

    return irf_boot
end
