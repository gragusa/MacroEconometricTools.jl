# ============================================================================
# Bootstrap Methods for Inference
# ============================================================================
#
# Three bootstrap methods for VAR impulse responses, each dispatched
# via compute_inference_bands() in var/irfs.jl:
#   - bootstrap_irf_wild: Rademacher wild bootstrap
#   - bootstrap_irf_standard: i.i.d. resampling
#   - bootstrap_irf_block: moving block bootstrap with position-specific centering

# ============================================================================
# Failure reporting helper
# ============================================================================

"""
    _report_bootstrap_failures(method_name, n_failed, reps)

Emit a summary warning when bootstrap replications fail, reporting
effective replications. Called once after the loop (not per-failure).
"""
function _report_bootstrap_failures(method_name::String, n_failed::Int, reps::Int)
    n_failed == 0 && return
    pct = round(100 * n_failed / reps; digits = 1)
    n_eff = reps - n_failed
    @warn "$method_name: $n_failed / $reps replications failed ($pct%). " *
          "Effective bootstrap replications: $n_eff. " *
          "Failed draws are filled with NaN and excluded from quantile computation. " *
          "If the failure rate is high, check model stability or increase sample size."
end

# ============================================================================
# Wild Bootstrap
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
        rng::AbstractRNG;
        normalization::AbstractNormalization = UnitStd()
) where {T}
    m = n_vars(model)
    n_lags_val = n_lags(model)

    # Preallocate output: (reps, horizon+1, n_vars, n_shocks)
    irf_boot = zeros(T, reps, horizon + 1, m, m)

    # Get residuals and original data
    u = residuals(model)
    n_obs = size(u, 1)
    Y_original = model.Y

    # Pre-allocate reusable buffers
    ω = Vector{T}(undef, n_obs)
    ū = Matrix{T}(undef, n_obs, m)

    n_failed = 0
    for r in 1:reps
        # Wild bootstrap: Rademacher weights (±1 with equal probability)
        for i in eachindex(ω)
            ω[i] = rand(rng, Bool) ? one(T) : -one(T)
        end

        # Multiply residuals by weights (broadcasts across columns)
        @inbounds for j in 1:m, i in 1:n_obs

            ū[i, j] = ω[i] * u[i, j]
        end

        # Simulate bootstrap VAR
        Y_boot = simulate_var(model, ū, Y_original)

        # Re-estimate and compute IRF
        try
            var_boot = refit_for_bootstrap(model, Y_boot, n_lags_val)

            # Compute bootstrap IRF
            P_boot = rotation_matrix(var_boot, identification)
            normalize!(P_boot, normalization)
            irf_view = view(irf_boot,r,:,:,:)
            copyto!(irf_view, compute_irf_point(var_boot, P_boot, horizon))
        catch
            n_failed += 1
            fill!(view(irf_boot,r,:,:,:), NaN)
        end
    end

    _report_bootstrap_failures("Wild bootstrap", n_failed, reps)
    return irf_boot
end

# ============================================================================
# Standard Bootstrap
# ============================================================================

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
        rng::AbstractRNG;
        normalization::AbstractNormalization = UnitStd()
) where {T}
    m = n_vars(model)
    n_lags_val = n_lags(model)
    n_obs = size(model.residuals, 1)

    # Preallocate output
    irf_boot = zeros(T, reps, horizon + 1, m, m)

    u = residuals(model)
    Y_original = model.Y

    # Pre-allocate reusable buffer
    ū = Matrix{T}(undef, n_obs, m)

    n_failed = 0
    for r in 1:reps
        # Standard bootstrap: resample rows with replacement
        @inbounds for i in 1:n_obs
            src = rand(rng, 1:n_obs)
            for j in 1:m
                ū[i, j] = u[src, j]
            end
        end

        # Simulate bootstrap VAR
        Y_boot = simulate_var(model, ū, Y_original)

        # Re-estimate and compute IRF
        try
            var_boot = refit_for_bootstrap(model, Y_boot, n_lags_val)

            P_boot = rotation_matrix(var_boot, identification)
            normalize!(P_boot, normalization)
            copyto!(view(irf_boot,r,:,:,:),
                compute_irf_point(var_boot, P_boot, horizon))
        catch
            n_failed += 1
            fill!(view(irf_boot,r,:,:,:), NaN)
        end
    end

    _report_bootstrap_failures("Standard bootstrap", n_failed, reps)
    return irf_boot
end

# ============================================================================
# Block Bootstrap
# ============================================================================

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
        rng::AbstractRNG;
        normalization::AbstractNormalization = UnitStd()
) where {T}
    m = n_vars(model)
    n_lags_val = n_lags(model)
    n_obs = size(model.residuals, 1)  # T
    ℓ = block_length

    # Number of blocks needed to cover T observations
    N = cld(n_obs, ℓ)

    # Preallocate
    irf_boot = zeros(T, reps, horizon + 1, m, m)
    u = residuals(model)
    Y_original = model.Y

    # Preallocate bootstrap residuals (may be longer than T)
    # We create N*ℓ observations, then trim to T
    ū_full = zeros(T, N * ℓ, m)

    # Maximum valid starting index for a block
    max_start = n_obs - ℓ + 1

    # Pre-compute position-specific means for centering
    mean_at_s = Matrix{T}(undef, ℓ, m)
    for s in 1:ℓ
        positions_in_original = s:ℓ:(n_obs - ℓ + s)
        for j in 1:m
            mean_at_s[s, j] = mean(view(u, positions_in_original, j))
        end
    end

    n_failed = 0
    for r in 1:reps
        # Step 1: Sample N random blocks
        for j_blk in 1:N
            start_idx = rand(rng, 1:max_start)
            block_start = 1 + ℓ * (j_blk - 1)
            @inbounds for s in 1:ℓ, j in 1:m

                ū_full[block_start + s - 1, j] = u[start_idx + s - 1, j]
            end
        end

        # Step 2: Position-specific centering
        @inbounds for s in 1:ℓ
            for j_blk in 0:(N - 1)
                row = j_blk * ℓ + s
                for j in 1:m
                    ū_full[row, j] -= mean_at_s[s, j]
                end
            end
        end

        # Step 3: Trim to original sample size
        ū = view(ū_full, 1:n_obs, :)

        # Step 4: Simulate bootstrap VAR
        Y_boot = simulate_var(model, ū, Y_original)

        # Step 5: Re-estimate and compute IRF
        try
            var_boot = refit_for_bootstrap(model, Y_boot, n_lags_val)

            P_boot = rotation_matrix(var_boot, identification)
            normalize!(P_boot, normalization)
            copyto!(view(irf_boot,r,:,:,:),
                compute_irf_point(var_boot, P_boot, horizon))
        catch
            n_failed += 1
            fill!(view(irf_boot,r,:,:,:), NaN)
        end
    end

    _report_bootstrap_failures("Block bootstrap", n_failed, reps)
    return irf_boot
end

# ============================================================================
# Backward-compatible wrapper
# ============================================================================

"""
    bootstrap_irf(model, identification, horizon, reps; method=:wild, block_length=10, rng=...)

Backward-compatible wrapper that dispatches to `bootstrap_irf_wild`,
`bootstrap_irf_standard`, or `bootstrap_irf_block` based on `method`.

The `parallel` keyword is deprecated; use Julia's built-in threading
or Distributed.pmap externally if parallel bootstrap is needed.
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

    if parallel != :none
        @warn "parallel keyword is deprecated in bootstrap_irf. " *
              "Falling back to serial execution."
    end

    if method == :wild
        return bootstrap_irf_wild(model, identification, horizon, reps, rng;
            normalization)
    elseif method == :standard
        return bootstrap_irf_standard(model, identification, horizon, reps, rng;
            normalization)
    else  # :block
        return bootstrap_irf_block(model, identification, horizon, reps, block_length, rng;
            normalization)
    end
end
