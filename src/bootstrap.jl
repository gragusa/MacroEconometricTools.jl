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
# Fast refit / IRF workspace (CholeskyID specialisation)
# ============================================================================
#
# The generic bootstrap inner loop calls `refit_for_bootstrap` → `fit(OLSVAR)`
# (allocating Y_work, lagged X, Y_est, X_est, residuals, Σ, coef reshape,
# companion form, and a full VARModel struct per rep), then
# `rotation_matrix` (Cholesky of vcov), then `compute_irf_point`
# (allocating Φ and irf_array) per rep. For Cholesky identification — the
# common case — all of that can be replaced by a workspace-backed in-place
# OLS via the normal equations + a fused IRF computation that reuses two
# companion-power buffers.
#
# Numerical note: Cholesky on (X'X) instead of QR on X is a few ulps less
# accurate but much faster for small K. For well-conditioned bootstrap
# samples (the universal case at bootstrap sample sizes) the differences
# are numerical noise. If factorisation fails (PosDefException), the caller
# catches the exception and fills the rep with NaN, matching the current
# failure-handling behaviour.

"""
    _BootstrapWorkspace{T}

Preallocated buffers for the Cholesky-identification fast path used by
`bootstrap_irf_wild`, `bootstrap_irf_standard`, and `bootstrap_irf_block`.
One instance is created at the top of each bootstrap function and reused
across every replication. This avoids the per-rep allocation of a full
VARModel, lagged-X matrix, residuals, Φ array, and IRF array that the
generic `fit(OLSVAR)` path incurs.

Fields:
- `X` :: `(T_eff × d)` regressor matrix, d = 1 + K·p. Rebuilt per rep from
  `Y_boot`.
- `G = X'X` :: `(d × d)` normal-equations matrix.
- `XtY = X'Y` :: `(d × K)` right-hand side.
- `A` :: `(d × K)` OLS coefficients (first row = intercept, remaining rows
  = vec'd lag coefficients by variable, stacked per-lag).
- `U` :: `(T_eff × K)` residuals Y − X·A.
- `Σ` :: `(K × K)` residual covariance.
- `Lchol` :: `(K × K)` lower-triangular Cholesky factor of `Σ`.
- `F` :: `(K·p × K·p)` companion matrix.
- `F_power`, `F_power_buf` :: `(K·p × K·p)` ping-pong buffers for companion
  power iteration `F_power ← F_power · F`.
"""
struct _BootstrapWorkspace{T}
    X::Matrix{T}
    G::Matrix{T}
    XtY::Matrix{T}
    A::Matrix{T}
    U::Matrix{T}
    Σ::Matrix{T}
    Lchol::Matrix{T}
    F::Matrix{T}
    F_power::Matrix{T}
    F_power_buf::Matrix{T}
end

function _BootstrapWorkspace{T}(n_obs::Int, n_vars_val::Int, n_lags_val::Int) where {T}
    d = 1 + n_vars_val * n_lags_val
    n_eff = n_obs - n_lags_val
    Kp = n_vars_val * n_lags_val
    return _BootstrapWorkspace{T}(
        Matrix{T}(undef, n_eff, d),       # X
        Matrix{T}(undef, d, d),           # G
        Matrix{T}(undef, d, n_vars_val),  # XtY
        Matrix{T}(undef, d, n_vars_val),  # A
        Matrix{T}(undef, n_eff, n_vars_val),  # U
        Matrix{T}(undef, n_vars_val, n_vars_val),  # Σ
        Matrix{T}(undef, n_vars_val, n_vars_val),  # Lchol
        zeros(T, Kp, Kp),                 # F (must start zero; only top blocks rewritten)
        Matrix{T}(undef, Kp, Kp),         # F_power
        Matrix{T}(undef, Kp, Kp)         # F_power_buf
    )
end

# Build X = [1  Y_{t-1}  …  Y_{t-p}] directly into ws.X, with rows t = p+1..T.
# This is equivalent to the column order `create_lags(Y, p)` produces after
# dropping the first p rows. Matching the column order of `create_lags` is
# important because our reshape of A back into lag matrices relies on it.
@inline function _fill_X_from_Y!(
        X::AbstractMatrix{T}, Y::AbstractMatrix{T},
        n_obs::Int, n_vars_val::Int, n_lags_val::Int) where {T}
    n_eff = n_obs - n_lags_val
    @inbounds for t in 1:n_eff
        # t-th row of X corresponds to time index (t + n_lags) in Y.
        X[t, 1] = one(T)
        col = 1
        for lag in 1:n_lags_val
            y_row = t + n_lags_val - lag   # Y row index for this lag
            for j in 1:n_vars_val
                col += 1
                X[t, col] = Y[y_row, j]
            end
        end
    end
    return X
end

"""
    _fast_refit_ols!(ws, Y_boot, n_vars, n_lags) -> Bool

Estimate the unconstrained VAR(p) OLS coefficients, residual covariance,
and companion matrix from `Y_boot` into the workspace `ws`, using the
normal-equations Cholesky solve.

Returns `true` on success, `false` if the factorisation failed (caller
should mark the rep as NaN). On success, `ws.A`, `ws.U`, `ws.Σ`, `ws.Lchol`,
and `ws.F` are all written.
"""
function _fast_refit_ols!(
        ws::_BootstrapWorkspace{T},
        Y_boot::AbstractMatrix{T},
        n_vars_val::Int,
        n_lags_val::Int) where {T}
    n_obs = size(Y_boot, 1)
    n_eff = n_obs - n_lags_val

    # 1. Build X (preallocated), Y_est slice is a view into Y_boot[n_lags+1:end, :]
    _fill_X_from_Y!(ws.X, Y_boot, n_obs, n_vars_val, n_lags_val)
    Y_est = @view Y_boot[(n_lags_val + 1):n_obs, :]

    # 2. Normal equations: G = X'X, XtY = X'·Y_est
    mul!(ws.G, ws.X', ws.X)
    mul!(ws.XtY, ws.X', Y_est)

    # 3. Cholesky-solve G · A = XtY in-place. Copy XtY into A first because
    #    ldiv! overwrites the rhs.
    copyto!(ws.A, ws.XtY)
    local chol_G
    try
        chol_G = cholesky!(Symmetric(ws.G, :U))
    catch err
        err isa LinearAlgebra.PosDefException || rethrow(err)
        return false
    end
    ldiv!(chol_G, ws.A)

    # 4. Residuals U = Y_est - X·A
    mul!(ws.U, ws.X, ws.A)
    @inbounds for j in 1:n_vars_val, i in 1:n_eff

        ws.U[i, j] = Y_est[i, j] - ws.U[i, j]
    end

    # 5. Σ = U'U / df, df = n_eff - (1 + K·p)
    df = n_eff - (1 + n_vars_val * n_lags_val)
    mul!(ws.Σ, ws.U', ws.U)
    @inbounds for j in 1:n_vars_val, i in 1:n_vars_val

        ws.Σ[i, j] /= df
    end

    # 6. Cholesky factor L of Σ for CholeskyID. `cholesky!` overwrites, so
    #    copy first.
    copyto!(ws.Lchol, ws.Σ)
    local chol_Σ
    try
        chol_Σ = cholesky!(Symmetric(ws.Lchol, :U))
    catch err
        err isa LinearAlgebra.PosDefException || rethrow(err)
        return false
    end
    # Zero the upper triangle so ws.Lchol is the lower-triangular L.
    # `Symmetric(ws.Lchol, :U)` was passed to cholesky!, which writes the
    # upper triangle of the factor; we want the lower factor for the IRF.
    # The Cholesky factor satisfies U'U = Σ ⇒ L = U' (transpose into place).
    # Use a local copy to transpose cleanly.
    @inbounds for j in 1:n_vars_val, i in 1:(j - 1)
        # copy upper-triangle entry U[i,j] into lower-triangle L[j,i]
        ws.Lchol[j, i] = ws.Lchol[i, j]
        ws.Lchol[i, j] = zero(T)
    end

    # 7. Build companion matrix F:
    #    F = [ A₁  A₂  …  A_p ]
    #        [ I   0   …  0   ]
    #        [ 0   I   …  0   ]
    #        [ … ]
    #    where A_lag is a K × K matrix. From our A layout:
    #    A[1, :] = intercept  (unused by companion)
    #    A[2 + (lag-1)*K : 1 + lag*K, :] = A_lag'   (K × K, needs transpose)
    # Top K rows of F:
    @inbounds for lag in 1:n_lags_val
        col_start = (lag - 1) * n_vars_val
        for col in 1:n_vars_val
            for row in 1:n_vars_val
                # A_lag[row, col] = A[1 + (lag-1)*K + col, row]
                ws.F[row, col_start + col] = ws.A[1 + (lag - 1) * n_vars_val + col, row]
            end
        end
    end
    # Identity block below: F was allocated with zeros() so only the
    # identity entries need setting. But since we reuse F across reps and
    # only rewrite the top rows above, the identity below must have been
    # written correctly on construction (it was: the zeros from allocation
    # plus the identity entries set below).
    @inbounds for i in 1:((n_lags_val - 1) * n_vars_val)
        ws.F[n_vars_val + i, i] = one(T)
    end

    return true
end

"""
    _fast_cholesky_irf!(irf_out, ws, horizon; normalization=UnitStd())

Compute the Cholesky-identified IRF from workspace state into the caller-
supplied 3D `irf_out` of shape `(horizon+1, K, K)`. Uses ping-pong buffers
for companion powers; no allocation.

The impact response is `P = ws.Lchol` (or a UnitEffect-scaled version);
subsequent horizons advance `F_power ← F_power · F` and multiply the top
K × K block by P.
"""
function _fast_cholesky_irf!(
        irf_out::AbstractArray{T, 3},
        ws::_BootstrapWorkspace{T},
        horizon::Int;
        normalization::AbstractNormalization = UnitStd()) where {T}
    K = size(ws.Σ, 1)
    # Apply normalization to produce the identification matrix P.
    # For UnitStd, P = L (no change). For UnitEffect, divide columns by P[k,k].
    P = ws.Lchol   # alias; normalize! mutates in place on a copy if needed
    if normalization isa UnitStd
        # impact
        @inbounds for j in 1:K, i in 1:K

            irf_out[1, i, j] = ws.Lchol[i, j]
        end
        P_mat = ws.Lchol
    else
        # For UnitEffect, the impact response has P[j,j] = 1 for each shock j.
        # Rescale Lchol columns into the impact slice and use that as P.
        @inbounds for j in 1:K
            diag_val = ws.Lchol[j, j]
            for i in 1:K
                irf_out[1, i, j] = ws.Lchol[i, j] / diag_val
            end
        end
        # For horizons h ≥ 1 we need P explicitly; reuse the impact slice.
        # Since irf_out[1, :, :] now holds the normalised P, multiply by
        # companion power below using a view.
        P_mat = @view irf_out[1, :, :]
    end

    horizon == 0 && return irf_out

    # Ping-pong companion powers between the two workspace buffers using
    # local aliases (can't rebind fields of an immutable struct).
    # Initialize cur = F (horizon 1 power), nxt is scratch for F^(h+1).
    cur = ws.F_power
    nxt = ws.F_power_buf
    copyto!(cur, ws.F)

    # Horizon 1: irf[2, :, :] = F[1:K, 1:K] · P (reading from cur)
    @inbounds for j in 1:K, i in 1:K

        s = zero(T)
        for kk in 1:K
            s += cur[i, kk] * P_mat[kk, j]
        end
        irf_out[2, i, j] = s
    end

    # Horizons 2..horizon: advance F_power by one factor of F, then
    # multiply the top K × K block by P.
    for h in 2:horizon
        mul!(nxt, cur, ws.F)
        cur, nxt = nxt, cur
        @inbounds for j in 1:K, i in 1:K

            s = zero(T)
            for kk in 1:K
                s += cur[i, kk] * P_mat[kk, j]
            end
            irf_out[h + 1, i, j] = s
        end
    end
    return irf_out
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
    Y_boot = Matrix{T}(undef, n_obs, m)
    intercept_ = model.coefficients.intercept
    lags_ = model.coefficients.lags

    # Fast path for CholeskyID: reuse a single workspace across all reps
    # (in-place refit + fused IRF computation). Falls back to the generic
    # `fit(OLSVAR)` path for other identification types (sign restrictions,
    # etc.), which can't be handled by the Cholesky-specialised helpers.
    use_fast = identification isa CholeskyID && identification.ordering === nothing
    ws = use_fast ? _BootstrapWorkspace{T}(n_obs, m, n_lags_val) : nothing

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

        # Simulate bootstrap VAR into the preallocated buffer.
        simulate_var!(Y_boot, intercept_, lags_, ū, Y_original)

        if use_fast
            # Fast path: Cholesky on normal equations, fused IRF.
            ok = _fast_refit_ols!(ws, Y_boot, m, n_lags_val)
            if ok
                try
                    _fast_cholesky_irf!(view(irf_boot,r,:,:,:), ws, horizon;
                        normalization = normalization)
                catch
                    n_failed += 1
                    fill!(view(irf_boot,r,:,:,:), NaN)
                end
            else
                n_failed += 1
                fill!(view(irf_boot,r,:,:,:), NaN)
            end
        else
            # Generic path (unchanged): fit(OLSVAR) + rotation_matrix + compute_irf_point.
            try
                var_boot = refit_for_bootstrap(model, Y_boot, n_lags_val)
                P_boot = rotation_matrix(var_boot, identification)
                normalize!(P_boot, normalization)
                irf_view = view(irf_boot,r,:,:,:)
                copyto!(irf_view, compute_irf_point(var_boot, P_boot, horizon))
            catch
                n_failed += 1
                fill!(view(irf_boot,r,:,:,:), NaN)
            end
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

    # Pre-allocate reusable buffers
    ū = Matrix{T}(undef, n_obs, m)
    Y_boot = Matrix{T}(undef, n_obs, m)
    intercept_ = model.coefficients.intercept
    lags_ = model.coefficients.lags

    use_fast = identification isa CholeskyID && identification.ordering === nothing
    ws = use_fast ? _BootstrapWorkspace{T}(n_obs, m, n_lags_val) : nothing

    n_failed = 0
    for r in 1:reps
        # Standard bootstrap: resample rows with replacement
        @inbounds for i in 1:n_obs
            src = rand(rng, 1:n_obs)
            for j in 1:m
                ū[i, j] = u[src, j]
            end
        end

        # Simulate bootstrap VAR into the preallocated buffer.
        simulate_var!(Y_boot, intercept_, lags_, ū, Y_original)

        if use_fast
            ok = _fast_refit_ols!(ws, Y_boot, m, n_lags_val)
            if ok
                try
                    _fast_cholesky_irf!(view(irf_boot,r,:,:,:), ws, horizon;
                        normalization = normalization)
                catch
                    n_failed += 1
                    fill!(view(irf_boot,r,:,:,:), NaN)
                end
            else
                n_failed += 1
                fill!(view(irf_boot,r,:,:,:), NaN)
            end
        else
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

    # Preallocated simulation buffer reused across reps.
    Y_boot = Matrix{T}(undef, n_obs, m)
    intercept_ = model.coefficients.intercept
    lags_ = model.coefficients.lags

    use_fast = identification isa CholeskyID && identification.ordering === nothing
    ws = use_fast ? _BootstrapWorkspace{T}(n_obs, m, n_lags_val) : nothing

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

        # Step 4: Simulate bootstrap VAR into the preallocated buffer.
        simulate_var!(Y_boot, intercept_, lags_, ū, Y_original)

        # Step 5: Re-estimate and compute IRF
        if use_fast
            ok = _fast_refit_ols!(ws, Y_boot, m, n_lags_val)
            if ok
                try
                    _fast_cholesky_irf!(view(irf_boot,r,:,:,:), ws, horizon;
                        normalization = normalization)
                catch
                    n_failed += 1
                    fill!(view(irf_boot,r,:,:,:), NaN)
                end
            else
                n_failed += 1
                fill!(view(irf_boot,r,:,:,:), NaN)
            end
        else
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
