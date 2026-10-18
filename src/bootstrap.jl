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
# Failed replications
# ============================================================================

# Exceptions a resampled series can legitimately trigger (a singular design or
# covariance matrix, a negative argument to `sqrt`). A replication that raises one
# of these is a failed draw; every other exception propagates.
const _DrawFailure = Union{
    LinearAlgebra.PosDefException, LinearAlgebra.SingularException,
    LinearAlgebra.LAPACKException, DomainError,
}

"""
    _try_draw!(compute!, outs...) -> Bool

Run `compute!(outs...)` and return whether it produced a usable draw: `false` when
it throws an exception in `_DrawFailure` or leaves a non-finite value in any of the
arrays `outs`. Any other exception is rethrown.
"""
function _try_draw!(compute!, outs::AbstractArray...)
    try
        compute!(outs...)
    catch err
        err isa _DrawFailure || rethrow()
        return false
    end
    return all(out -> all(isfinite, out), outs)
end

"""
    _check_failure_rate(method_name, n_failed, reps, max_failure_rate)

Throw when every replication failed or more than a fraction `max_failure_rate` of
the `reps` replications failed; otherwise warn when any failed. Callers exclude the
failed replications from every statistic computed from the draws.
"""
function _check_failure_rate(
        method_name::String, n_failed::Integer, reps::Integer, max_failure_rate::Real
    )
    n_failed == 0 && return nothing
    pct = round(100 * n_failed / reps; digits = 1)
    if n_failed == reps || n_failed > max_failure_rate * reps
        error(
            "$method_name: $n_failed / $reps replications failed ($pct%), more than " *
                "max_failure_rate = $max_failure_rate. Failed replications raise a " *
                "singular or non-positive-definite matrix, or produce non-finite " *
                "responses; check the stability of the model and the sample size."
        )
    end
    @warn "$method_name: $n_failed / $reps replications failed ($pct%). They are " *
        "excluded; bands and standard errors use the remaining $(reps - n_failed)."
    return nothing
end

"""
    _successful_draws(method_name, draws, ok, max_failure_rate)

Apply `_check_failure_rate` and return the draws (first dimension) marked in `ok`.
"""
function _successful_draws(
        method_name::String, draws::AbstractArray{T, 4}, ok::AbstractVector{Bool},
        max_failure_rate::Real
    ) where {T}
    n_failed = count(!, ok)
    _check_failure_rate(method_name, n_failed, length(ok), max_failure_rate)
    return n_failed == 0 ? draws : draws[ok, :, :, :]
end

function _check_max_failure_rate(max_failure_rate::Real)
    0 <= max_failure_rate < 1 ||
        throw(ArgumentError("max_failure_rate must lie in [0, 1), got $max_failure_rate"))
    return max_failure_rate
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
# are numerical noise. A failed factorisation throws `PosDefException`, which
# `_try_draw!` counts as a failed replication.

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
        n_obs::Int, n_vars_val::Int, n_lags_val::Int
    ) where {T}
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
    _fast_refit_ols!(ws, Y_boot, n_vars, n_lags) -> ws

Estimate the unconstrained VAR(p) OLS coefficients, residual covariance,
and companion matrix from `Y_boot` into the workspace `ws`, using the
normal-equations Cholesky solve. Writes `ws.A`, `ws.U`, `ws.Σ`, `ws.Lchol`,
and `ws.F`; throws `PosDefException` when `X'X` or `Σ` is not positive definite.
"""
function _fast_refit_ols!(
        ws::_BootstrapWorkspace{T},
        Y_boot::AbstractMatrix{T},
        n_vars_val::Int,
        n_lags_val::Int
    ) where {T}
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
    ldiv!(cholesky!(Symmetric(ws.G, :U)), ws.A)

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
    cholesky!(Symmetric(ws.Lchol, :U))
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

    return ws
end

"""
    _fast_cholesky_irf!(irf_out, ws, horizon; normalization=UnitStd())

Compute the Cholesky-identified IRF from workspace state into the caller-
supplied 3D `irf_out` of shape `(K, K, horizon+1)`. Uses ping-pong buffers
for companion powers; no allocation.

The impact response is `P = ws.Lchol` (or a UnitEffect-scaled version);
subsequent horizons advance `F_power ← F_power · F` and multiply the top
K × K block by P.
"""
function _fast_cholesky_irf!(
        irf_out::AbstractArray{T, 3},
        ws::_BootstrapWorkspace{T},
        horizon::Int;
        normalization::AbstractNormalization = UnitStd()
    ) where {T}
    K = size(ws.Σ, 1)
    # Apply normalization to produce the identification matrix P.
    # For UnitStd, P = L (no change). For UnitEffect, divide columns by P[k,k].
    P = ws.Lchol   # alias; normalize! mutates in place on a copy if needed
    if normalization isa UnitStd
        # impact
        @inbounds for j in 1:K, i in 1:K

            irf_out[i, j, 1] = ws.Lchol[i, j]
        end
        P_mat = ws.Lchol
    else
        # For UnitEffect, the impact response has P[j,j] = 1 for each shock j.
        # Rescale Lchol columns into the impact slice and use that as P.
        @inbounds for j in 1:K
            diag_val = ws.Lchol[j, j]
            for i in 1:K
                irf_out[i, j, 1] = ws.Lchol[i, j] / diag_val
            end
        end
        # For horizons h ≥ 1 we need P explicitly; reuse the impact slice.
        # Since irf_out[:, :, 1] now holds the normalised P, multiply by
        # companion power below using a view.
        P_mat = @view irf_out[:, :, 1]
    end

    horizon == 0 && return irf_out

    # Ping-pong companion powers between the two workspace buffers using
    # local aliases (can't rebind fields of an immutable struct).
    # Initialize cur = F (horizon 1 power), nxt is scratch for F^(h+1).
    cur = ws.F_power
    nxt = ws.F_power_buf
    copyto!(cur, ws.F)

    # Horizon 1: irf[:, :, 2] = F[1:K, 1:K] · P (reading from cur)
    @inbounds for j in 1:K, i in 1:K

        s = zero(T)
        for kk in 1:K
            s += cur[i, kk] * P_mat[kk, j]
        end
        irf_out[i, j, 2] = s
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
            irf_out[i, j, h + 1] = s
        end
    end
    return irf_out
end

# One bootstrap IRF draw `(K, K, horizon+1)` into `out`, refitting the VAR on the
# simulated series `Y_boot`: through the workspace when the identification is the
# default Cholesky (`ws` given), through `refit_for_bootstrap` otherwise.
function _bootstrap_irf_draw!(
        out, ws::_BootstrapWorkspace, model, Y_boot, identification, horizon, normalization
    )
    _fast_refit_ols!(ws, Y_boot, n_vars(model), n_lags(model))
    return _fast_cholesky_irf!(out, ws, horizon; normalization)
end

function _bootstrap_irf_draw!(
        out, ::Nothing, model, Y_boot, identification, horizon, normalization
    )
    var_boot = refit_for_bootstrap(model, Y_boot, n_lags(model))
    P_boot = rotation_matrix(var_boot, identification)
    normalize!(P_boot, normalization)
    return copyto!(out, compute_irf_point(var_boot, P_boot, horizon))
end

# ============================================================================
# Wild Bootstrap
# ============================================================================

"""
    bootstrap_irf_wild(model, identification, horizon, reps, rng)

Wild bootstrap for VAR impulse responses.

Returns the IRF draws of the successful replications (see `bootstrap_irf`). For an assembled `IRFResult` with bands, use
`irf(model, identification; inference=WildBootstrap(...))`, which calls this.

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
        normalization::AbstractNormalization = UnitStd(),
        max_failure_rate::Real = 0.1
    ) where {T}
    _check_max_failure_rate(max_failure_rate)
    m = n_vars(model)
    n_lags_val = n_lags(model)

    # Preallocate output: (reps, n_vars, n_shocks, horizon+1)
    irf_boot = zeros(T, reps, m, m, horizon + 1)

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

    ok = falses(reps)
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

        ok[r] = _try_draw!(view(irf_boot, r, :, :, :)) do out
            _bootstrap_irf_draw!(out, ws, model, Y_boot, identification, horizon, normalization)
        end
    end

    return _successful_draws("Wild bootstrap", irf_boot, ok, max_failure_rate)
end

# ============================================================================
# Standard Bootstrap
# ============================================================================

"""
    bootstrap_irf_standard(model, identification, horizon, reps, rng)

Standard i.i.d. bootstrap for VAR impulse responses.

Returns the IRF draws of the successful replications (see `bootstrap_irf`). For an assembled `IRFResult` with bands, use
`irf(model, identification; inference=Bootstrap(...))`, which calls this.

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
        normalization::AbstractNormalization = UnitStd(),
        max_failure_rate::Real = 0.1
    ) where {T}
    _check_max_failure_rate(max_failure_rate)
    m = n_vars(model)
    n_lags_val = n_lags(model)
    n_obs = size(model.residuals, 1)

    # Preallocate output
    irf_boot = zeros(T, reps, m, m, horizon + 1)

    u = residuals(model)
    Y_original = model.Y

    # Pre-allocate reusable buffers
    ū = Matrix{T}(undef, n_obs, m)
    Y_boot = Matrix{T}(undef, n_obs, m)
    intercept_ = model.coefficients.intercept
    lags_ = model.coefficients.lags

    use_fast = identification isa CholeskyID && identification.ordering === nothing
    ws = use_fast ? _BootstrapWorkspace{T}(n_obs, m, n_lags_val) : nothing

    ok = falses(reps)
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

        ok[r] = _try_draw!(view(irf_boot, r, :, :, :)) do out
            _bootstrap_irf_draw!(out, ws, model, Y_boot, identification, horizon, normalization)
        end
    end

    return _successful_draws("Standard bootstrap", irf_boot, ok, max_failure_rate)
end

# ============================================================================
# Block Bootstrap
# ============================================================================

"""
    bootstrap_irf_block(model, identification, horizon, reps, block_length, rng)

Moving block bootstrap for VAR impulse responses.

Returns the IRF draws of the successful replications (see `bootstrap_irf`). For an assembled `IRFResult` with bands, use
`irf(model, identification; inference=BlockBootstrap(...))`, which calls this.

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
        normalization::AbstractNormalization = UnitStd(),
        max_failure_rate::Real = 0.1
    ) where {T}
    _check_max_failure_rate(max_failure_rate)
    m = n_vars(model)
    n_lags_val = n_lags(model)
    n_obs = size(model.residuals, 1)  # T
    ℓ = block_length

    # Number of blocks needed to cover T observations
    N = cld(n_obs, ℓ)

    # Preallocate
    irf_boot = zeros(T, reps, m, m, horizon + 1)
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

    ok = falses(reps)
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
        ok[r] = _try_draw!(view(irf_boot, r, :, :, :)) do out
            _bootstrap_irf_draw!(out, ws, model, Y_boot, identification, horizon, normalization)
        end
    end

    return _successful_draws("Block bootstrap", irf_boot, ok, max_failure_rate)
end

# ============================================================================
# Backward-compatible wrapper
# ============================================================================

"""
    bootstrap_irf(model, identification, horizon, reps; method=:wild, block_length=10, max_failure_rate=0.1, rng=...)

Backward-compatible wrapper that dispatches to `bootstrap_irf_wild`,
`bootstrap_irf_standard`, or `bootstrap_irf_block` based on `method`.

Returns the bootstrap IRF draws `(n_ok, n_vars, n_shocks, horizon+1)`, where
`n_ok ≤ reps` counts the successful replications. A replication fails when its
refit raises `PosDefException`, `SingularException`, `LAPACKException`, or
`DomainError`, or when its responses are not all finite; any other exception
propagates. Failed replications are dropped with a warning, and the call throws when
more than a fraction `max_failure_rate` (in `[0, 1)`) of them fail. Most
users should instead call `irf(model, identification; inference=...)`, which runs
the same bootstrap internally and returns an `IRFResult` with point estimate,
standard errors, and confidence bands assembled. This family is the lower-level
entry point for callers that want the draws themselves.

The `parallel` keyword is deprecated; use Julia's built-in threading
or Distributed.pmap externally if parallel bootstrap is needed.
"""
function bootstrap_irf(
        model::VARModel{T}, identification::AbstractIdentification,
        horizon::Int, reps::Int;
        method::Symbol = :wild,
        block_length::Int = 10,
        normalization::AbstractNormalization = UnitStd(),
        max_failure_rate::Real = 0.1,
        parallel::Symbol = :none,
        rng::AbstractRNG = Random.default_rng()
    ) where {T}
    method ∈ [:wild, :standard, :block] ||
        throw(ArgumentError("method must be :wild, :standard, or :block"))

    if parallel != :none
        @warn "parallel keyword is deprecated in bootstrap_irf. " *
            "Falling back to serial execution."
    end

    if method == :wild
        return bootstrap_irf_wild(
            model, identification, horizon, reps, rng;
            normalization, max_failure_rate
        )
    elseif method == :standard
        return bootstrap_irf_standard(
            model, identification, horizon, reps, rng;
            normalization, max_failure_rate
        )
    else  # :block
        return bootstrap_irf_block(
            model, identification, horizon, reps, block_length, rng;
            normalization, max_failure_rate
        )
    end
end
