# ============================================================================
# Bootstrap Methods for Inference
# ============================================================================
#
# Residual bootstraps for VAR impulse responses (`WildBootstrap`, `Bootstrap`,
# `BlockBootstrap`). They differ only in how the residuals are resampled
# (`_residual_resampler`); one replication loop (`_bootstrap_irf_draws`) serves
# `bootstrap_irf` and `compute_inference_bands` in var/irfs.jl.

# ============================================================================
# Failed replications
# ============================================================================

# Exceptions a resampled series can legitimately trigger (a singular design or
# covariance matrix, a negative argument to `sqrt`). A replication that raises one
# of these is a failed draw; every other exception propagates.
const _DrawFailure = Union{
    LinearAlgebra.PosDefException, LinearAlgebra.SingularException,
    LinearAlgebra.LAPACKException, DomainError
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

function _check_block_length(block_length::Integer, n_obs::Integer)
    1 <= block_length <= n_obs || throw(
        ArgumentError(
        "block_length = $block_length must lie in 1:$n_obs (the number of residual periods)"
    )
    )
    return nothing
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

Preallocated buffers for the Cholesky-identification fast path of
`_bootstrap_irf_draws`. One instance is created per bootstrap call and reused
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
    P = ws.Lchol   # alias; normalize_impact! mutates in place on a copy if needed
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
    normalize_impact!(P_boot, normalization)
    return copyto!(out, compute_irf_point(var_boot, P_boot, horizon))
end

# ============================================================================
# Residual resampling
# ============================================================================
#
# `_residual_resampler(inference, u)` returns a function `rng -> ū` that fills and
# returns one resampled residual series shaped like `u`, reusing its buffers
# across replications.

_bootstrap_name(::WildBootstrap) = "Wild bootstrap"
_bootstrap_name(::Bootstrap) = "Standard bootstrap"
_bootstrap_name(::BlockBootstrap) = "Block bootstrap"

# Wild bootstrap: each period's residual vector is multiplied by a Rademacher
# weight (±1 with equal probability), which keeps conditional heteroskedasticity
# and the cross-equation correlation of the residuals.
function _residual_resampler(::WildBootstrap, u::AbstractMatrix{T}) where {T}
    ω = similar(u, axes(u, 1))
    ū = similar(u)
    return function (rng)
        for i in eachindex(ω)
            ω[i] = rand(rng, Bool) ? one(T) : -one(T)
        end
        ū .= ω .* u
        return ū
    end
end

# Standard bootstrap: residual vectors resampled i.i.d. with replacement.
function _residual_resampler(::Bootstrap, u::AbstractMatrix)
    Base.require_one_based_indexing(u)
    n_obs = size(u, 1)
    ū = similar(u)
    return function (rng)
        for i in 1:n_obs
            ū[i, :] .= view(u, rand(rng, 1:n_obs), :)
        end
        return ū
    end
end

# Moving block bootstrap: N = ⌈T/ℓ⌉ blocks of ℓ consecutive residual vectors,
# drawn with replacement, concatenated, and trimmed to T rows. Position s within
# a block is centered by the mean of the residuals at s, s+ℓ, s+2ℓ, ..., so the
# resampled residuals have mean (approximately) zero and the simulated series
# does not drift.
function _residual_resampler(inference::BlockBootstrap, u::AbstractMatrix{T}) where {T}
    Base.require_one_based_indexing(u)
    n_obs, m = size(u)
    ℓ = inference.block_length
    _check_block_length(ℓ, n_obs)
    N = cld(n_obs, ℓ)
    max_start = n_obs - ℓ + 1
    mean_at_s = [mean(view(u, s:ℓ:(n_obs - ℓ + s), j)) for s in 1:ℓ, j in 1:m]
    ū_full = Matrix{T}(undef, N * ℓ, m)
    ū = view(ū_full, 1:n_obs, :)
    return function (rng)
        for b in 1:N
            start = rand(rng, 1:max_start)
            for s in 1:ℓ, j in 1:m

                ū_full[ℓ * (b - 1) + s, j] = u[start + s - 1, j] - mean_at_s[s, j]
            end
        end
        return ū
    end
end

# Residual bootstraps refit the VAR on a resampled series; identifications that
# carry data aligned with the residuals (an instrument) refuse them.
_check_resamples_instrument(::AbstractIdentification, ::_ResidualBootstrap) = nothing

# ============================================================================
# Replication loop
# ============================================================================

# Bootstrap IRF draws `(n_ok, K, K, horizon+1)` of the successful replications:
# each replication simulates the VAR from resampled residuals, starting from the
# observed initial values, and refits it (see `_bootstrap_irf_draw!`). Failed
# replications are dropped under the rules of `_successful_draws`.
function _bootstrap_irf_draws(
        model::VARModel{T}, identification::AbstractIdentification,
        inference::_ResidualBootstrap, horizon::Integer,
        normalization::AbstractNormalization, rng::AbstractRNG
) where {T}
    _check_resamples_instrument(identification, inference)
    u = residuals(model)
    n_obs, m = size(u)
    resample = _residual_resampler(inference, u)

    Y_boot = Matrix{T}(undef, n_obs, m)
    intercept_ = model.coefficients.intercept
    lags_ = model.coefficients.lags

    # The workspace path handles only CholeskyID in the model's variable order;
    # other identifications refit through `refit_for_bootstrap`.
    use_fast = identification isa CholeskyID && identification.ordering === nothing
    ws = use_fast ? _BootstrapWorkspace{T}(n_obs, m, n_lags(model)) : nothing

    reps = inference.reps
    irf_boot = zeros(T, reps, m, m, horizon + 1)
    ok = falses(reps)
    for r in 1:reps
        simulate_var!(Y_boot, intercept_, lags_, resample(rng), model.Y)
        ok[r] = _try_draw!(view(irf_boot, r, :, :, :)) do out
            _bootstrap_irf_draw!(
                out, ws, model, Y_boot, identification, horizon, normalization)
        end
    end

    return _successful_draws(
        _bootstrap_name(inference), irf_boot, ok, inference.max_failure_rate
    )
end

# ============================================================================
# Entry point
# ============================================================================

"""
    bootstrap_irf(model, identification, inference; horizon=24, normalization=UnitStd(), rng=Random.default_rng())

Bootstrap draws of the impulse responses of `model` under `identification`.
`inference` is a `WildBootstrap`, `Bootstrap`, or `BlockBootstrap`; it sets the
number of replications, the residual resampling scheme, and `max_failure_rate`
(its `save_draws` field is ignored: the draws are the return value).

Each replication resamples the residuals, simulates the VAR from the observed
initial values, refits it, and computes the responses up to `horizon` with the
impact matrix normalized by `normalization`.

Returns an array `(n_ok, n_vars, n_shocks, horizon+1)`, where `n_ok ≤ inference.reps`
counts the successful replications. A replication fails when its refit raises
`PosDefException`, `SingularException`, `LAPACKException`, or `DomainError`, or
when its responses are not all finite; any other exception propagates. Failed
replications are dropped with a warning, and the call throws when more than a
fraction `inference.max_failure_rate` of them fail.

`irf(model, identification; inference)` runs the same bootstrap and returns an
`IRFResult` with point estimate, standard errors, and bands. For
`IVIdentification` use `ProxySVARMBB` through `irf` or `proxy_svar_mbb`: the
residual bootstraps do not resample the instrument and throw an `ArgumentError`.

# Example
```julia
model = fit(OLSVAR, Y, 2)
draws = bootstrap_irf(model, CholeskyID(), BlockBootstrap(1000; block_length = 8);
    horizon = 24, rng = StableRNG(1))
```
"""
function bootstrap_irf(
        model::VARModel, identification::AbstractIdentification,
        inference::_ResidualBootstrap;
        horizon::Integer = 24,
        normalization::AbstractNormalization = UnitStd(),
        rng::AbstractRNG = Random.default_rng()
)
    horizon > 0 || throw(ArgumentError("horizon must be positive"))
    return _bootstrap_irf_draws(
        model, identification, inference, horizon, normalization, rng)
end

function bootstrap_irf(
        ::VARModel, ::AbstractIdentification, inference::InferenceType; kwargs...
)
    throw(
        ArgumentError(
        "bootstrap_irf supports WildBootstrap, Bootstrap, and BlockBootstrap " *
        "(got $(nameof(typeof(inference)))). For ProxySVARMBB draws use " *
        "`proxy_svar_mbb`; for Analytic() use `irf`."
    )
    )
end

# ============================================================================
# Deprecated forms
# ============================================================================

function bootstrap_irf(
        model::VARModel, identification::AbstractIdentification,
        horizon::Int, reps::Int;
        method::Symbol = :wild,
        block_length::Int = 10,
        normalization::AbstractNormalization = UnitStd(),
        max_failure_rate::Real = 0.1,
        rng::AbstractRNG = Random.default_rng()
)
    inference, spelled = if method === :wild
        WildBootstrap(reps; max_failure_rate), "WildBootstrap(reps)"
    elseif method === :standard
        Bootstrap(reps; max_failure_rate), "Bootstrap(reps)"
    elseif method === :block
        BlockBootstrap(reps; block_length, max_failure_rate),
        "BlockBootstrap(reps; block_length)"
    else
        throw(ArgumentError("method must be :wild, :standard, or :block"))
    end
    Base.depwarn(
        "`bootstrap_irf(model, identification, horizon, reps; method = $(repr(method)))` " *
        "is deprecated; use `bootstrap_irf(model, identification, $spelled; horizon)`.",
        :bootstrap_irf
    )
    return bootstrap_irf(model, identification, inference; horizon, normalization, rng)
end

function bootstrap_irf_wild(
        model::VARModel, identification::AbstractIdentification,
        horizon::Int, reps::Int, rng::AbstractRNG;
        normalization::AbstractNormalization = UnitStd(), max_failure_rate::Real = 0.1
)
    Base.depwarn(
        "`bootstrap_irf_wild(model, identification, horizon, reps, rng)` is deprecated; " *
        "use `bootstrap_irf(model, identification, WildBootstrap(reps); horizon, rng)`.",
        :bootstrap_irf_wild
    )
    return bootstrap_irf(
        model, identification, WildBootstrap(reps; max_failure_rate);
        horizon, normalization, rng
    )
end

function bootstrap_irf_standard(
        model::VARModel, identification::AbstractIdentification,
        horizon::Int, reps::Int, rng::AbstractRNG;
        normalization::AbstractNormalization = UnitStd(), max_failure_rate::Real = 0.1
)
    Base.depwarn(
        "`bootstrap_irf_standard(model, identification, horizon, reps, rng)` is " *
        "deprecated; use `bootstrap_irf(model, identification, Bootstrap(reps); " *
        "horizon, rng)`.",
        :bootstrap_irf_standard
    )
    return bootstrap_irf(
        model, identification, Bootstrap(reps; max_failure_rate);
        horizon, normalization, rng
    )
end

function bootstrap_irf_block(
        model::VARModel, identification::AbstractIdentification,
        horizon::Int, reps::Int, block_length::Int, rng::AbstractRNG;
        normalization::AbstractNormalization = UnitStd(), max_failure_rate::Real = 0.1
)
    Base.depwarn(
        "`bootstrap_irf_block(model, identification, horizon, reps, block_length, rng)` " *
        "is deprecated; use `bootstrap_irf(model, identification, " *
        "BlockBootstrap(reps; block_length); horizon, rng)`.",
        :bootstrap_irf_block
    )
    return bootstrap_irf(
        model, identification, BlockBootstrap(reps; block_length, max_failure_rate);
        horizon, normalization, rng
    )
end
