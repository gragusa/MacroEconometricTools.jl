# ============================================================================
# Proxy-SVAR Bootstrap — Jentsch & Lunsford (2022) MBB
# ============================================================================
#
# Asymptotically valid bootstrap inference for proxy-SVARs.
# Key features:
#   - Joint resampling of (residuals, proxy) in blocks
#   - Position-specific centering (J&L correction)
#   - Anderson-Rubin confidence sets (weak-instrument robust)
#   - Hall's percentile intervals (bias correction)
#
# References:
#   Jentsch & Lunsford (2022), "Asymptotically Valid Bootstrap Inference
#     for Proxy SVARs"
#   Anderson & Rubin (1949)

# ============================================================================
# Proxy-SVAR dynamics: IRFs, FEVD, SVMA from estimated parameters
# ============================================================================

"""
    proxy_svar_dynamics(A_est, Σ_uu, Σ_um, H1, p, norm_scale, n_imp, target)

Compute IRFs, normalized IRFs, FEVD, and structural VMA from estimated
proxy-SVAR parameters.

# Arguments
- `A_est`: (K, 1+K*p) coefficient matrix [intercept | A₁ | ... | Aₚ] (row = equation)
- `Σ_uu`: (K, K) residual covariance (no df correction)
- `Σ_um`: (K,) covariance of residuals with proxy
- `H1`: (K,) identified column of structural impact matrix
- `p`: number of VAR lags
- `norm_scale`: normalization for IRF (typically -1.0, so impact on target = -1)
- `n_imp`: number of horizons (including 0)
- `target`: index of the instrumented variable (normalization is on `H1[target]`)

# Returns
Named tuple: `(irf, irf_norm, fevd, svma)`, each (K, n_imp) matrix.
"""
function proxy_svar_dynamics(
        A_est::AbstractMatrix{T}, Σ_uu::AbstractMatrix{T},
        Σ_um::AbstractVector{T}, H1::AbstractVector{T},
        p::Int, norm_scale::Float64, n_imp::Int, target::Int
    ) where {T}
    buf = ProxyDynamicsBuffers{T}(length(H1), p, n_imp)
    proxy_svar_dynamics!(buf, A_est, Σ_uu, Σ_um, H1, p, norm_scale, n_imp, target)
    return (irf = buf.irf, irf_norm = buf.irf_norm, fevd = buf.fevd, svma = buf.svma)
end

"""
    ProxyDynamicsBuffers{T}

Pre-allocated workspace for `proxy_svar_dynamics!`. Holds output matrices and
scratch arrays for the companion-form recursion.
"""
struct ProxyDynamicsBuffers{T}
    irf::Matrix{T}           # (K, n_imp)
    irf_norm::Matrix{T}      # (K, n_imp)
    fevd::Matrix{T}          # (K, n_imp)
    svma::Matrix{T}          # (K, n_imp)
    comp::Matrix{T}          # (K*p, K*p) — companion matrix
    comp_power::Matrix{T}    # (K*p, K*p)
    comp_power_buf::Matrix{T} # (K*p, K*p)
    denom::Matrix{T}         # (K, K)
    Φ_Σ::Matrix{T}           # (K, K)
    numer::Vector{T}         # (K,)
    irf_h::Vector{T}         # (K,)
end

function ProxyDynamicsBuffers{T}(K::Int, p::Int, n_imp::Int) where {T}
    Kp = K * p
    return ProxyDynamicsBuffers{T}(
        Matrix{T}(undef, K, n_imp),
        Matrix{T}(undef, K, n_imp),
        Matrix{T}(undef, K, n_imp),
        Matrix{T}(undef, K, n_imp),
        zeros(T, Kp, Kp),
        Matrix{T}(undef, Kp, Kp),
        Matrix{T}(undef, Kp, Kp),
        Matrix{T}(undef, K, K),
        Matrix{T}(undef, K, K),
        Vector{T}(undef, K),
        Vector{T}(undef, K)
    )
end

"""
    proxy_svar_dynamics!(buf, A_est, Σ_uu, Σ_um, H1, p, norm_scale, n_imp, target)

In-place variant of `proxy_svar_dynamics`. Writes results into `buf`'s output
matrices (`buf.irf`, `buf.irf_norm`, `buf.fevd`, `buf.svma`) and returns `buf`.
"""
function proxy_svar_dynamics!(
        buf::ProxyDynamicsBuffers{T},
        A_est::AbstractMatrix{T}, Σ_uu::AbstractMatrix{T},
        Σ_um::AbstractVector{T}, H1::AbstractVector{T},
        p::Int, norm_scale::Float64, n_imp::Int, target::Int
    ) where {T}
    Base.require_one_based_indexing(A_est, Σ_uu, Σ_um, H1)
    K = length(H1)
    Kp = K * p
    size(buf.irf) == (K, n_imp) && size(buf.comp) == (Kp, Kp) || throw(
        DimensionMismatch(
            "buffers sized for K=$(size(buf.irf, 1)), K*p=$(size(buf.comp, 1)), n_imp=$(size(buf.irf, 2)); got K=$K, p=$p, n_imp=$n_imp"
        )
    )
    size(A_est) == (K, 1 + Kp) && size(Σ_uu) == (K, K) && length(Σ_um) == K || throw(
        DimensionMismatch(
            "expected A_est $(K)×$(1 + Kp), Σ_uu $(K)×$K, Σ_um of length $K; got $(size(A_est)), $(size(Σ_uu)), $(length(Σ_um))"
        )
    )
    1 <= target <= K || throw(ArgumentError("target must lie in 1:$K, got $target"))

    # Companion matrix (reset to zero; reused buffer)
    fill!(buf.comp, zero(T))
    if p == 1
        buf.comp[:, :] .= view(A_est, :, 2:(K + 1))
    else
        buf.comp[1:K, :] .= view(A_est, :, 2:(K * p + 1))
        @inbounds for j in 1:(K * (p - 1))
            buf.comp[K + j, j] = one(T)
        end
    end

    irf_out = buf.irf
    irf_norm = buf.irf_norm
    fevd = buf.fevd
    svma = buf.svma

    # Horizon 0
    @inbounds for k in 1:K
        irf_out[k, 1] = H1[k]
        fevd[k, 1] = H1[k]^2 / Σ_uu[k, k]
        svma[k, 1] = Σ_um[k]
    end

    # Normalized IRF at horizon 0
    inv_H1_t = norm_scale / H1[target]
    @inbounds for k in 1:K
        irf_norm[k, 1] = inv_H1_t * H1[k]
    end

    # comp_power starts as I; use local bindings so we can pointer-swap safely
    # (swap only affects this call — the struct fields stay unchanged).
    comp_power = buf.comp_power
    comp_power_buf = buf.comp_power_buf
    fill!(comp_power, zero(T))
    @inbounds for i in 1:Kp
        comp_power[i, i] = one(T)
    end

    @inbounds for k in 1:K
        buf.numer[k] = H1[k]^2
    end
    copyto!(buf.denom, Σ_uu)

    @inbounds for h in 2:n_imp
        mul!(comp_power_buf, comp_power, buf.comp)
        comp_power, comp_power_buf = comp_power_buf, comp_power
        Φ_h = view(comp_power, 1:K, 1:K)

        # IRF
        mul!(buf.irf_h, Φ_h, H1)
        @inbounds for k in 1:K
            irf_out[k, h] = buf.irf_h[k]
            irf_norm[k, h] = inv_H1_t * buf.irf_h[k]
            buf.numer[k] += buf.irf_h[k]^2
        end

        # FEVD denominator: denom += Φ_h * Σ_uu * Φ_h'
        mul!(buf.Φ_Σ, Φ_h, Σ_uu)
        mul!(buf.denom, buf.Φ_Σ, Φ_h', one(T), one(T))
        @inbounds for k in 1:K
            fevd[k, h] = buf.numer[k] / buf.denom[k, k]
        end

        # Structural VMA
        mul!(view(svma, :, h), Φ_h, Σ_um)
    end

    return buf
end

# ============================================================================
# estimate_proxy_svar
# ============================================================================

"""
    estimate_proxy_svar(yy, xx, mm)

Estimate proxy-SVAR using the Jentsch & Lunsford (2022) formula:
  Σ_uu = U'U / T  (no df correction)
  Σ_um = U'm / T
  φ = √(Σ_um' Σ_uu⁻¹ Σ_um)
  H1 = Σ_um / φ

Allocating form of [`estimate_proxy_svar!`](@ref).

Returns: (A_est, U_est, Σ_uu, Σ_um, H1)
- A_est: (K, 1+K*p) coefficient matrix (row = equation)
- U_est: (T_eff, K) residuals
- Σ_uu: (K, K) residual covariance
- Σ_um: (K,) covariance with proxy
- H1: (K,) identified impact column
"""
function estimate_proxy_svar(
        yy::AbstractMatrix{T},
        xx::AbstractMatrix{T},
        mm::AbstractVector{T}
    ) where {T <: Union{Float32, Float64}}
    buf = ProxySvarBuffers{T}(size(yy, 1), size(yy, 2), size(xx, 2))
    return estimate_proxy_svar!(buf, yy, xx, mm)
end

"""
    ProxySvarBuffers{T}

Pre-allocated workspace for `estimate_proxy_svar!`. Reused across bootstrap
draws to avoid per-draw allocations. `T` must be a LAPACK real type
(`Float32` or `Float64`).
"""
struct ProxySvarBuffers{T <: Union{Float32, Float64}}
    A_raw::Matrix{T}      # (d, K)  — raw coefficients (d = 1 + K*p)
    A_est::Matrix{T}      # (K, d)  — transposed
    U_est::Matrix{T}      # (TT, K) — residuals
    Σ_uu::Matrix{T}       # (K, K)
    Σ_um::Vector{T}       # (K,)
    H1::Vector{T}         # (K,)
    X_qr::Matrix{T}       # (TT, d) — overwritten by the QR factors of X
    Y_qr::Matrix{T}       # (TT, K) — overwritten by Q'Y
    qr_ws::QRWs{T}        # LAPACK geqrf workspace
    orm_ws::QROrmWs{T}    # LAPACK ormqr workspace (shares τ with qr_ws)
    Σ_uu_work::Matrix{T}  # (K, K) — scratch for factorization
    Σ_um_work::Vector{T}  # (K,)   — scratch for `Σ_uu \ Σ_um`
end

function ProxySvarBuffers{T}(TT::Int, KK::Int, d::Int) where {T <: Union{Float32, Float64}}
    TT >= d || throw(
        ArgumentError("need at least as many observations as regressors; got TT=$TT, d=$d")
    )
    X_qr = Matrix{T}(undef, TT, d)
    Y_qr = Matrix{T}(undef, TT, KK)
    qr_ws = QRWs(X_qr)
    orm_ws = QROrmWs(qr_ws, 'L', 'T', X_qr, Y_qr)
    return ProxySvarBuffers{T}(
        Matrix{T}(undef, d, KK),
        Matrix{T}(undef, KK, d),
        Matrix{T}(undef, TT, KK),
        Matrix{T}(undef, KK, KK),
        Vector{T}(undef, KK),
        Vector{T}(undef, KK),
        X_qr,
        Y_qr,
        qr_ws,
        orm_ws,
        Matrix{T}(undef, KK, KK),
        Vector{T}(undef, KK)
    )
end

"""
    estimate_proxy_svar!(buf, yy, xx, mm) -> (A_est, U_est, Σ_uu, Σ_um, H1)

In-place proxy-SVAR estimation using pre-allocated `buf::ProxySvarBuffers`.
The OLS step uses a Householder QR of `xx` (no normal equations), so accuracy
does not degrade with the squared condition number of `xx`.
Returns views/aliases into `buf` — do not retain across bootstrap iterations.
Throws `SingularException` when `xx` is numerically rank deficient (a diagonal
entry of its R factor is below `max(TT, d) * eps * max|R[i, i]|`) and
`PosDefException` when the residual covariance is not positive definite.
"""
function estimate_proxy_svar!(
        buf::ProxySvarBuffers{T},
        yy::AbstractMatrix{T}, xx::AbstractMatrix{T},
        mm::AbstractVector{T}
    ) where {T}
    Base.require_one_based_indexing(yy, xx, mm)
    size(xx) == size(buf.X_qr) && size(yy) == size(buf.Y_qr) && length(mm) == size(yy, 1) ||
        throw(
        DimensionMismatch(
            "buffers sized for xx $(size(buf.X_qr)), yy $(size(buf.Y_qr)); got xx $(size(xx)), yy $(size(yy)), mm of length $(length(mm))"
        )
    )
    TT, KK = size(yy)
    d = size(xx, 2)

    # OLS by QR: X = QR  ⇒  A_raw = R \ (Q'Y)[1:d, :]
    copyto!(buf.X_qr, xx)
    copyto!(buf.Y_qr, yy)
    LAPACK.geqrf!(buf.qr_ws, buf.X_qr)
    LAPACK.ormqr!(buf.orm_ws, 'L', 'T', buf.X_qr, buf.Y_qr)
    R = view(buf.X_qr, 1:d, 1:d)
    # Numerically rank-deficient X: a pivot this small turns into coefficients
    # of order 1/eps instead of an error.
    tol = max(TT, d) * eps(T) * maximum(i -> abs(R[i, i]), 1:d)
    for i in 1:d
        abs(R[i, i]) > tol || throw(SingularException(i))
    end
    QtY = view(buf.Y_qr, 1:d, :)
    LAPACK.trtrs!('U', 'N', 'N', R, QtY)
    copyto!(buf.A_raw, QtY)

    # Residuals: U = Y - X * A_raw
    copyto!(buf.U_est, yy)
    mul!(buf.U_est, xx, buf.A_raw, -one(T), one(T))

    _proxy_impact!(buf, mm)

    # A_est = A_raw' (row = equation)
    transpose!(buf.A_est, buf.A_raw)

    return buf.A_est, buf.U_est, buf.Σ_uu, buf.Σ_um, buf.H1
end

# Identification step on the residuals in `buf.U_est`: fills `buf.Σ_uu`,
# `buf.Σ_um`, and `buf.H1`. Throws `PosDefException` when `Σ_uu` is not positive
# definite.
function _proxy_impact!(buf::ProxySvarBuffers{T}, mm::AbstractVector{T}) where {T}
    TT, KK = size(buf.U_est)
    length(mm) == TT ||
        throw(DimensionMismatch("proxy has length $(length(mm)); residuals have $TT rows"))

    # Σ_uu = U'U / T ; Σ_um = U'm / T
    mul!(buf.Σ_uu, buf.U_est', buf.U_est)
    buf.Σ_uu ./= TT
    mul!(buf.Σ_um, buf.U_est', mm)
    buf.Σ_um ./= TT

    # φ² = Σ_um' * (Σ_uu \ Σ_um) ; H1 = Σ_um / √φ²
    copyto!(buf.Σ_uu_work, buf.Σ_uu)
    copyto!(buf.Σ_um_work, buf.Σ_um)
    Fuu = cholesky!(Symmetric(buf.Σ_uu_work))
    ldiv!(Fuu, buf.Σ_um_work)          # Σ_um_work = Σ_uu \ Σ_um
    φ² = dot(buf.Σ_um, buf.Σ_um_work)
    φ = sqrt(φ²)
    for k in 1:KK
        buf.H1[k] = buf.Σ_um[k] / φ
    end
    return buf
end

# ============================================================================
# Core MBB function
# ============================================================================

"""
    proxy_svar_mbb(model::VARModel, id::IVIdentification, horizon, inf::ProxySVARMBB;
                   coverage=[0.68, 0.90, 0.95], rng=Random.default_rng(),
                   block_indices=nothing) -> NamedTuple

Jentsch & Lunsford (2022) corrected moving block bootstrap for SVAR-IV.

Jointly resamples (residuals, instrument) in overlapping blocks with position-specific
centering of the residuals, simulates the VAR from the resampled residuals, and
re-estimates the proxy-SVAR on each simulated sample. Produces percentile intervals,
Hall's intervals, and (when `inf.compute_ar` is true) Anderson-Rubin confidence sets,
each at every level in `coverage`.

A replication fails when its re-estimation raises `PosDefException`,
`SingularException`, `LAPACKException`, or `DomainError`, or produces non-finite
values; other exceptions propagate. Failed replications are dropped (with a warning)
from every statistic and from the returned draws; `n_failed` counts them. The call
throws when more than `inf.max_failure_rate` of the replications fail.

When `inf.compute_ar == true` and `inf.ar_grid === nothing`, a default grid of
`range(-10, 10; length=201)` is used.

# Arguments
- `model`: Estimated VAR model
- `id`: `IVIdentification` carrying the external instrument, which must be observed in
  every residual period
- `horizon`: IRF horizon (positive)
- `inf`: `ProxySVARMBB` inference specification

# Keyword Arguments
- `coverage`: Coverage levels in (0, 1); stored sorted in the result.
- `rng`: Random number generator for the block draws.
- `block_indices`: `nothing`, or an integer matrix of size
  `(inf.reps, cld(T, inf.block_length))` of 1-based block starts, where `T` is the
  number of residual periods. Row `b` lists the blocks of replication `b` and is used
  in place of draws from `rng`, so a run recorded elsewhere (e.g. by another
  implementation) can be reproduced exactly.

# Returns
A `NamedTuple`. Every matrix is `(variable, horizon)` with `horizon + 1` columns
(impact first) and describes the response to the instrumented shock. Each
`(lower, upper)` pair holds vectors of matrices aligned with `coverage`.
- `coverage::Vector{Float64}`
- `point_irf`, `point_irf_norm`, `point_fevd`, `point_svma`: point estimates;
  `point_irf_norm` is scaled so that the impact on the target variable equals
  `inf.norm_scale`.
- `irf_bands`, `irf_norm_bands`, `fevd_bands`: percentile intervals `(lower, upper)`.
  At level `α`, the bounds are the `round(q * n) + 1`-th smallest of the `n` successful
  draws for `q = (1 - α) / 2` and `q = (1 + α) / 2`.
- `halls_irf_norm_bands`: Hall's intervals for the normalized IRF `(lower, upper)`,
  `2 * point_irf_norm .- upper` and `2 * point_irf_norm .- lower` of the percentile
  intervals.
- `ar`: `nothing`, or `(grid, rates, included)`. `rates[g, k, h]` is the fraction of
  draws whose Anderson-Rubin statistic for value `grid[g]` of the normalized response
  of variable `k` at horizon `h - 1` is ≤ 0; `included[i]` is a `BitArray` of the same
  size marking the grid values inside the confidence set at level `coverage[i]`,
  i.e. those with `(1 - α) / 2 ≤ rate ≤ (1 + α) / 2`.
- `irf_draws`, `irf_norm_draws`: successful draws, `(draw, variable, horizon)`.
- `n_failed`: number of failed replications.
"""
function proxy_svar_mbb(
        model::VARModel{T}, id::IVIdentification,
        horizon::Int, inf::ProxySVARMBB;
        coverage::AbstractVector{<:Real} = [0.68, 0.9, 0.95],
        rng::AbstractRNG = Random.default_rng(),
        block_indices::Union{Nothing, AbstractMatrix{<:Integer}} = nothing
    ) where {T}
    coverage = collect(Float64, _validate_irf_args(horizon, coverage))
    resolved = _resolve_iv(model, id)
    ν = model.residuals
    TT = size(ν, 1)
    Z, target = _extract_instrument(resolved.instrument, TT, n_lags(model), model.names)
    proxy = vec(_require_observed(Z, "ProxySVARMBB"))

    # Default AR grid when compute_ar=true but no grid specified
    inf_resolved = if inf.compute_ar && inf.ar_grid === nothing
        ProxySVARMBB(
            inf.reps; block_length = inf.block_length,
            compute_ar = true, ar_grid = collect(range(-10.0, 10.0; length = 201)),
            norm_scale = inf.norm_scale, save_draws = inf.save_draws,
            max_failure_rate = inf.max_failure_rate
        )
    else
        inf
    end

    return _proxy_svar_mbb_impl(
        model, proxy, horizon, inf_resolved, target, coverage; rng, block_indices
    )
end

# Deprecated `IVSVAR` models carry the instrument in the spec.
function proxy_svar_mbb(
        model::VARModel{T, <:IVSVAR},
        horizon::Int, inf::ProxySVARMBB;
        coverage::AbstractVector{<:Real} = [0.68, 0.9, 0.95],
        rng::AbstractRNG = Random.default_rng(),
        block_indices::Union{Nothing, AbstractMatrix{<:Integer}} = nothing
    ) where {T}
    return proxy_svar_mbb(
        model, IVIdentification(model.spec.instrument),
        horizon, inf; coverage, rng, block_indices
    )
end

function _proxy_svar_mbb_impl(
        model::VARModel{T}, proxy::Vector{T},
        horizon::Int, inf::ProxySVARMBB, target::Int, coverage::Vector{Float64};
        rng::AbstractRNG = Random.default_rng(),
        block_indices::Union{Nothing, AbstractMatrix{<:Integer}} = nothing
    ) where {T}
    ν = model.residuals
    TT, KK = size(ν)
    p_val = n_lags(model)
    ℓ = inf.block_length
    n_boot = inf.reps
    n_imp = horizon + 1
    s = inf.norm_scale

    length(proxy) == TT ||
        throw(DimensionMismatch("Proxy length ($(length(proxy))) must match residuals ($TT)"))
    _check_block_length(ℓ, TT)

    # ── Build A_est in Python format: (K, 1+K*p) ─────────────────────────
    A_est = Matrix(transpose(coef(model)))

    # Transpose for simulation: (1+K*p, K)
    A_sim = A_est'

    # Initial conditions: [1, y_{p}, y_{p-1}, ..., y_{1}]
    y_init = zeros(T, 1 + KK * p_val)
    y_init[1] = one(T)
    for lag in 1:p_val
        y_init[(1 + (lag - 1) * KK + 1):(1 + lag * KK)] .= model.Y[p_val + 1 - lag, :]
    end

    # ── Step 1: Form overlapping blocks ───────────────────────────────────
    n_blocks = TT - ℓ + 1
    u_blocks = zeros(T, ℓ, KK, n_blocks)
    m_blocks = zeros(T, ℓ, n_blocks)

    for b in 1:n_blocks
        u_blocks[:, :, b] .= ν[b:(b + ℓ - 1), :]
        m_blocks[:, b] .= proxy[b:(b + ℓ - 1)]
    end

    # ── Step 2: J&L position-specific centering ───────────────────────────
    # Residuals only. Centering exists to keep the simulated VAR from drifting,
    # and the proxy is not fed through that recursion — it enters only through
    # `Σ_um`, whose mean is part of the moment being bootstrapped.
    # For position s (1-indexed), mean over all blocks at that position.
    u_center_block = zeros(T, ℓ, KK)

    for s in 1:ℓ
        # All residual values at position s across overlapping blocks
        # = ν[s], ν[s+1], ..., ν[s + n_blocks - 1] = ν[s:(TT - ℓ + s)]
        u_center_block[s, :] .= vec(mean(ν[s:(TT - ℓ + s), :]; dims = 1))
    end

    # Tile centering to full resampled length
    n_resample = cld(TT, ℓ)  # ceil(TT / ℓ)
    u_center = zeros(T, n_resample * ℓ, KK)
    for j in 1:n_resample
        u_center[((j - 1) * ℓ + 1):(j * ℓ), :] .= u_center_block
    end

    if block_indices !== nothing
        size(block_indices) == (n_boot, n_resample) || throw(
            DimensionMismatch(
                "block_indices must be ($n_boot, $n_resample), got $(size(block_indices))"
            )
        )
        all(i -> 1 <= i <= n_blocks, block_indices) || throw(
            ArgumentError(
                "block_indices must all lie in 1:$n_blocks"
            )
        )
    end

    # ── Step 3: Storage for bootstrap results (draw, variable, horizon) ──
    irf_store = zeros(T, n_boot, KK, n_imp)
    irf_norm_store = zeros(T, n_boot, KK, n_imp)
    fevd_store = zeros(T, n_boot, KK, n_imp)

    # AR statistics: (grid, variable, horizon, draw)
    ar_store = if inf.compute_ar
        zeros(T, length(inf.ar_grid), KK, n_imp, n_boot)
    else
        nothing
    end

    # Pre-allocate reusable buffers outside the loop
    dim_state = 1 + KK * p_val
    u_temp = zeros(T, n_resample * ℓ, KK)
    m_temp = zeros(T, n_resample * ℓ)
    x_star = zeros(T, TT, dim_state)
    y_star = Matrix{T}(undef, TT, KK)
    xt_buf = Vector{T}(undef, KK)

    # Workspace for the inner estimator and dynamics (reused across draws)
    est_buf = ProxySvarBuffers{T}(TT, KK, dim_state)
    dyn_buf = ProxyDynamicsBuffers{T}(KK, p_val, n_imp)

    # ── Point estimate: the model's coefficients and residuals ───────────
    copyto!(est_buf.U_est, ν)
    _proxy_impact!(est_buf, proxy)
    dyn_point = proxy_svar_dynamics(
        A_est, est_buf.Σ_uu, est_buf.Σ_um, est_buf.H1,
        p_val, s, n_imp, target
    )

    # ── Step 4: Bootstrap loop ────────────────────────────────────────────
    ok = falses(n_boot)

    for b in 1:n_boot
        # 4a: Resample blocks with replacement
        for j in 1:n_resample
            idx = block_indices === nothing ? rand(rng, 1:n_blocks) :
                block_indices[b, j]
            blk_start = (j - 1) * ℓ + 1
            @inbounds for s in 1:ℓ, k in 1:KK

                u_temp[blk_start + s - 1, k] = u_blocks[s, k, idx]
            end
            @inbounds for s in 1:ℓ
                m_temp[blk_start + s - 1] = m_blocks[s, idx]
            end
        end

        # 4b: Apply J&L centering (residuals only)
        u_temp .-= u_center

        # 4c: Trim to T observations (views avoid allocation)
        u_star_v = view(u_temp, 1:TT, :)
        m_star = view(m_temp, 1:TT)

        # 4d: Simulate bootstrap VAR
        # y_star[t] = A_sim' * x_star[t] + u_star[t]
        x_star[1, :] .= y_init
        copyto!(y_star, u_star_v)

        for t in 1:TT
            # y_star[t, :] += A_sim' * x_star[t, :]
            mul!(xt_buf, A_sim', view(x_star, t, :))
            @inbounds for k in 1:KK
                y_star[t, k] += xt_buf[k]
            end

            # Update state for next period
            if t < TT
                x_star[t + 1, 1] = one(T)
                @inbounds for k in 1:KK
                    x_star[t + 1, 1 + k] = y_star[t, k]
                end
                if p_val > 1
                    @inbounds for k in 1:(KK * (p_val - 1))
                        x_star[t + 1, KK + 1 + k] = x_star[t, 1 + k]
                    end
                end
            end
        end

        # 4e: Re-estimate proxy-SVAR on bootstrap sample (in-place)
        ok[b] = _try_draw!(
            view(irf_store, b, :, :), view(irf_norm_store, b, :, :),
            view(fevd_store, b, :, :)
        ) do irf_b, irf_norm_b, fevd_b
            A_star, U_star, Σ_uu_star,
                Σ_um_star, H1_star = estimate_proxy_svar!(est_buf, y_star, x_star, m_star)

            # 4f: Compute bootstrap dynamics (in-place)
            proxy_svar_dynamics!(
                dyn_buf, A_star, Σ_uu_star, Σ_um_star, H1_star,
                p_val, s, n_imp, target
            )

            copyto!(irf_b, dyn_buf.irf)
            copyto!(irf_norm_b, dyn_buf.irf_norm)
            copyto!(fevd_b, dyn_buf.fevd)
        end

        # 4g: AR statistics — anchor on target variable's impact
        if ok[b] && inf.compute_ar
            svma_star = dyn_buf.svma
            svma_target = svma_star[target, 1]
            for h in 1:n_imp, k in 1:KK, g in eachindex(inf.ar_grid)
                ar_store[g, k, h, b] = s * svma_star[k, h] - svma_target * inf.ar_grid[g]
            end
            ok[b] = all(isfinite, view(ar_store, :, :, :, b))
        end
    end

    # Failed draws are dropped from every store before any statistic is computed.
    n_failed = count(!, ok)
    _check_failure_rate("Proxy-SVAR MBB", n_failed, n_boot, inf.max_failure_rate)
    if n_failed > 0
        irf_store = irf_store[ok, :, :]
        irf_norm_store = irf_norm_store[ok, :, :]
        fevd_store = fevd_store[ok, :, :]
        ar_store = ar_store === nothing ? nothing : ar_store[:, :, :, ok]
    end

    # ── Step 5: Percentile intervals ──────────────────────────────────────
    irf_norm_bands = _percentile_intervals(irf_norm_store, coverage)

    # ── Step 6: AR confidence sets ────────────────────────────────────────
    ar_result = if inf.compute_ar
        _ar_confidence_sets(ar_store, inf.ar_grid, s, target, coverage)
    else
        nothing
    end

    return (
        coverage = coverage,
        point_irf = dyn_point.irf,
        point_irf_norm = dyn_point.irf_norm,
        point_fevd = dyn_point.fevd,
        point_svma = dyn_point.svma,
        irf_bands = _percentile_intervals(irf_store, coverage),
        irf_norm_bands = irf_norm_bands,
        fevd_bands = _percentile_intervals(fevd_store, coverage),
        halls_irf_norm_bands = _halls_intervals(irf_norm_bands, dyn_point.irf_norm),
        ar = ar_result,
        irf_draws = irf_store,
        irf_norm_draws = irf_norm_store,
        n_failed = n_failed,
    )
end

# Tail probabilities `(1 - α) / 2` and `(1 + α) / 2` of a two-sided interval with
# coverage `α`, rounded to 12 significant digits so that a decimal level gives its
# decimal tails (`α = 0.95` gives `0.025`, not `0.025000000000000022`). Percentile
# ranks and AR inclusion are step functions of the tails, so the representation
# error of `1 - α` would otherwise move them at exact ties.
_interval_tails(α::Real) = (round((1 - α) / 2; sigdigits = 12), round((1 + α) / 2; sigdigits = 12))

# Rank of the `q` quantile among `n` sorted draws. The reference implementation
# indexes the sorted draws 0-based at `round(q * n)`, i.e. the `round(q * n) + 1`-th
# smallest.
_percentile_rank(q::Real, n::Integer) = clamp(round(Int, q * n) + 1, 1, n)

"""
    _percentile_intervals(store, coverage) -> (lower, upper)

Percentile intervals from draws `store` of size `(draw, variable, horizon)`, one
`(variable, horizon)` matrix per level in `coverage`.
"""
function _percentile_intervals(store::AbstractArray{T, 3}, coverage::AbstractVector{<:Real}) where {T}
    n = size(store, 1)
    sorted = sort(store; dims = 1)
    lower = Vector{Matrix{T}}(undef, length(coverage))
    upper = Vector{Matrix{T}}(undef, length(coverage))
    for (i, α) in enumerate(coverage)
        q_lo, q_hi = _interval_tails(α)
        lower[i] = sorted[_percentile_rank(q_lo, n), :, :]
        upper[i] = sorted[_percentile_rank(q_hi, n), :, :]
    end
    return (lower = lower, upper = upper)
end

"""
    _halls_intervals(bands, point) -> (lower, upper)

Hall's intervals from percentile intervals `bands`:
`lower = 2 * point - upper_percentile`, `upper = 2 * point - lower_percentile`.
"""
function _halls_intervals(bands::NamedTuple, point::AbstractMatrix)
    return (
        lower = [2 .* point .- u for u in bands.upper],
        upper = [2 .* point .- l for l in bands.lower],
    )
end

"""
    _ar_confidence_sets(ar_store, grid, scale, target, coverage) -> (grid, rates, included)

Anderson-Rubin confidence sets from bootstrap statistics `ar_store` of size
`(grid, variable, horizon, draw)`. `rates[g, k, h]` is the fraction of draws with
statistic ≤ 0; at level `α`, `grid[g]` is included when the rate lies in
`[(1 - α) / 2, (1 + α) / 2]`. The target variable's impact equals `scale` by
construction, so grid values `≈ scale` are always included at `(target, impact)`.
"""
function _ar_confidence_sets(
        ar_store::AbstractArray{T, 4}, grid::AbstractVector{<:Real}, scale::Real,
        target::Integer, coverage::AbstractVector{<:Real}
    ) where {T}
    n = size(ar_store, 4)
    rates = dropdims(count(<=(0), ar_store; dims = 4); dims = 4) ./ n
    at_scale = findall(g -> g ≈ scale, grid)
    included = map(coverage) do α
        q_lo, q_hi = _interval_tails(α)
        inc = q_lo .<= rates .<= q_hi
        inc[at_scale, target, 1] .= true
        inc
    end
    return (grid = grid, rates = rates, included = included)
end

# ============================================================================
# Integration with irf() dispatch
# ============================================================================

"""
    compute_inference_bands(model, id::IVIdentification, irf_point,
        inf::ProxySVARMBB, coverage, normalization, rng)

Dispatch for Jentsch-Lunsford MBB on SVAR-IV models. Works with any VARModel
(not just IVSVAR) as long as the identification carries an instrument.

When `inf.compute_ar == true`, the bands at each level in `coverage` are the
smallest and largest grid values in the Anderson-Rubin confidence set at that level
(weak-instrument robust). Otherwise, percentile bands are computed from the draws.
"""
function compute_inference_bands(
        model::VARModel{T},
        identification::IVIdentification,
        irf_point::Array{T, 3},
        inf::ProxySVARMBB,
        coverage::Vector{Float64},
        ::AbstractNormalization,  # ProxySVARMBB uses its own norm_scale
        rng::AbstractRNG
    ) where {T}
    horizon = size(irf_point, 3) - 1
    id = _resolve_iv(model, identification)
    _,
        target = _extract_instrument(
        id.instrument, size(model.residuals, 1),
        n_lags(model), model.names
    )

    mbb = proxy_svar_mbb(model, id, horizon, inf; coverage, rng)
    mbb.coverage == coverage ||
        throw(ArgumentError("coverage must be sorted, got $coverage"))

    K = n_vars(model)
    n_imp = horizon + 1

    # Draws of the identified shock only. Each draw is normalized by *its own*
    # impact on the target variable (J&L 2022), which fixes that impact at
    # `inf.norm_scale`; using the point-estimate impact instead would leave the
    # draws' own scale varying. `norm_factor` puts the draws back on the scale of
    # `irf_point` (which has the identified column at position `target`).
    target_impact = irf_point[target, target, 1]
    norm_factor = target_impact / inf.norm_scale
    irf_norm_draws = mbb.irf_norm_draws  # (draw, variable, horizon)
    n_boot = size(irf_norm_draws, 1)

    draws_4d = zeros(T, n_boot, K, K, n_imp)
    for h in 1:n_imp, k in 1:K
        draws_4d[:, k, target, h] .= view(irf_norm_draws, :, k, h) .* norm_factor
    end
    # Non-identified columns: fill with point estimate (no variation)
    for j in 1:K
        j == target && continue
        for h in 1:n_imp, k in 1:K

            draws_4d[:, k, j, h] .= irf_point[k, j, h]
        end
    end

    stderr = dropdims(std(draws_4d; dims = 1); dims = 1)

    lower = Vector{Array{T, 3}}(undef, length(coverage))
    upper = Vector{Array{T, 3}}(undef, length(coverage))

    for (i, α) in enumerate(coverage)
        lb = zeros(T, K, K, n_imp)
        ub = zeros(T, K, K, n_imp)

        if inf.compute_ar
            # AR confidence sets: convex hull [min, max] of included grid points.
            # Conservative but plottable; the full sets are in `mbb.ar.included`.
            grid = mbb.ar.grid
            for h in 1:n_imp, k in 1:K

                included = view(mbb.ar.included[i], :, k, h)
                if any(included)
                    # `grid` is already on the normalized scale (impact on the
                    # target variable equals `inf.norm_scale`), so it only needs
                    # the same rescaling as the draws. `norm_factor` can be
                    # negative, so order the endpoints after rescaling.
                    g_lo = minimum(grid[included]) * norm_factor
                    g_hi = maximum(grid[included]) * norm_factor
                    lb[k, target, h] = min(g_lo, g_hi)
                    ub[k, target, h] = max(g_lo, g_hi)
                else
                    lb[k, target, h] = T(NaN)
                    ub[k, target, h] = T(NaN)
                end
            end
        else
            # Percentile CIs from draws (consistent with cumulation pipeline)
            lb_raw, ub_raw = compute_bands_from_draws(irf_point, draws_4d, [α])
            lb .= lb_raw[1]
            ub .= ub_raw[1]
        end

        lower[i] = lb
        upper[i] = ub
    end

    return draws_4d, stderr, lower, upper
end

# The residual bootstraps refit the VAR on a resampled series without resampling
# the instrument, so the instrument has no valid alignment with the refitted residuals.
function _check_resamples_instrument(::IVIdentification, inference::_ResidualBootstrap)
    throw(
        ArgumentError(
            "$(nameof(typeof(inference))) does not resample the instrument and is not " *
                "supported with IVIdentification. Use inference = ProxySVARMBB(reps), " *
                "which resamples residuals and instrument jointly."
        )
    )
end

# Error guard: ProxySVARMBB requires IVIdentification
function compute_inference_bands(
        ::VARModel{T},
        identification::AbstractIdentification,
        ::Array{T, 3},
        ::ProxySVARMBB,
        ::Vector{Float64},
        ::AbstractNormalization,
        ::AbstractRNG
    ) where {T}
    throw(
        ArgumentError(
            "ProxySVARMBB inference requires IVIdentification. " *
                "Use IVIdentification(Z, target_shock) or a different inference method."
        )
    )
end
