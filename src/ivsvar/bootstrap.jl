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
function proxy_svar_dynamics(A_est::AbstractMatrix{T}, Σ_uu::AbstractMatrix{T},
        Σ_um::AbstractVector{T}, H1::AbstractVector{T},
        p::Int, norm_scale::Float64, n_imp::Int, target::Int) where {T}
    K = length(H1)

    # Companion matrix
    comp = zeros(T, K * p, K * p)
    if p == 1
        comp[:, :] .= A_est[:, 2:(K + 1)]
    else
        comp[1:K, :] .= A_est[:, 2:(K * p + 1)]
        comp[(K + 1):(K * p), 1:(K * (p - 1))] .= I(K * (p - 1))
    end

    # Storage
    irf_out = zeros(T, K, n_imp)
    irf_norm = zeros(T, K, n_imp)
    fevd = zeros(T, K, n_imp)
    svma = zeros(T, K, n_imp)

    # Horizon 0
    irf_out[:, 1] .= H1
    for k in 1:K
        fevd[k, 1] = H1[k]^2 / Σ_uu[k, k]
    end
    svma[:, 1] .= Σ_um

    # Normalized IRF at horizon 0 (normalize by impact on target variable)
    inv_H1_t = norm_scale / H1[target]
    for k in 1:K
        irf_norm[k, 1] = inv_H1_t * H1[k]
    end

    # Higher horizons
    Kp = K * p
    comp_power = Matrix{T}(I, Kp, Kp)
    comp_power_buf = similar(comp_power)
    numer = zeros(T, K)
    @inbounds for k in 1:K
        numer[k] = H1[k]^2
    end
    denom = copy(Σ_uu)

    # Pre-allocate temporaries
    irf_h = Vector{T}(undef, K)
    Φ_Σ = Matrix{T}(undef, K, K)

    for h in 2:n_imp
        mul!(comp_power_buf, comp_power, comp)
        comp_power, comp_power_buf = comp_power_buf, comp_power
        Φ_h = view(comp_power, 1:K, 1:K)

        # IRF
        mul!(irf_h, Φ_h, H1)
        irf_out[:, h] .= irf_h

        # Normalized IRF
        @inbounds for k in 1:K
            irf_norm[k, h] = inv_H1_t * irf_h[k]
        end

        # FEVD
        @inbounds for k in 1:K
            numer[k] += irf_h[k]^2
        end
        mul!(Φ_Σ, Φ_h, Σ_uu)
        # denom += Φ_h * Σ_uu * Φ_h'
        mul!(denom, Φ_Σ, Φ_h', one(T), one(T))
        @inbounds for k in 1:K
            fevd[k, h] = numer[k] / denom[k, k]
        end

        # Structural VMA
        mul!(view(svma, :, h), Φ_h, Σ_um)
    end

    return (irf = irf_out, irf_norm = irf_norm, fevd = fevd, svma = svma)
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
    ProxyDynamicsBuffers{T}(
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
function proxy_svar_dynamics!(buf::ProxyDynamicsBuffers{T},
        A_est::AbstractMatrix{T}, Σ_uu::AbstractMatrix{T},
        Σ_um::AbstractVector{T}, H1::AbstractVector{T},
        p::Int, norm_scale::Float64, n_imp::Int, target::Int) where {T}
    K = length(H1)
    Kp = K * p

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

Returns: (A_est, U_est, Σ_uu, Σ_um, H1)
- A_est: (K, 1+K*p) coefficient matrix (row = equation)
- U_est: (T_eff, K) residuals
- Σ_uu: (K, K) residual covariance
- Σ_um: (K,) covariance with proxy
- H1: (K,) identified impact column
"""
function estimate_proxy_svar(yy::AbstractMatrix{T},
        xx::AbstractMatrix{T},
        mm::AbstractVector{T}) where {T}
    TT, KK = size(yy)

    # OLS: A = (X'X)⁻¹ X'Y
    A_est_raw = xx \ yy  # (1+K*p, K)

    # Residuals
    U_est = yy - xx * A_est_raw

    # Covariance matrices (no df correction)
    Σ_uu = (U_est' * U_est) ./ TT
    Σ_um = vec((U_est' * mm) ./ TT)

    # Identification: H1 = Σ_um / φ
    φ² = dot(Σ_um, Σ_uu \ Σ_um)
    φ = sqrt(φ²)
    H1 = Σ_um ./ φ

    # Transpose A to (K, 1+K*p) — row = equation
    A_est = A_est_raw'

    return A_est, U_est, Σ_uu, Σ_um, H1
end

"""
    ProxySvarBuffers{T}

Pre-allocated workspace for `estimate_proxy_svar!`. Reused across bootstrap
draws to avoid per-draw allocations.
"""
struct ProxySvarBuffers{T}
    A_raw::Matrix{T}      # (d, K)  — raw coefficients (d = 1 + K*p)
    A_est::Matrix{T}      # (K, d)  — transposed
    U_est::Matrix{T}      # (TT, K) — residuals
    Σ_uu::Matrix{T}       # (K, K)
    Σ_um::Vector{T}       # (K,)
    H1::Vector{T}         # (K,)
    XtX::Matrix{T}        # (d, d) — normal equations LHS
    XtY::Matrix{T}        # (d, K) — normal equations RHS
    Σ_uu_work::Matrix{T}  # (K, K) — scratch for factorization
    Σ_um_work::Vector{T}  # (K,)   — scratch for `Σ_uu \ Σ_um`
end

function ProxySvarBuffers{T}(TT::Int, KK::Int, d::Int) where {T}
    ProxySvarBuffers{T}(
        Matrix{T}(undef, d, KK),
        Matrix{T}(undef, KK, d),
        Matrix{T}(undef, TT, KK),
        Matrix{T}(undef, KK, KK),
        Vector{T}(undef, KK),
        Vector{T}(undef, KK),
        Matrix{T}(undef, d, d),
        Matrix{T}(undef, d, KK),
        Matrix{T}(undef, KK, KK),
        Vector{T}(undef, KK)
    )
end

"""
    estimate_proxy_svar!(buf, yy, xx, mm) -> (A_est, U_est, Σ_uu, Σ_um, H1)

In-place proxy-SVAR estimation using pre-allocated `buf::ProxySvarBuffers`.
Returns views/aliases into `buf` — do not retain across bootstrap iterations.
Throws `PosDefException` or `SingularException` on rank-deficient inputs; the
caller is expected to catch these and record a failed draw.
"""
function estimate_proxy_svar!(buf::ProxySvarBuffers{T},
        yy::AbstractMatrix{T}, xx::AbstractMatrix{T},
        mm::AbstractVector{T}) where {T}
    TT, KK = size(yy)
    d = size(xx, 2)

    # Normal equations: (X'X) A = X'Y  →  solve via Cholesky (X'X is SPD in
    # non-degenerate bootstrap samples; rank deficiency throws → caught upstream).
    mul!(buf.XtX, xx', xx)
    mul!(buf.XtY, xx', yy)
    F = cholesky!(Symmetric(buf.XtX))  # in-place factorization
    copyto!(buf.A_raw, buf.XtY)
    ldiv!(F, buf.A_raw)                # A_raw = (X'X) \ (X'Y)

    # Residuals: U = Y - X * A_raw
    copyto!(buf.U_est, yy)
    mul!(buf.U_est, xx, buf.A_raw, -one(T), one(T))

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
    @inbounds for k in 1:KK
        buf.H1[k] = buf.Σ_um[k] / φ
    end

    # A_est = A_raw' (row = equation)
    transpose!(buf.A_est, buf.A_raw)

    return buf.A_est, buf.U_est, buf.Σ_uu, buf.Σ_um, buf.H1
end

# ============================================================================
# Core MBB function
# ============================================================================

"""
    proxy_svar_mbb(model::VARModel, id::IVIdentification, horizon, inf::ProxySVARMBB;
                   rng) -> NamedTuple

Jentsch & Lunsford (2022) corrected moving block bootstrap for SVAR-IV.

Jointly resamples (residuals, instrument) in overlapping blocks with position-specific
centering. Produces percentile CIs, Hall's CIs, and optionally Anderson-Rubin
confidence sets.

When `inf.compute_ar == true` and `inf.ar_grid === nothing`, a default grid of
`range(-10, 10; length=201)` is used.

# Arguments
- `model`: Estimated VAR model (any spec — does not require IVSVAR)
- `id`: `IVIdentification` carrying the external instrument
- `horizon`: IRF horizon
- `inf`: `ProxySVARMBB` inference specification
- `rng`: Random number generator for reproducibility
"""
function proxy_svar_mbb(model::VARModel{T}, id::IVIdentification,
        horizon::Int, inf::ProxySVARMBB;
        rng::AbstractRNG = Random.default_rng()) where {T}
    resolved = _resolve_iv(model, id)
    ν = model.residuals
    TT = size(ν, 1)
    Z, target = _extract_instrument(resolved.instrument, TT, n_lags(model), model.names)
    proxy = vec(Z)

    # Default AR grid when compute_ar=true but no grid specified
    inf_resolved = if inf.compute_ar && inf.ar_grid === nothing
        ProxySVARMBB(inf.reps; block_length = inf.block_length,
            compute_ar = true, ar_grid = collect(range(-10.0, 10.0; length = 201)),
            norm_scale = inf.norm_scale, save_draws = inf.save_draws)
    else
        inf
    end

    return _proxy_svar_mbb_impl(model, proxy, horizon, inf_resolved, target; rng = rng)
end

# Backward compat: instrument in model
function proxy_svar_mbb(model::VARModel{T, <:IVSVAR},
        horizon::Int, inf::ProxySVARMBB;
        rng::AbstractRNG = Random.default_rng()) where {T}
    return proxy_svar_mbb(model, IVIdentification(model.spec.instrument),
        horizon, inf; rng = rng)
end

"""
Internal implementation of the Jentsch-Lunsford MBB. Called by `proxy_svar_mbb`.
"""
function _proxy_svar_mbb_impl(model::VARModel{T}, proxy::Vector{T},
        horizon::Int, inf::ProxySVARMBB, target::Int;
        rng::AbstractRNG = Random.default_rng()) where {T}
    ν = model.residuals
    TT, KK = size(ν)
    p_val = n_lags(model)
    ℓ = inf.block_length
    n_boot = inf.reps
    n_imp = horizon + 1
    s = inf.norm_scale

    length(proxy) == TT ||
        error("Proxy length ($(length(proxy))) must match residuals ($TT)")

    # ── Build A_est in Python format: (K, 1+K*p) ─────────────────────────
    coefs = coef(model)
    A_est = zeros(T, KK, 1 + KK * p_val)
    A_est[:, 1] .= coefs.intercept
    for lag in 1:p_val
        A_est[:, (1 + (lag - 1) * KK + 1):(1 + lag * KK)] .= coefs.lags[:, :, lag]
    end

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
    # For position s (1-indexed), mean over all blocks at that position
    u_center_block = zeros(T, ℓ, KK)
    m_center_block = zeros(T, ℓ)

    for s in 1:ℓ
        # All residual values at position s across overlapping blocks
        # = ν[s], ν[s+1], ..., ν[s + n_blocks - 1] = ν[s:(TT - ℓ + s)]
        u_center_block[s, :] .= vec(mean(ν[s:(TT - ℓ + s), :]; dims = 1))
        m_center_block[s] = mean(proxy[s:(TT - ℓ + s)])
    end

    # Tile centering to full resampled length
    n_resample = cld(TT, ℓ)  # ceil(TT / ℓ)
    u_center = zeros(T, n_resample * ℓ, KK)
    m_center = zeros(T, n_resample * ℓ)
    for j in 1:n_resample
        u_center[((j - 1) * ℓ + 1):(j * ℓ), :] .= u_center_block
        m_center[((j - 1) * ℓ + 1):(j * ℓ)] .= m_center_block
    end

    # ── Step 3: Storage for bootstrap results ─────────────────────────────
    irf_store = zeros(T, KK, n_imp, n_boot)
    irf_norm_store = zeros(T, KK, n_imp, n_boot)
    fevd_store = zeros(T, KK, n_imp, n_boot)

    ar_store = if inf.compute_ar
        n_grid = length(inf.ar_grid)
        zeros(T, KK, n_imp, n_grid, n_boot)
    else
        nothing
    end

    # ── Step 4: Bootstrap loop ────────────────────────────────────────────
    n_failed = 0
    dim_state = 1 + KK * p_val

    # Pre-allocate reusable buffers outside the loop
    u_temp = zeros(T, n_resample * ℓ, KK)
    m_temp = zeros(T, n_resample * ℓ)
    x_star = zeros(T, TT, dim_state)
    y_star = Matrix{T}(undef, TT, KK)
    xt_buf = Vector{T}(undef, KK)

    # Workspace for the inner estimator and dynamics (reused across draws)
    est_buf = ProxySvarBuffers{T}(TT, KK, dim_state)
    dyn_buf = ProxyDynamicsBuffers{T}(KK, p_val, n_imp)

    for b in 1:n_boot
        # 4a: Resample blocks with replacement
        for j in 1:n_resample
            idx = rand(rng, 1:n_blocks)
            blk_start = (j - 1) * ℓ + 1
            @inbounds for s in 1:ℓ, k in 1:KK

                u_temp[blk_start + s - 1, k] = u_blocks[s, k, idx]
            end
            @inbounds for s in 1:ℓ
                m_temp[blk_start + s - 1] = m_blocks[s, idx]
            end
        end

        # 4b: Apply J&L centering
        u_temp .-= u_center
        m_temp .-= m_center

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
        try
            A_star, U_star, Σ_uu_star,
            Σ_um_star, H1_star = estimate_proxy_svar!(est_buf, y_star, x_star, m_star)

            # 4f: Compute bootstrap dynamics (in-place)
            proxy_svar_dynamics!(dyn_buf, A_star, Σ_uu_star, Σ_um_star, H1_star,
                p_val, s, n_imp, target)

            irf_store[:, :, b] .= dyn_buf.irf
            irf_norm_store[:, :, b] .= dyn_buf.irf_norm
            fevd_store[:, :, b] .= dyn_buf.fevd

            # 4g: AR statistics — anchor on target variable's impact
            if inf.compute_ar
                svma_star = dyn_buf.svma
                svma_target = svma_star[target, 1]
                @inbounds for g in eachindex(inf.ar_grid)
                    grid_g = inf.ar_grid[g]
                    for h in 1:n_imp
                        for k in 1:KK
                            ar_store[k, h, g, b] = s * svma_star[k, h] -
                                                   svma_target * grid_g
                        end
                    end
                end
            end
        catch e
            n_failed += 1
            # Fill failed draws with NaN rather than biasing with previous draw
            fill!(view(irf_store,:,:,b), NaN)
            fill!(view(irf_norm_store,:,:,b), NaN)
            fill!(view(fevd_store,:,:,b), NaN)
        end
    end

    _report_bootstrap_failures("Proxy-SVAR MBB", n_failed, n_boot)

    # ── Step 5: Percentile confidence intervals ───────────────────────────
    ci68_irf, ci95_irf = _percentile_intervals(irf_store, n_boot)
    ci68_irf_norm, ci95_irf_norm = _percentile_intervals(irf_norm_store, n_boot)
    ci68_fevd, ci95_fevd = _percentile_intervals(fevd_store, n_boot)

    # ── Step 6: Hall's intervals ──────────────────────────────────────────
    # Compute point estimates for Hall's correction
    Σ_uu_point = (ν' * ν) ./ TT
    Σ_um_point = vec((ν' * proxy) ./ TT)
    φ_point = sqrt(dot(Σ_um_point, Σ_uu_point \ Σ_um_point))
    H1_point = Σ_um_point ./ φ_point

    dyn_point = proxy_svar_dynamics(A_est, Σ_uu_point, Σ_um_point, H1_point,
        p_val, s, n_imp, target)

    halls68_irf_norm = _halls_intervals(ci68_irf_norm, dyn_point.irf_norm)
    halls95_irf_norm = _halls_intervals(ci95_irf_norm, dyn_point.irf_norm)

    # ── Step 7: AR confidence sets ────────────────────────────────────────
    ar_result = if inf.compute_ar
        _ar_confidence_sets(ar_store, inf.ar_grid, s, n_boot, target)
    else
        nothing
    end

    return (
        ci68_irf = ci68_irf,
        ci95_irf = ci95_irf,
        ci68_irf_norm = ci68_irf_norm,
        ci95_irf_norm = ci95_irf_norm,
        ci68_fevd = ci68_fevd,
        ci95_fevd = ci95_fevd,
        halls68_irf_norm = halls68_irf_norm,
        halls95_irf_norm = halls95_irf_norm,
        ar = ar_result,
        point_irf = dyn_point.irf,
        point_irf_norm = dyn_point.irf_norm,
        point_fevd = dyn_point.fevd,
        point_svma = dyn_point.svma,
        irf_store = irf_store,
        n_failed = n_failed
    )
end

# ============================================================================
# Percentile intervals
# ============================================================================

"""
    _percentile_intervals(store, n_boot) -> (ci68, ci95)

Compute 68% and 95% percentile confidence intervals from bootstrap draws.
`store` has shape (K, n_imp, n_boot). Returns (2, n_imp, K) arrays.
"""
function _percentile_intervals(store::Array{T, 3}, n_boot::Int) where {T}
    KK, n_imp, _ = size(store)

    # Sort along bootstrap dimension
    sorted = sort(store; dims = 3)

    # Quantile indices (matching Python's round())
    i16 = round(Int, 0.16 * n_boot)
    i84 = round(Int, 0.84 * n_boot)
    i025 = round(Int, 0.025 * n_boot)
    i975 = round(Int, 0.975 * n_boot)

    # Clamp to valid range
    i16 = clamp(i16, 1, n_boot)
    i84 = clamp(i84, 1, n_boot)
    i025 = clamp(i025, 1, n_boot)
    i975 = clamp(i975, 1, n_boot)

    ci68 = zeros(T, 2, n_imp, KK)
    ci95 = zeros(T, 2, n_imp, KK)

    for k in 1:KK
        for h in 1:n_imp
            ci68[1, h, k] = sorted[k, h, i16]
            ci68[2, h, k] = sorted[k, h, i84]
            ci95[1, h, k] = sorted[k, h, i025]
            ci95[2, h, k] = sorted[k, h, i975]
        end
    end

    return ci68, ci95
end

# ============================================================================
# Hall's percentile intervals
# ============================================================================

"""
    _halls_intervals(pctile_ci, irf_point) -> (2, n_imp, K) array

Hall's bias-corrected percentile intervals:
  lower = 2*θ̂ - upper_percentile
  upper = 2*θ̂ - lower_percentile
"""
function _halls_intervals(pctile_ci::Array{T, 3},
        irf_point::Matrix{T}) where {T}
    _, n_imp, KK = size(pctile_ci)
    halls = zeros(T, 2, n_imp, KK)

    for k in 1:KK
        for h in 1:n_imp
            halls[1, h, k] = 2 * irf_point[k, h] - pctile_ci[2, h, k]
            halls[2, h, k] = 2 * irf_point[k, h] - pctile_ci[1, h, k]
        end
    end

    return halls
end

# ============================================================================
# Anderson-Rubin confidence sets
# ============================================================================

"""
    _ar_confidence_sets(ar_store, grid, scale, n_boot, target)

Compute Anderson-Rubin confidence sets from bootstrap test statistics.

For each grid point g, compute the proportion of bootstrap draws where
the test statistic T_b(g) ≤ 0. Include g in the confidence set if this
proportion is within the critical region. The target shock's own-variable
impact at h=1 is always included (equals `scale` by construction).

Returns named tuple with `index68`, `index90`, `index95` (boolean arrays)
and `grid`.
"""
function _ar_confidence_sets(ar_store::Array{T, 4},
        grid::Vector{Float64}, scale::Float64, n_boot::Int, target::Int) where {T}
    KK, n_imp, n_grid, _ = size(ar_store)

    # Compute rejection rates: P(T_b(g) ≤ 0)
    rates = zeros(KK, n_imp, n_grid)
    for k in 1:KK, h in 1:n_imp, g in 1:n_grid
        count = 0
        for b in 1:n_boot
            if ar_store[k, h, g, b] <= 0
                count += 1
            end
        end
        rates[k, h, g] = count / n_boot
    end

    # Inclusion indices
    index68 = falses(n_grid, n_imp, KK)
    index90 = falses(n_grid, n_imp, KK)
    index95 = falses(n_grid, n_imp, KK)

    for k in 1:KK, h in 1:n_imp, g in 1:n_grid
        # Special case: target shock's own-variable impact equals `scale` by
        # construction — always include it at (h=1, k=target).
        if grid[g] ≈ scale && h == 1 && k == target
            index68[g, h, k] = true
            index90[g, h, k] = true
            index95[g, h, k] = true
        else
            r = rates[k, h, g]
            index68[g, h, k] = 0.16 ≤ r ≤ 0.84
            index90[g, h, k] = 0.05 ≤ r ≤ 0.95
            index95[g, h, k] = 0.025 ≤ r ≤ 0.975
        end
    end

    return (index68 = index68, index90 = index90, index95 = index95,
        grid = grid, rates = rates)
end

# ============================================================================
# Integration with irf() dispatch
# ============================================================================

"""
    compute_inference_bands(model, id::IVIdentification, irf_point,
        inf::ProxySVARMBB, coverage, rng)

Dispatch for Jentsch-Lunsford MBB on SVAR-IV models. Works with any VARModel
(not just IVSVAR) as long as the identification carries an instrument.

When `inf.compute_ar == true`, confidence bands use the convex hull of the
Anderson-Rubin confidence set (weak-instrument robust). Otherwise, percentile CIs
are used.
"""
function compute_inference_bands(
        model::VARModel{T},
        identification::IVIdentification,
        irf_point::Array{T, 3},
        inf::ProxySVARMBB,
        coverage::Vector{Float64},
        ::AbstractNormalization,  # ProxySVARMBB uses its own norm_scale
        rng::AbstractRNG) where {T}
    horizon = size(irf_point, 1) - 1
    id = _resolve_iv(model, identification)
    _,
    target = _extract_instrument(id.instrument, size(model.residuals, 1),
        n_lags(model), model.names)

    # Run the full MBB
    mbb = proxy_svar_mbb(model, id, horizon, inf; rng = rng)

    K = n_vars(model)
    n_imp = horizon + 1

    # Construct 4D draws from irf_store (K, n_imp, n_boot) — identified shock only.
    # Normalize by the target variable's impact so the draws align with `irf_point`
    # (which has the identified column at position `target`).
    scale_factor = mbb.point_irf[target, 1]
    irf_store = mbb.irf_store  # (K, n_imp, n_boot)
    n_boot = size(irf_store, 3)

    draws_4d = zeros(T, n_boot, n_imp, K, K)
    # Fill identified shock column at position `target` after normalization
    for b in 1:n_boot
        for h in 1:n_imp, k in 1:K

            draws_4d[b, h, k, target] = irf_store[k, h, b] / scale_factor
        end
    end
    # Non-identified columns: fill with point estimate (no variation)
    for j in 1:K
        j == target && continue
        for h in 1:n_imp, k in 1:K

            draws_4d[:, h, k, j] .= irf_point[h, k, j]
        end
    end

    draws = inf.save_draws ? draws_4d : nothing
    stderr = dropdims(std(draws_4d; dims = 1); dims = 1)

    lower = Vector{Array{T, 3}}(undef, length(coverage))
    upper = Vector{Array{T, 3}}(undef, length(coverage))

    for (i, α) in enumerate(coverage)
        lb = zeros(T, n_imp, K, K)
        ub = zeros(T, n_imp, K, K)

        if inf.compute_ar && mbb.ar !== nothing
            # AR confidence sets: convex hull [min, max] of included grid points.
            # Conservative but plottable. Raw boolean masks in mbb.ar for full detail.
            ar_idx = if α ≤ 0.70
                mbb.ar.index68
            elseif α ≤ 0.92
                mbb.ar.index90
            else
                mbb.ar.index95
            end
            grid = mbb.ar.grid
            for h in 1:n_imp, k in 1:K

                included = ar_idx[:, h, k]
                if any(included)
                    lb[h, k, target] = minimum(grid[included]) / scale_factor
                    ub[h, k, target] = maximum(grid[included]) / scale_factor
                else
                    lb[h, k, target] = T(NaN)
                    ub[h, k, target] = T(NaN)
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

# Auto-dispatch: BlockBootstrap + IVIdentification → ProxySVARMBB (J&L corrected)
function compute_inference_bands(
        model::VARModel{T},
        identification::IVIdentification,
        irf_point::Array{T, 3},
        inf::BlockBootstrap,
        coverage::Vector{Float64},
        normalization::AbstractNormalization,
        rng::AbstractRNG) where {T}
    # Promote to Jentsch-Lunsford MBB which jointly resamples (residuals, proxy)
    proxy_inf = ProxySVARMBB(inf.reps; block_length = inf.block_length,
        save_draws = inf.save_draws)
    return compute_inference_bands(
        model, identification, irf_point, proxy_inf, coverage, normalization, rng)
end

# Error guard: ProxySVARMBB requires IVIdentification
function compute_inference_bands(
        ::VARModel{T},
        identification::AbstractIdentification,
        ::Array{T, 3},
        ::ProxySVARMBB,
        ::Vector{Float64},
        ::AbstractNormalization,
        ::AbstractRNG) where {T}
    throw(ArgumentError(
        "ProxySVARMBB inference requires IVIdentification. " *
        "Use IVIdentification(Z, target_shock) or a different inference method."))
end

export proxy_svar_mbb, proxy_svar_dynamics
