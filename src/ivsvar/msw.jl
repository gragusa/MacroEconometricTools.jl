# ============================================================================
# Montiel Olea, Stock & Watson (2021) Confidence Sets for Proxy-SVARs
# ============================================================================
#
# Analytic confidence sets that are robust to weak instruments.
# No bootstrap required. Uses heteroskedasticity-robust covariance.
#
# References:
#   Montiel Olea, Stock & Watson (2021), "Inference in SVARs Identified
#     with an External Instrument"

"""
    msw_confidence_set(model::VARModel, id::IVIdentification; norm_scale=-1.0, horizon=20)
    msw_confidence_set(model::VARModel{T, <:IVSVAR}; norm_scale=-1.0, horizon=20)

Compute Montiel Olea, Stock & Watson (2021) confidence sets for an SVAR-IV.

These are analytic (no bootstrap), robust to weak instruments, and based on
inverting a Wald-type test statistic using the quadratic formula.

# Returns
Named tuple with:
- `cs68_irf_norm`: 68% confidence set for normalized IRFs
- `cs95_irf_norm`: 95% confidence set for normalized IRFs
- `vcv_matrix`: Full variance-covariance matrix
- `wald_stat`: Wald statistic for proxy strength
- `bounded68::Bool`: Whether 68% set is bounded
- `bounded95::Bool`: Whether 95% set is bounded
"""
function msw_confidence_set(model::VARModel{T}, id::IVIdentification;
        norm_scale::Float64 = -1.0,
        horizon::Int = 20) where {T}
    resolved = _resolve_iv(model, id)
    ν = model.residuals
    Z, _ = _extract_instrument(resolved.instrument, size(ν, 1), n_lags(model), model.names)
    return _msw_impl(model, vec(Z); norm_scale = norm_scale, horizon = horizon)
end

# Backward compat: instrument in model
function msw_confidence_set(model::VARModel{T, <:IVSVAR};
        norm_scale::Float64 = -1.0,
        horizon::Int = 20) where {T}
    return msw_confidence_set(model, IVIdentification(model.spec.instrument);
        norm_scale = norm_scale, horizon = horizon)
end

function _msw_impl(model::VARModel{T}, proxy::Vector{T};
        norm_scale::Float64 = -1.0,
        horizon::Int = 20) where {T}
    ν = model.residuals
    TT, KK = size(ν)
    p_val = n_lags(model)
    n_imp = horizon + 1
    s = norm_scale

    # ── Step 1: Construct RHS matrix ──────────────────────────────────────
    # rhs = [1, y_{t-1}, ..., y_{t-p}] for each t
    Y_full = model.Y  # (T_total, K) — includes initial lags
    T_total = size(Y_full, 1)

    rhs = ones(T, TT, 1 + KK * p_val)
    for lag in 1:p_val
        # y_{t-lag} aligned with residuals (which start at observation p+1)
        rhs[:, (1 + (lag - 1) * KK + 1):(1 + lag * KK)] .= Y_full[(p_val + 1 - lag):(T_total - lag), :]
    end

    # ── Step 2: Covariance matrix construction ────────────────────────────
    dim_total = KK * (KK * p_val + 1) + KK  # Total dimension for Ω

    zz = zeros(T, TT, dim_total)

    # Products of RHS regressors with residuals: x_{t,j} * u_{t,k}
    counter = 1
    for j in 1:(1 + KK * p_val)
        for k in 1:KK
            zz[:, counter] .= ν[:, k] .* rhs[:, j]
            counter += 1
        end
    end

    # Covariance of proxy with residuals
    γ_est = vec(proxy' * ν ./ TT)  # (K,)

    for k in 1:KK
        zz[:, KK * (KK * p_val + 1) + k] .= ν[:, k] .* proxy .- γ_est[k]
    end

    # Heteroskedasticity-robust covariance: Ω = z'z / T
    Ω = (zz' * zz) ./ TT

    # ── Step 3: Auxiliary variance matrix ─────────────────────────────────
    I_K = Matrix{T}(I, KK, KK)
    dim_A = KK * (KK * p_val + 1)

    Q1 = (rhs' * rhs) ./ TT
    Q1_inv = inv(Q1)

    Q2 = (proxy' * rhs ./ TT) * Q1_inv  # (1, 1+K*p)

    aux = Matrix{T}(I, dim_total, dim_total)
    aux[1:dim_A, 1:dim_A] .= kron(Q1_inv, I_K)
    aux[(dim_A + 1):dim_total, 1:dim_A] .= -kron(Q2, I_K)

    V = aux * Ω * aux'

    # Extract submatrices
    # V11: variance of lag coefficients (excluding intercept)
    V11 = V[(KK + 1):dim_A, (KK + 1):dim_A]
    # V31: covariance of γ with lag coefficients
    V31 = V[(dim_A + 1):dim_total, (KK + 1):dim_A]
    # V33: variance of γ
    V33 = V[(dim_A + 1):dim_total, (dim_A + 1):dim_total]

    # ── Step 4: Companion matrix and G matrices ───────────────────────────
    coefs = coef(model)
    A_est = zeros(T, KK, 1 + KK * p_val)
    A_est[:, 1] .= coefs.intercept
    for lag in 1:p_val
        A_est[:, (1 + (lag - 1) * KK + 1):(1 + lag * KK)] .= coefs.lags[:, :, lag]
    end

    comp = zeros(T, KK * p_val, KK * p_val)
    if p_val == 1
        comp .= A_est[:, 2:(KK + 1)]
    else
        comp[1:KK, :] .= A_est[:, 2:(KK * p_val + 1)]
        comp[(KK + 1):(KK * p_val), 1:(KK * (p_val - 1))] .= I(KK * (p_val - 1))
    end

    # Compute bigA[h] = comp^h and G matrices
    bigA = zeros(T, KK * p_val, KK * p_val, n_imp)
    bigA[:, :, 1] .= I(KK * p_val)

    dim_G = KK * KK
    dim_G_cols = KK * KK * p_val
    G = zeros(T, dim_G, dim_G_cols, n_imp + 1)
    G[:, :, 2] .= kron(bigA[1:KK, :, 1], I_K)

    comp_power = Matrix{T}(I, KK * p_val, KK * p_val)

    for h in 1:(n_imp - 1)
        comp_power = comp_power * comp'
        bigA[:, :, h + 1] .= comp_power

        # G matrix: sum of Kronecker products
        temp = zeros(T, dim_G, dim_G_cols)
        for n in 0:h
            Φ_n = bigA[1:KK, 1:KK, n + 1]'
            temp .+= kron(bigA[1:KK, :, h - n + 1], Φ_n)
        end
        G[:, :, h + 2] .= temp
    end

    # ── Step 5: Wald statistic and critical values ────────────────────────
    wald = TT * γ_est[1]^2 / V33[1, 1]

    crit68 = 0.9889   # χ²(1) at 68%
    crit95 = 3.8415   # χ²(1) at 95%

    # ── Step 6: Solve quadratic for each (variable, horizon) ──────────────
    aa68 = TT * γ_est[1]^2 - crit68 * V33[1, 1]
    aa95 = TT * γ_est[1]^2 - crit95 * V33[1, 1]

    bounded68 = aa68 > 0
    bounded95 = aa95 > 0

    cs68 = bounded68 ? zeros(T, 2, n_imp, KK) : zeros(T, 4, n_imp, KK)
    cs95 = bounded95 ? zeros(T, 2, n_imp, KK) : zeros(T, 4, n_imp, KK)

    for k in 1:KK
        for h in 1:n_imp
            Φ_h = bigA[1:KK, 1:KK, h]'

            # Kronecker product for variance computation
            Mkron = kron(γ_est', I_K[k:k, :])

            temp1 = (Φ_h[k:k, :] * γ_est)[1]
            temp2 = Mkron * G[:, :, h]

            # b coefficient
            bpart1 = -TT * temp1 * γ_est[1]
            bpart2 = (temp2 * V31[1:1, :]')[1] + (Φ_h[k:k, :] * V33[:, 1:1])[1]
            bb68 = 2 * s * (bpart1 + crit68 * bpart2)
            bb95 = 2 * s * (bpart1 + crit95 * bpart2)

            # c coefficient
            cpart1 = TT * temp1^2
            cpart2 = (temp2 * V11 * temp2')[1] +
                     2 * (temp2 * V31' * Φ_h[k:k, :]')[1] +
                     (Φ_h[k:k, :] * V33 * Φ_h[k:k, :]')[1]
            cc68 = s^2 * (cpart1 - crit68 * cpart2)
            cc95 = s^2 * (cpart1 - crit95 * cpart2)

            # Solve quadratic
            _solve_msw_quadratic!(cs68, aa68, bb68, cc68, h, k, bounded68)
            _solve_msw_quadratic!(cs95, aa95, bb95, cc95, h, k, bounded95)
        end
    end

    # Normalization at impact
    if bounded68
        cs68[:, 1, 1] .= s
    else
        cs68[1:2, 1, 1] .= s
        cs68[3:4, 1, 1] .= 0
    end
    if bounded95
        cs95[:, 1, 1] .= s
    else
        cs95[1:2, 1, 1] .= s
        cs95[3:4, 1, 1] .= 0
    end

    return (
        cs68_irf_norm = cs68,
        cs95_irf_norm = cs95,
        vcv_matrix = V,
        wald_stat = wald,
        bounded68 = bounded68,
        bounded95 = bounded95
    )
end

"""
    _solve_msw_quadratic!(cs, aa, bb, cc, h, k, bounded)

Solve the MSW quadratic inequality and store roots in confidence set array.
"""
function _solve_msw_quadratic!(cs::Array{T, 3},
        aa::T, bb::T, cc::T,
        h::Int, k::Int, bounded::Bool) where {T}
    disc = bb^2 - 4 * aa * cc

    if bounded
        # Bounded set: standard quadratic roots
        if disc > 0
            sq = sqrt(disc)
            cs[1, h, k] = (-bb - sq) / (2 * aa)
            cs[2, h, k] = (-bb + sq) / (2 * aa)
        else
            # No real roots → empty set at this (h, k)
            cs[1, h, k] = T(NaN)
            cs[2, h, k] = T(NaN)
        end
    else
        # Unbounded set: disjoint intervals
        cs[1, h, k] = T(-Inf)
        cs[4, h, k] = T(Inf)
        if disc > 0
            sq = sqrt(disc)
            cs[2, h, k] = (-bb + sq) / (2 * aa)
            cs[3, h, k] = (-bb - sq) / (2 * aa)
        else
            cs[2, h, k] = T(Inf)
            cs[3, h, k] = T(-Inf)
        end
    end
end

export msw_confidence_set
