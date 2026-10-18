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
    msw_confidence_set(model::VARModel, id::IVIdentification;
                       coverage=[0.68, 0.90, 0.95], norm_scale=-1.0, horizon=20)
    msw_confidence_set(model::VARModel{T, <:IVSVAR}; coverage, norm_scale, horizon)

Compute Montiel Olea, Stock & Watson (2021) confidence sets for an SVAR-IV.

These are analytic (no bootstrap), robust to weak instruments, and based on
inverting a Wald-type test statistic using the quadratic formula. The critical
value at level `α` is `quantile(Chisq(1), α)`. The instrument must be observed in
every residual period.

# Returns
A `NamedTuple`; vector fields are aligned with `coverage`.
- `coverage::Vector{Float64}`: the levels, sorted.
- `cs_irf_norm`: one array per level for the normalized IRF (impact on the target
  variable equal to `norm_scale`), of size `(2, variable, horizon)` when the set is
  bounded (`[lower, upper]`) and `(4, variable, horizon)` when it is not
  (`(-Inf, a] ∪ [b, Inf)` stored as `[-Inf, a, b, Inf]`). `horizon` has
  `horizon + 1` entries, impact first. Entries are `NaN` where the set is empty.
- `bounded::Vector{Bool}`: whether the set at each level is bounded, which holds
  when the Wald statistic exceeds the critical value.
- `vcv_matrix`: full variance-covariance matrix of the reduced-form coefficients and
  the residual-instrument covariances.
- `wald_stat`: Wald statistic for proxy strength.
"""
function msw_confidence_set(
        model::VARModel{T}, id::IVIdentification;
        coverage::AbstractVector{<:Real} = [0.68, 0.9, 0.95],
        norm_scale::Float64 = -1.0,
        horizon::Int = 20
) where {T}
    coverage = collect(Float64, _validate_irf_args(horizon, coverage))
    resolved = _resolve_iv(model, id)
    ν = model.residuals
    Z,
    target = _extract_instrument(resolved.instrument, size(ν, 1), n_lags(model), model.names)
    return _msw_impl(
        model, vec(_require_observed(Z, "msw_confidence_set")),
        target, coverage; norm_scale, horizon
    )
end

# Deprecated `IVSVAR` models carry the instrument in the spec.
function msw_confidence_set(
        model::VARModel{T, <:IVSVAR};
        coverage::AbstractVector{<:Real} = [0.68, 0.9, 0.95],
        norm_scale::Float64 = -1.0,
        horizon::Int = 20
) where {T}
    return msw_confidence_set(
        model, IVIdentification(model.spec.instrument);
        coverage, norm_scale, horizon
    )
end

function _msw_impl(
        model::VARModel{T}, proxy::Vector{T}, target::Int, coverage::Vector{Float64};
        norm_scale::Float64 = -1.0,
        horizon::Int = 20
) where {T}
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
    A_est = Matrix(transpose(coef(model)))

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
    wald = TT * γ_est[target]^2 / V33[target, target]
    crit = [quantile(Chisq(1), α) for α in coverage]

    # ── Step 6: Solve quadratic for each (variable, horizon) ──────────────
    aa = [TT * γ_est[target]^2 - c * V33[target, target] for c in crit]
    bounded = aa .> 0
    cs = [fill(T(NaN), bnd ? 2 : 4, KK, n_imp) for bnd in bounded]

    for h in 1:n_imp, k in 1:KK

        Φ_h = bigA[1:KK, 1:KK, h]'

        # Kronecker product for variance computation
        Mkron = kron(γ_est', I_K[k:k, :])

        temp1 = (Φ_h[k:k, :] * γ_est)[1]
        temp2 = Mkron * G[:, :, h]

        # b coefficient
        bpart1 = -TT * temp1 * γ_est[target]
        bpart2 = (temp2 * V31[target:target, :]')[1] +
                 (Φ_h[k:k, :] * V33[:, target:target])[1]

        # c coefficient
        cpart1 = TT * temp1^2
        cpart2 = (temp2 * V11 * temp2')[1] +
                 2 * (temp2 * V31' * Φ_h[k:k, :]')[1] +
                 (Φ_h[k:k, :] * V33 * Φ_h[k:k, :]')[1]

        for i in eachindex(crit)
            bb = 2 * s * (bpart1 + crit[i] * bpart2)
            cc = s^2 * (cpart1 - crit[i] * cpart2)
            _solve_msw_quadratic!(cs[i], aa[i], bb, cc, k, h, bounded[i])
        end
    end

    # Normalization at impact (target variable's IRF equals `s` by construction)
    for i in eachindex(cs)
        cs[i][1:2, target, 1] .= s
        bounded[i] || (cs[i][3:4, target, 1] .= 0)
    end

    return (
        coverage = coverage,
        cs_irf_norm = cs,
        bounded = bounded,
        vcv_matrix = V,
        wald_stat = wald
    )
end

"""
    _solve_msw_quadratic!(cs, aa, bb, cc, k, h, bounded)

Solve the MSW quadratic inequality and store its roots in `cs[:, k, h]`.
"""
function _solve_msw_quadratic!(
        cs::Array{T, 3},
        aa::T, bb::T, cc::T,
        k::Int, h::Int, bounded::Bool
) where {T}
    disc = bb^2 - 4 * aa * cc

    return if bounded
        # Bounded set: standard quadratic roots
        if disc > 0
            sq = sqrt(disc)
            cs[1, k, h] = (-bb - sq) / (2 * aa)
            cs[2, k, h] = (-bb + sq) / (2 * aa)
        else
            # No real roots → empty set at this (k, h)
            cs[1, k, h] = T(NaN)
            cs[2, k, h] = T(NaN)
        end
    else
        # Unbounded set: disjoint intervals
        cs[1, k, h] = T(-Inf)
        cs[4, k, h] = T(Inf)
        if disc > 0
            sq = sqrt(disc)
            cs[2, k, h] = (-bb + sq) / (2 * aa)
            cs[3, k, h] = (-bb - sq) / (2 * aa)
        else
            cs[2, k, h] = T(Inf)
            cs[3, k, h] = T(-Inf)
        end
    end
end
