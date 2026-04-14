# ============================================================================
# IV-SVAR Estimation — Proxy-SVAR (Stock & Watson 2018)
# ============================================================================
#
# Three-step procedure:
#   1. Estimate reduced-form VAR(p) by OLS
#   2. Use external instrument Z_t to identify the first column of Θ₀
#      via 2SLS on VAR residuals
#   3. IRFs: Θ_h = C_h · Θ₀  (computed by irf() using companion form)
#
# References:
#   Stock & Watson (2018), "Identification and estimation of dynamic causal
#     effects in macroeconomics using external instruments"
#   Mertens & Ravn (2013), "The dynamic effects of personal and corporate
#     income tax changes in the United States"

"""
    StatsBase.fit(::Type{IVSVAR}, Y, n_lags; instrument, kwargs...)

Fit a proxy-SVAR model following Stock & Watson (2018).

# Algorithm
1. Estimate reduced-form VAR(p) by OLS (with optional constraints)
2. Extract VAR residuals ν_t
3. Regress ν₁_t on Z_t (first stage → relevance)
4. Regress ν_{i,t} on ν̂₁_t for i=2,...,n (second stage → impact coefficients)
5. Construct full structural impact matrix P

# Arguments
- `::Type{IVSVAR}`: Specification type
- `Y::AbstractMatrix{T}`: Data matrix (T_total × n_vars). The variable in column
  `instrument.target_shock` is the "policy variable" whose structural shock is
  identified.
- `n_lags::Int`: VAR lag order

# Keyword Arguments
- `instrument::AbstractInstrument`: External instrument.  Z can have the same
  number of rows as Y (T_total); the first `n_lags` rows are dropped
  automatically.  Pre-trimmed Z (T_total - n_lags rows) also accepted.
- `constraints::Vector{<:AbstractConstraint}`: VAR coefficient constraints
- `names::Union{Nothing,Vector{Symbol}}`: Variable names
- `demean::Bool=false`: Demean data before estimation

# Returns
- `VARModel{T, IVSVAR{I}}` with metadata fields:
  - `structural_impact::Matrix{T}` — full n×n impact matrix (first column identified)
  - `first_stage_F::Float64` — first-stage F-statistic
  - `target_shock::Int` — index of the identified shock

# Example
```julia
Z = ExternalInstrument(proxy_data, 1)  # proxy for shock 1
model = fit(IVSVAR, Y, 4; instrument=Z)
result = irf(model, IVIdentification(); horizon=48)
```
"""
function StatsBase.fit(::Type{IVSVAR}, Y::AbstractMatrix{T}, n_lags::Int;
        instrument::AbstractInstrument,
        constraints::Vector{<:AbstractConstraint} = AbstractConstraint[],
        names::Union{Nothing, Vector{Symbol}} = nothing,
        demean::Bool = false) where {T <: AbstractFloat}

    # ── Step 1: Reduced-form VAR ──────────────────────────────────────────
    var_ols = fit(OLSVAR, Y, n_lags;
        constraints = constraints, names = names, demean = demean)
    ν = residuals(var_ols)                 # (T-p) × n
    n_obs, n = size(ν)
    var_names = var_ols.names

    # ── Extract instrument data ───────────────────────────────────────────
    Z, target = _extract_instrument(instrument, n_obs, n_lags, var_names)

    # ── Step 2: 2SLS identification ───────────────────────────────────────
    β_iv, F_stat, π_hat = _iv_identify(ν, Z, target)

    # ── Step 3: Build full structural impact matrix ───────────────────────
    P = _build_full_impact_matrix(β_iv, target, Matrix(var_ols.Σ))

    # ── Assemble VARModel with IVSVAR spec ────────────────────────────────
    spec = IVSVAR(instrument)

    metadata = (
        n_obs_total = size(Y, 1),
        n_obs_used = n_obs,
        demean = demean,
        timestamp = now(),
        structural_impact = P,
        first_stage_F = F_stat,
        first_stage_coef = π_hat,
        target_shock = target,
        iv_coefficients = β_iv
    )

    return VARModel(spec, var_ols.Y, var_ols.X, var_ols.coefficients,
        var_ols.residuals, var_ols.Σ, var_ols.companion, var_names, metadata)
end

# ============================================================================
# Internal helpers
# ============================================================================

"""
    _resize_instrument(instrument::ExternalInstrument, n_obs) -> ExternalInstrument

Trim or pad instrument to match a different residual count (used in bootstrap).
"""
function _resize_instrument(instrument::ExternalInstrument{T, S}, n_obs::Int) where {T, S}
    Z = instrument.Z
    n_orig = size(Z, 1)
    if n_orig == n_obs
        return instrument
    elseif n_orig > n_obs
        # Trim from the end (most common case in bootstrap)
        return ExternalInstrument(Z[1:n_obs, :], instrument.target_shock;
            method = instrument.method)
    else
        error("Cannot expand instrument from $n_orig to $n_obs rows")
    end
end

function _resize_instrument(instrument::ProxyIV{T}, n_obs::Int) where {T}
    Z = instrument.proxies
    n_orig = size(Z, 1)
    if n_orig == n_obs
        return instrument
    elseif n_orig > n_obs
        return ProxyIV(Z[1:n_obs, :], instrument.target_shocks;
            relevance_threshold = instrument.relevance_threshold)
    else
        error("Cannot expand proxy from $n_orig to $n_obs rows")
    end
end

# ============================================================================
# 2SLS identification core
# ============================================================================

"""
    _iv_identify(ν, Z, target) -> (β_iv, F_stat, π_hat)

Two-stage least squares identification of one structural shock from VAR
residuals and an external instrument (Stock & Watson 2018).

Single-shock identification via equation-by-equation 2SLS. For multiple-shock
identification (k proxies for k shocks), system GMM is required.

# Arguments
- `ν`: (T, K) VAR residuals
- `Z`: (T, k_z) instrument matrix
- `target`: index of the shock to identify

# Returns
- `β_iv`: (K,) identified impact column (unit effect normalization: β_iv[target] = 1)
- `F_stat`: first-stage F-statistic
- `π_hat`: first-stage coefficients
"""
function _iv_identify(ν::Matrix{T}, Z::Matrix{T}, target::Int) where {T}
    n_obs, n = size(ν)
    k_z = size(Z, 2)

    # Partition: ν₁ = residuals of target variable
    ν₁ = ν[:, target]
    idx_other = setdiff(1:n, target)

    # First stage: ν₁ = Z π + e
    Z_const = hcat(ones(T, n_obs), Z)
    π_hat = Z_const \ ν₁
    ν₁_hat = Z_const * π_hat
    e_fs = ν₁ .- ν₁_hat

    # First-stage F-statistic (exclusion of Z)
    RSS_r = sum(abs2, ν₁ .- mean(ν₁))
    RSS_u = sum(abs2, e_fs)
    F_stat = ((RSS_r - RSS_u) / k_z) / (RSS_u / (n_obs - k_z - 1))

    # Second stage: ν_{i} = α_i + β_i ν̂₁ + u_i  for i ≠ target
    X_ss = hcat(ones(T, n_obs), ν₁_hat)
    β_iv = zeros(T, n)
    β_iv[target] = one(T)
    for i in idx_other
        coef_i = X_ss \ ν[:, i]
        β_iv[i] = coef_i[2]
    end

    return β_iv, F_stat, π_hat
end

# ============================================================================
# Resolve IV identification — handles both new and backward-compat paths
# ============================================================================

"""
    _resolve_iv(model::VARModel, id::IVIdentification) -> IVIdentification

Ensure the identification object has an instrument. If `id.instrument` is `nothing`
(backward compat with `IVIdentification()`), extract the instrument from an IVSVAR model.
"""
function _resolve_iv(model::VARModel, id::IVIdentification)
    if id.instrument !== nothing
        return id
    elseif model.spec isa IVSVAR
        return IVIdentification(model.spec.instrument)
    else
        throw(ArgumentError(
            "IVIdentification requires an instrument: use IVIdentification(Z, target_shock)"))
    end
end

# ============================================================================
# Instrument extraction
# ============================================================================

"""
    _extract_instrument(instrument, n_obs, n_lags, names) -> (Z, target::Int)

Extract instrument matrix and target shock index, trimming if needed.
Resolves `Symbol` target shocks against `names`.

Z should have the same number of rows as the original data Y.  Internally, rows
are dropped from the start so that Z aligns with VAR residuals (`n_obs` rows).
If Z already has `n_obs` rows it is used as-is.
"""
function _extract_instrument(instrument::ExternalInstrument,
        n_obs::Int, n_lags::Int, names::Vector{Symbol})
    Z = instrument.Z
    n_z = size(Z, 1)
    target = _resolve_target(instrument.target_shock, names)
    if n_z == n_obs
        return Z, target
    elseif n_z > n_obs
        # Trim from the start: keep last n_obs rows
        return Z[(n_z - n_obs + 1):end, :], target
    else
        throw(DimensionMismatch(
            "Instrument has $n_z rows but needs at least $n_obs. " *
            "Z should have the same number of rows as Y."))
    end
end

function _extract_instrument(instrument::ProxyIV{T},
        n_obs::Int, n_lags::Int, names::Vector{Symbol}) where {T}
    Z = instrument.proxies
    n_z = size(Z, 1)
    length(instrument.target_shocks) == 1 ||
        throw(ArgumentError(
            "SVAR-IV currently supports single-shock identification. " *
            "Got $(length(instrument.target_shocks)) target shocks."))
    target = instrument.target_shocks[1]
    if n_z == n_obs
        return Z, target
    elseif n_z > n_obs
        return Z[(n_z - n_obs + 1):end, :], target
    else
        throw(DimensionMismatch(
            "Proxy has $n_z rows but needs at least $n_obs. " *
            "Proxy should have the same number of rows as Y."))
    end
end

"""
    _build_full_impact_matrix(β_iv, target, Σ) -> Matrix

Build the full n×n structural impact matrix P.

The first column (reordered to position `target`) is the IV-identified column β_iv.
Remaining columns are filled via Cholesky of the residual covariance
Σ - β_iv · β_iv', providing a complete rotation matrix.

This follows Mertens & Ravn (2013): only the `target` column is externally
identified; the rest use a Cholesky-based fill for the orthogonal complement.
"""
function _build_full_impact_matrix(β_iv::Vector{T}, target::Int, Σ::Matrix{T}) where {T}
    n = length(β_iv)
    P = zeros(T, n, n)

    # Place the identified column
    P[:, target] = β_iv

    # Residual covariance after removing identified shock contribution
    # Under unit variance normalization for shock 1:
    #   Σ = β_iv β_iv' + Σ_rest  →  Σ_rest = Σ - β_iv β_iv'
    Σ_rest = Σ - β_iv * β_iv'

    # Σ_rest should be positive semi-definite. Regularize if needed.
    # Use eigendecomposition for robustness
    idx_other = setdiff(1:n, target)
    Σ_rest_sub = Σ_rest[idx_other, idx_other]

    # Eigenvalue cleanup: force small negatives to zero
    eig = eigen(Symmetric(Σ_rest_sub))
    λ = max.(eig.values, zero(T))
    L_sub = eig.vectors * Diagonal(sqrt.(λ))

    # Fill remaining columns with Cholesky-like decomposition of Σ_rest
    # Map back to full matrix
    for (j_local, j_global) in enumerate(idx_other)
        P[idx_other, j_global] = L_sub[:, j_local]
    end

    # Cross terms: P[target, j] for j ≠ target
    # From Σ_rest[target, idx_other] = P[target, idx_other] * L_sub'
    # → P[target, idx_other] = Σ_rest[target, idx_other] * pinv(L_sub')
    cross = Σ_rest[target, idx_other]
    if any(!iszero, cross) && any(!iszero, λ)
        P[target, idx_other] = vec(cross' * pinv(L_sub'))
    end

    return P
end

# ============================================================================
# Bootstrap re-estimation helper
# ============================================================================

"""
    refit_for_bootstrap(model::VARModel{T, <:IVSVAR}, Y_boot, n_lags_val) where T

Re-estimate a proxy-SVAR model on bootstrap data. Used internally by
the bootstrap machinery.
"""
function refit_for_bootstrap(model::VARModel{T, <:IVSVAR}, Y_boot::Matrix{T},
        n_lags_val::Int) where {T}
    constraints_arg = model.coefficients.constraints
    constraints_arg = constraints_arg === nothing ? AbstractConstraint[] : constraints_arg

    # Adjust instrument to match bootstrap residual count
    n_resid_boot = size(Y_boot, 1) - n_lags_val
    instrument = _resize_instrument(model.spec.instrument, n_resid_boot)

    return fit(IVSVAR, Y_boot, n_lags_val;
        instrument = instrument,
        constraints = constraints_arg,
        names = model.names)
end

"""
    refit_for_bootstrap(model::VARModel{T}, Y_boot, n_lags_val) where T

Re-estimate a non-IVSVAR model on bootstrap data. Generic fallback.
"""
function refit_for_bootstrap(model::VARModel{T}, Y_boot::Matrix{T},
        n_lags_val::Int) where {T}
    constraints_arg = model.coefficients.constraints
    constraints_arg = constraints_arg === nothing ? AbstractConstraint[] : constraints_arg
    return fit(typeof(model.spec), Y_boot, n_lags_val;
        constraints = constraints_arg,
        names = model.names)
end
