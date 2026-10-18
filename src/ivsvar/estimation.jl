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

Deprecated. Fit the reduced form with `fit(OLSVAR, Y, n_lags)` and pass the
instrument to `IVIdentification`:

```julia
model = fit(OLSVAR, Y, 4)
result = irf(model, IVIdentification(proxy_data, 1); horizon=48)
first_stage_F(model, IVIdentification(proxy_data, 1))
```

Returns a `VARModel{T, IVSVAR{I}}` whose `metadata` holds `structural_impact`,
`first_stage_F`, `first_stage_coef`, `target_shock`, and `iv_coefficients`.
"""
function StatsBase.fit(
        ::Type{IVSVAR}, Y::AbstractMatrix{T}, n_lags::Int;
        instrument::AbstractInstrument, kwargs...
    ) where {T <: AbstractFloat}
    Base.depwarn(
        "`fit(IVSVAR, Y, n_lags; instrument)` is deprecated; use " *
            "`fit(OLSVAR, Y, n_lags)` and `IVIdentification(instrument)`.",
        :fit
    )
    return _fit_ivsvar(Y, n_lags; instrument, kwargs...)
end

function _fit_ivsvar(
        Y::AbstractMatrix{T}, n_lags::Int;
        instrument::AbstractInstrument,
        constraints::Vector{<:AbstractConstraint} = AbstractConstraint[],
        names::Union{Nothing, Vector{Symbol}} = nothing,
        demean::Bool = false
    ) where {T <: AbstractFloat}

    # ── Step 1: Reduced-form VAR ──────────────────────────────────────────
    var_ols = fit(
        OLSVAR, Y, n_lags;
        constraints = constraints, names = names, demean = demean
    )
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
        iv_coefficients = β_iv,
    )

    return VARModel(
        spec, var_ols.Y, var_ols.X, var_ols.coefficients,
        var_ols.residuals, var_ols.Σ, var_ols.companion, var_names, var_ols.dates,
        metadata
    )
end

# ============================================================================
# Internal helpers
# ============================================================================

"""
    _resize_instrument(instrument::ExternalInstrument, n_obs) -> ExternalInstrument

Keep the first `n_obs` rows of the instrument (used when refitting a deprecated
`IVSVAR` model on bootstrap data).
"""
function _resize_instrument(instrument::ExternalInstrument, n_obs::Int)
    Z = instrument.Z
    n_orig = size(Z, 1)
    if n_orig == n_obs
        return instrument
    elseif n_orig > n_obs
        return ExternalInstrument(Z[1:n_obs, :], instrument.target_shock)
    else
        throw(DimensionMismatch("Cannot expand instrument from $n_orig to $n_obs rows"))
    end
end

# ============================================================================
# 2SLS identification core
# ============================================================================

"""
    _observed_rows(Z) -> Vector{Int}

Rows of `Z` in which every column is observed (not `missing`).
"""
function _observed_rows(Z::AbstractMatrix)
    return [i for i in axes(Z, 1) if all(!ismissing, view(Z, i, :))]
end

"""
    _require_observed(Z, what) -> Matrix

Return `Z` with a non-missing element type, or throw if any row has a missing value.
`what` names the procedure in the error message.
"""
function _require_observed(Z::AbstractMatrix, what::AbstractString)
    n_missing = size(Z, 1) - length(_observed_rows(Z))
    n_missing == 0 || throw(
        ArgumentError(
            "$what requires an instrument observed in every residual period; it is " *
                "missing in $n_missing of $(size(Z, 1)) periods."
        )
    )
    return convert(Matrix{nonmissingtype(eltype(Z))}, Z)
end

"""
    _iv_identify(ν, Z, target) -> (β_iv, F_stat, π_hat)

Two-stage least squares identification of one structural shock from VAR
residuals and an external instrument (Stock & Watson 2018). Rows in which `Z`
has a `missing` entry are dropped from both stages.

Single-shock identification via equation-by-equation 2SLS. For multiple-shock
identification (k proxies for k shocks), system GMM is required.

# Arguments
- `ν`: (T, K) VAR residuals
- `Z`: (T, k_z) instrument matrix, aligned with `ν`
- `target`: index of the shock to identify

# Returns
- `β_iv`: (K,) identified impact column (unit effect normalization: β_iv[target] = 1)
- `F_stat`: first-stage F-statistic
- `π_hat`: first-stage coefficients
"""
function _iv_identify(ν::AbstractMatrix{T}, Z::AbstractMatrix, target::Int) where {T}
    size(Z, 1) == size(ν, 1) || throw(
        DimensionMismatch(
            "instrument has $(size(Z, 1)) rows but the residuals have $(size(ν, 1))"
        )
    )
    rows = _observed_rows(Z)
    k_z = size(Z, 2)
    length(rows) > k_z + 1 || throw(
        ArgumentError(
            "instrument is observed in $(length(rows)) residual periods; 2SLS with " *
                "$k_z instrument(s) needs at least $(k_z + 2)"
        )
    )
    return _iv_identify_complete(Matrix{T}(ν[rows, :]), Matrix{T}(Z[rows, :]), target)
end

function _iv_identify_complete(ν::Matrix{T}, Z::Matrix{T}, target::Int) where {T}
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
(the deprecated `IVIdentification()`), take the instrument from an `IVSVAR` model.
"""
function _resolve_iv(model::VARModel, id::IVIdentification)
    if id.instrument !== nothing
        return id
    elseif model.spec isa IVSVAR
        return IVIdentification(model.spec.instrument)
    else
        throw(
            ArgumentError(
                "IVIdentification requires an instrument: use IVIdentification(Z, target_shock)"
            )
        )
    end
end

# ============================================================================
# Instrument extraction
# ============================================================================

"""
    _extract_instrument(instrument, n_obs, n_lags, names) -> (Z, target::Int)

Return the instrument rows aligned with the `n_obs` VAR residuals and the target
shock index (a `Symbol` target is resolved against `names`).

`Z` must have `n_obs + n_lags` rows (same length as the data `Y`; the first
`n_lags` rows are dropped) or `n_obs` rows (already aligned with the residuals).
"""
function _extract_instrument(
        instrument::ExternalInstrument,
        n_obs::Int, n_lags::Int, names::Vector{Symbol}
    )
    Z = instrument.Z
    n_z = size(Z, 1)
    target = _resolve_target(instrument.target_shock, names)
    if n_z == n_obs
        return Z, target
    elseif n_z == n_obs + n_lags
        return Z[(n_lags + 1):end, :], target
    else
        throw(
            DimensionMismatch(
                "Instrument has $n_z rows; expected $(n_obs + n_lags) (the number of rows of Y) " *
                    "or $n_obs (the number of VAR residuals, Y without its first $n_lags rows)."
            )
        )
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
# Matrix-level IV impact (for per-draw identification)
# ============================================================================

"""
    iv_impact(ν, Σ, Z, target::Int) -> (P, F_stat)

Structural impact matrix from external-instrument (proxy-SVAR) identification,
operating on raw matrices. `ν` are the reduced-form residuals (`n_obs × n_vars`),
`Σ` the residual covariance, `Z` the instrument aligned to `ν`, and `target` the
index of the instrumented shock. Rows of `Z` with a `missing` entry are dropped
from the 2SLS step; `Σ` is used as given.

The `target` column is identified by the instrument (Stock & Watson 2018,
Mertens & Ravn 2013); the remaining columns fill the orthogonal complement via
the Cholesky factor of `Σ - β β'`. `F_stat` is the first-stage F.

This is the matrix-level counterpart to `rotation_matrix(model, ::IVIdentification)`.
A package looping over reduced-form draws calls it with each draw's `(ν, Σ)` and
the fixed instrument.
"""
function iv_impact(
        ν::AbstractMatrix{T}, Σ::AbstractMatrix{T}, Z::AbstractMatrix,
        target::Int
    ) where {T}
    β_iv, F_stat, _ = _iv_identify(ν, Z, target)
    P = _build_full_impact_matrix(β_iv, target, Matrix(Σ))
    return P, F_stat
end

"""
    iv_instrument(id::IVIdentification, n_obs, n_lags, names) -> (Z, target::Int)

Resolve an `IVIdentification` to its aligned instrument matrix and target shock
index for `n_obs` residual rows. Wraps instrument extraction so a per-draw loop
can resolve `(Z, target)` once before iterating.
"""
function iv_instrument(id::IVIdentification, n_obs::Int, n_lags::Int, names::Vector{Symbol})
    id.instrument === nothing &&
        throw(ArgumentError("IVIdentification requires an instrument: use IVIdentification(Z, target_shock)"))
    return _extract_instrument(id.instrument, n_obs, n_lags, names)
end

# ============================================================================
# Bootstrap re-estimation helper
# ============================================================================

"""
    refit_for_bootstrap(model::VARModel{T, <:IVSVAR}, Y_boot, n_lags_val) where T

Re-estimate a proxy-SVAR model on bootstrap data. Used internally by
the bootstrap machinery.
"""
function refit_for_bootstrap(
        model::VARModel{T, <:IVSVAR}, Y_boot::Matrix{T},
        n_lags_val::Int
    ) where {T}
    constraints_arg = model.coefficients.constraints
    constraints_arg = constraints_arg === nothing ? AbstractConstraint[] : constraints_arg

    # Adjust instrument to match bootstrap residual count
    n_resid_boot = size(Y_boot, 1) - n_lags_val
    instrument = _resize_instrument(model.spec.instrument, n_resid_boot)

    return _fit_ivsvar(
        Y_boot, n_lags_val;
        instrument,
        constraints = constraints_arg,
        names = model.names
    )
end

"""
    refit_for_bootstrap(model::VARModel{T}, Y_boot, n_lags_val) where T

Re-estimate a non-IVSVAR model on bootstrap data. Generic fallback.
"""
function refit_for_bootstrap(
        model::VARModel{T}, Y_boot::Matrix{T},
        n_lags_val::Int
    ) where {T}
    constraints_arg = model.coefficients.constraints
    constraints_arg = constraints_arg === nothing ? AbstractConstraint[] : constraints_arg
    return fit(
        typeof(model.spec), Y_boot, n_lags_val;
        constraints = constraints_arg,
        names = model.names
    )
end
