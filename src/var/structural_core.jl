# ============================================================================
# Matrix-level structural identification core
# ============================================================================
#
# Low-level building blocks for structural identification that operate on raw
# coefficient/covariance matrices rather than a fitted `VARModel`. The model-
# level methods in `identification.jl` and `irfs.jl` delegate here, and
# packages that loop over many reduced-form draws (e.g. posterior draws from a
# Bayesian VAR) call these directly to avoid reconstructing a `VARModel` per
# draw.
#
# Conventions (shared with the rest of the package):
# - Impact matrix `P` maps structural shocks to reduced-form residuals: u = P ε.
# - `P[i, j]` is the impact of structural shock `j` on variable `i`.
# - Coefficient matrix `B` is `(1 + n_vars*n_lags) × n_vars`: row 1 is the
#   intercept; rows for lag k (k = 1..p) start at `2 + (k-1)*n_vars` and hold
#   `A_k` transposed, i.e. `A_k[i, j] == B[2 + (k-1)*n_vars + (j-1), i]`.
# - MA arrays `Φ` are `(n_vars, n_vars, horizon+1)` with `Φ[:, :, 1] = I`.

# ----------------------------------------------------------------------------
# Sign-restriction matrix normalization (+/- function literals → ±1)
# ----------------------------------------------------------------------------

"""
    _normalize_sign_matrix(m) -> Matrix{Int}

Normalize a sign restriction matrix, converting the `+` and `-` functions to
`+1` and `-1`. Integer matrices pass through after validation.

When using `+`/`-` in a matrix literal, wrap them in parentheses to avoid
parsing issues, e.g. `[(+) 0; (-) (+)]`.
"""
function _normalize_sign_matrix(m::AbstractMatrix)
    # `copyto!` drops any offset axes of `m`; the result is always 1-based.
    return copyto!(Matrix{Int}(undef, size(m)), map(_sign_entry, m))
end

_sign_entry(::typeof(+)) = 1
_sign_entry(::typeof(-)) = -1
function _sign_entry(val::Integer)
    val in (-1, 0, 1) ||
        throw(ArgumentError("Sign restriction values must be -1, 0, or +1, got $val"))
    return Int(val)
end
function _sign_entry(val)
    throw(ArgumentError("Invalid sign restriction element: $val. Use +, -, 0, 1, or -1"))
end

# ----------------------------------------------------------------------------
# MA coefficients directly from B (no companion matrix)
# ----------------------------------------------------------------------------

"""
    compute_ma_matrices_from_B!(Φ, B, temp, horizon, n_vars, n_lags) -> Φ

Compute moving-average coefficient matrices into `Φ` directly from the VAR
coefficient matrix `B`, using the recursion `Φ_h = Σₖ A_k Φ_{h-k}`.

This is O(n³p·h), avoiding the O(n³p³) cost of forming and powering the
companion matrix. `temp` is an `(n_vars, n_vars)` work buffer. `Φ` must be
`(n_vars, n_vars, H)` with `H ≥ horizon+1`; slices `1:horizon+1` are written,
with `Φ[:, :, 1]` set to the identity.

See the file header for the `B` layout.
"""
function compute_ma_matrices_from_B!(
        Φ::Array{T, 3}, B::AbstractMatrix{T}, temp::Matrix{T},
        horizon::Int, n_vars::Int, n_lags::Int
) where {T}
    Base.require_one_based_indexing(B)
    horizon >= 0 || throw(ArgumentError("horizon must be non-negative, got $horizon"))
    size(B) == (1 + n_vars * n_lags, n_vars) || throw(
        DimensionMismatch(
        "B must be $(1 + n_vars * n_lags)×$n_vars for n_vars=$n_vars, n_lags=$n_lags; got $(size(B))"
    )
    )
    (size(Φ, 1), size(Φ, 2)) == (n_vars, n_vars) && size(Φ, 3) > horizon || throw(
        DimensionMismatch(
        "Φ must be $n_vars×$n_vars×H with H ≥ $(horizon + 1); got $(size(Φ))"
    )
    )
    size(temp) == (n_vars, n_vars) || throw(
        DimensionMismatch("temp must be $n_vars×$n_vars; got $(size(temp))")
    )

    @inbounds for j in 1:n_vars
        for i in 1:n_vars
            Φ[i, j, 1] = i == j ? one(T) : zero(T)
        end
    end

    @inbounds for h in 1:horizon
        for j in 1:n_vars
            for i in 1:n_vars
                Φ[i, j, h + 1] = zero(T)
            end
        end

        for k in 1:min(h, n_lags)
            row_start = 2 + (k - 1) * n_vars
            Φ_prev = view(Φ, :, :, (h - k + 1))

            # temp = A_k * Φ_{h-k}, where A_k[i, j] = B[row_start + j - 1, i]
            for m in 1:n_vars
                for i in 1:n_vars
                    s = zero(T)
                    for j in 1:n_vars
                        s += B[row_start + j - 1, i] * Φ_prev[j, m]
                    end
                    temp[i, m] = s
                end
            end

            for j in 1:n_vars
                for i in 1:n_vars
                    Φ[i, j, h + 1] += temp[i, j]
                end
            end
        end
    end

    return Φ
end

# ----------------------------------------------------------------------------
# Random orthogonal matrices (in-place)
# ----------------------------------------------------------------------------

"""
    generate_random_orthogonal!(Q_out, X_buffer, rng) -> Q_out

Fill `Q_out` with a random orthogonal matrix, using `X_buffer` (same size) for
the random normal draw. The result is normalized by the signs of the `R`
diagonal so the mapping from the normal draw to `Q` is deterministic.
"""
function generate_random_orthogonal!(
        Q_out::AbstractMatrix{T}, X_buffer::AbstractMatrix{T},
        rng::AbstractRNG
) where {T}
    n = size(Q_out, 1)
    randn!(rng, X_buffer)
    qr_fact = qr(X_buffer)
    @inbounds for j in 1:n
        s = sign(qr_fact.R[j, j])
        for i in 1:n
            Q_out[i, j] = qr_fact.Q[i, j] * s
        end
    end
    return Q_out
end

# ----------------------------------------------------------------------------
# Sign-restriction and narrative checks (matrix level, precomputed Φ)
# ----------------------------------------------------------------------------

"""
    check_sign_restrictions(P, restrictions, Φ, horizon) -> Bool

Check whether impact matrix `P` satisfies sign restrictions, given precomputed
MA matrices `Φ` (`(n_vars, n_vars, H)` with `H ≥ horizon+1`). Restrictions are
`+1`/`-1`/`0` on `[response, shock]`; impact (h=0) is `P` itself, later horizons
are `Φ_h P`.
"""
function check_sign_restrictions(
        P::AbstractMatrix{T}, restrictions::Matrix{Int},
        Φ::Array{T, 3}, horizon::Int
) where {T}
    Base.require_one_based_indexing(P)
    n_vars = LinearAlgebra.checksquare(P)
    size(restrictions) == (n_vars, n_vars) || throw(
        DimensionMismatch(
        "restrictions must be $n_vars×$n_vars; got $(size(restrictions))"
    )
    )
    size(Φ, 3) > horizon || throw(
        DimensionMismatch(
        "Φ holds horizons 0:$(size(Φ, 3) - 1); restrictions need 0:$horizon"
    )
    )

    for i in 1:n_vars
        for j in 1:n_vars
            if restrictions[i, j] == 1 && P[i, j] < 0
                return false
            elseif restrictions[i, j] == -1 && P[i, j] > 0
                return false
            end
        end
    end

    if horizon > 0
        for h in 1:horizon
            IRF_h = Φ[:, :, h + 1] * P
            for i in 1:n_vars
                for j in 1:n_vars
                    if restrictions[i, j] == 1 && IRF_h[i, j] < 0
                        return false
                    elseif restrictions[i, j] == -1 && IRF_h[i, j] > 0
                        return false
                    end
                end
            end
        end
    end

    return true
end

"""
    compute_structural_shocks(U, P) -> ε

Recover structural shocks `ε` from reduced-form residuals `U` (`n_obs × n_vars`)
and impact matrix `P`, via `u_t = P ε_t` ⇒ `ε_t = P⁻¹ u_t`. Returns an
`n_obs × n_vars` matrix.
"""
function compute_structural_shocks(U::AbstractMatrix{T}, P::AbstractMatrix{T}) where {T}
    return Matrix((P \ U')')
end

"""
    check_narrative_restrictions(ε, narrative_shocks, dates=nothing) -> Bool

Check whether structural shocks `ε` (`n_obs × n_vars`) satisfy the narrative
sign restrictions. `dates` resolves date-based restrictions to row indices; it
must be supplied when any restriction carries a `Dates.TimeType` date.
"""
function check_narrative_restrictions(
        ε::AbstractMatrix{T},
        narrative_shocks::Vector{<:NarrativeShockRestriction},
        dates = nothing
) where {T}
    for nr in narrative_shocks
        date_idx = _resolve_narrative_date(nr, dates)
        shock_value = ε[date_idx, nr.shock]
        if nr.sign == 1 && shock_value < 0
            return false
        elseif nr.sign == -1 && shock_value > 0
            return false
        end
    end
    return true
end

"""
    _resolve_narrative_date(nr::NarrativeShockRestriction, dates) -> Int

Resolve a narrative restriction's date to a row index. Integer dates pass
through; `Dates.TimeType` dates are looked up in `dates`.
"""
_resolve_narrative_date(nr::NarrativeShockRestriction{Int}, dates) = nr.date

function _resolve_narrative_date(
        nr::NarrativeShockRestriction{D},
        dates
) where {D <: Dates.TimeType}
    dates === nothing &&
        throw(
            ArgumentError(
            "narrative restriction dated $(nr.date) needs a model fitted with `dates`; " *
            "fit with `dates = ...` or give the date as a row index"
        )
        )
    idx = findfirst(==(nr.date), dates)
    idx === nothing && throw(
        ArgumentError(
        "narrative restriction date $(nr.date) not found in the model dates " *
        "($(first(dates)) to $(last(dates)))"
    )
    )
    return idx
end

# ----------------------------------------------------------------------------
# Matrix-level structural impact (Cholesky)
# ----------------------------------------------------------------------------

"""
    structural_impact(Σ_L, id::CholeskyID; ordering_idx=id.ordering) -> P

Cholesky impact matrix from an already-computed lower Cholesky factor `Σ_L`
(where `Σ = Σ_L Σ_L'`). Without an ordering, returns `Σ_L` itself.

With an ordering, a permutation of the variables, returns the lower Cholesky
factor of `Σ[ordering_idx, ordering_idx]` with its rows and columns mapped back
to the original variable order, so that column `j` is the shock of variable `j`
and `P * P' ≈ Σ`.

`ordering_idx` defaults to an integer `id.ordering`. A `Symbol` ordering needs
the variable names, so callers resolve it to indices and pass `ordering_idx`;
an integer `id.ordering` that differs from `ordering_idx` is an error.

`rotation_matrix(model, ::CholeskyID)` delegates here with `Σ_L` from the
model's residual covariance.
"""
function structural_impact(
        Σ_L::AbstractMatrix{T}, id::CholeskyID;
        ordering_idx::Union{Nothing, AbstractVector{Int}} = _ordering_index(id)
) where {T}
    if id.ordering isa Vector{Int} && ordering_idx != id.ordering
        throw(
            ArgumentError(
            "ordering_idx = $(ordering_idx) contradicts CholeskyID ordering $(id.ordering)"
        )
        )
    end
    if id.ordering isa Vector{Symbol} && ordering_idx === nothing
        throw(
            ArgumentError(
            "CholeskyID ordering $(id.ordering) is given by name; pass ordering_idx with the matching variable indices"
        )
        )
    end
    ordering_idx === nothing && return Σ_L
    Base.require_one_based_indexing(Σ_L, ordering_idx)
    n = LinearAlgebra.checksquare(Σ_L)
    length(ordering_idx) == n || throw(
        ArgumentError(
        "ordering_idx must be a permutation of 1:$n; got $(ordering_idx)"
    )
    )
    inv_perm = invperm(ordering_idx)
    L_rows = Σ_L[ordering_idx, :]
    P_ordered = cholesky(Symmetric(L_rows * L_rows')).L
    return P_ordered[inv_perm, inv_perm]
end

_ordering_index(id::CholeskyID) = id.ordering isa Vector{Int} ? id.ordering : nothing
