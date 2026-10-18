# ============================================================================
# Core Type Hierarchy
# ============================================================================

# First, define abstract types that will be used as type parameters

"""
    AbstractInstrument

Abstract type for instrument specifications in IV-SVAR models.
"""
abstract type AbstractInstrument end

"""
    AbstractVARSpec

Abstract type for VAR model specifications.

Concrete subtypes include:
- `OLSVAR`: Ordinary least squares estimation
- `IVSVAR`: Instrumental variable structural VAR (deprecated; use `OLSVAR`
  with `IVIdentification`)
"""
abstract type AbstractVARSpec end

"""
    OLSVAR <: AbstractVARSpec

Ordinary least squares VAR estimation specification.
"""
struct OLSVAR <: AbstractVARSpec end

"""
    IVSVAR{I} <: AbstractVARSpec

Instrumental Variable Structural VAR specification. Deprecated: fit the model
with `OLSVAR` and pass the instrument to `IVIdentification`.

# Fields
- `instrument::I`: Instrument specification (external, proxy, high-frequency)
"""
struct IVSVAR{I <: AbstractInstrument} <: AbstractVARSpec
    instrument::I
end

# ============================================================================
# Instrument Types (for IV-SVAR)
# ============================================================================

"""
    ExternalInstrument{T, S} <: AbstractInstrument

External instrument (proxy) for one structural shock, used by `IVIdentification`
(Stock & Watson 2018, Mertens & Ravn 2013). The impact column is estimated by
two-stage least squares on the VAR residuals.

# Fields
- `Z::Matrix{Union{Missing, T}}`: Instrument matrix (rows are time periods, columns
  are instruments); `missing` marks periods in which the instrument is not observed
- `target_shock::S`: Shock to identify — `Int` (index) or `Symbol` (variable name)

# Constructors
```julia
ExternalInstrument(Z)                           # target_shock = 1
ExternalInstrument(Z; target_shock = :logIP)    # by variable name
ExternalInstrument(Z, 4)                        # by index
ExternalInstrument(Z, :logIP)                   # by variable name
```

`Z` may be a vector (one instrument) or a matrix. A `Symbol` target is resolved
against the model's variable names when the instrument is used.

# Row alignment
`Z` must have either as many rows as the data `Y` the VAR was fitted on (the first
`n_lags` rows are then dropped so that `Z` lines up with the residuals) or as many
rows as the residuals. Any other length throws a `DimensionMismatch`.

# Partially observed instruments
The instrument may cover a shorter sample than the VAR: set `Z` to `missing` in
the periods where it is not observed. The reduced-form VAR (coefficients and
residual covariance) uses the full sample; the 2SLS estimate of the impact
column and the first-stage F use only the residual periods in which every column
of `Z` is observed. `NaN` and `Inf` entries are rejected. Bootstrap inference
(`ProxySVARMBB`) and `msw_confidence_set` require an instrument observed in every
residual period.
"""
struct ExternalInstrument{T <: AbstractFloat, S <: Union{Int, Symbol}} <: AbstractInstrument
    Z::Matrix{Union{Missing, T}}
    target_shock::S

    function ExternalInstrument{T, S}(Z, target_shock) where {T, S}
        if target_shock isa Integer
            target_shock > 0 || throw(ArgumentError("target_shock must be positive"))
        end
        all(z -> ismissing(z) || isfinite(z), Z) ||
            throw(
                ArgumentError(
                "instrument contains NaN or Inf; mark periods " *
                "without an instrument value with `missing`"
            )
            )
        return new{T, S}(Z, target_shock)
    end
end

# ============================================================================
# Constraint abstract type (concrete subtypes defined in constraints.jl)
# ============================================================================

"""
    AbstractConstraint

Abstract type for linear constraints on VAR coefficients.
Concrete subtypes: `ZeroConstraint`, `FixedConstraint`, `BlockExogeneity`.
"""
abstract type AbstractConstraint end

# ============================================================================
# VAR Coefficient Structure
# ============================================================================

"""
    VARCoefficients{T}

Storage structure for VAR coefficients with constraint information.

# Fields
- `intercept::Vector{T}`: Intercept coefficients (n_vars)
- `lags::Array{T,3}`: Lag coefficients (n_vars, n_vars, n_lags)
- `constraints::Union{Nothing, Vector{AbstractConstraint}}`: Applied constraints
  (`nothing` for an unconstrained fit)
"""
struct VARCoefficients{T <: AbstractFloat}
    intercept::Vector{T}
    lags::Array{T, 3}
    constraints::Union{Nothing, Vector{AbstractConstraint}}
end

function VARCoefficients(
        intercept::AbstractVector, lags::AbstractArray{<:Any, 3},
        constraints = nothing
)
    T = promote_type(eltype(intercept), eltype(lags))
    return VARCoefficients{T}(intercept, lags, constraints)
end

# ============================================================================
# VAR Model Structure
# ============================================================================

"""
    VARModel{T, S, D}

Vector Autoregression model.

# Type Parameters
- `T<:AbstractFloat`: Numeric type for computations
- `S<:AbstractVARSpec`: Model specification type
- `D`: Type of the `dates` field (`Nothing` or an `AbstractVector`)

# Fields
- `spec::S`: Model specification
- `Y::Matrix{T}`: Original data matrix (T × n_vars)
- `X::Matrix{T}`: Lagged data matrix for estimation
- `coefficients::VARCoefficients{T}`: Estimated coefficients
- `residuals::Matrix{T}`: Residuals (T-p × n_vars)
- `Σ::Symmetric{T,Matrix{T}}`: Residual covariance matrix
- `companion::Matrix{T}`: Companion form matrix
- `names::Vector{Symbol}`: Variable names
- `dates::D`: Dates of the residual rows, or `nothing`; see [`dates`](@ref)
- `metadata::NamedTuple`: Fit information not read by the package:
  `n_obs_total`, `n_obs_used`, `demean`, `timestamp`
"""
struct VARModel{
    T <: AbstractFloat, S <: AbstractVARSpec, D <: Union{Nothing, AbstractVector}
}
    spec::S
    Y::Matrix{T}
    X::Matrix{T}
    coefficients::VARCoefficients{T}
    residuals::Matrix{T}
    Σ::Symmetric{T, Matrix{T}}
    companion::Matrix{T}
    names::Vector{Symbol}
    dates::D
    metadata::NamedTuple

    function VARModel{T, S, D}(
            spec, Y, X, coefficients, residuals, Σ, companion,
            names, dates, metadata
    ) where {T, S, D}
        if dates !== nothing && length(dates) != size(residuals, 1)
            throw(
                DimensionMismatch(
                "dates must have one entry per residual row " *
                "(got $(length(dates)) for $(size(residuals, 1)) rows)"
            )
            )
        end
        return new{T, S, D}(
            spec, Y, X, coefficients, residuals, Σ, companion,
            names, dates, metadata
        )
    end
end

function VARModel(
        spec::AbstractVARSpec, Y::AbstractMatrix, X, coefficients, residuals,
        Σ, companion, names, dates, metadata
)
    return VARModel{eltype(Y), typeof(spec), typeof(dates)}(
        spec, Y, X, coefficients,
        residuals, Σ, companion, names, dates, metadata
    )
end

# ============================================================================
# Identification Schemes
# ============================================================================

"""
    AbstractIdentification

Abstract type for identification schemes in structural VAR analysis.
"""
abstract type AbstractIdentification end

"""
    CholeskyID <: AbstractIdentification

Cholesky (recursive) identification scheme.

# Fields
- `ordering::Union{Nothing,Vector{Symbol},Vector{Int}}`: Variable ordering
  (nothing = use data order). Symbols order by variable name (resolved against
  the model); integers order by position and must be a permutation of
  `1:length(ordering)`. Names and positions must be unique; whether they match
  the model's variables is checked when the model is identified.
"""
struct CholeskyID <: AbstractIdentification
    ordering::Union{Nothing, Vector{Symbol}, Vector{Int}}

    function CholeskyID(ordering)
        return new(_check_cholesky_ordering(ordering))
    end
end

_check_cholesky_ordering(::Nothing) = nothing

function _check_cholesky_ordering(ordering::AbstractVector{Symbol})
    isempty(ordering) && throw(ArgumentError("CholeskyID ordering must not be empty"))
    allunique(ordering) ||
        throw(ArgumentError("CholeskyID ordering has repeated variable names: $ordering"))
    return Vector{Symbol}(ordering)
end

function _check_cholesky_ordering(ordering::AbstractVector{<:Integer})
    isempty(ordering) && throw(ArgumentError("CholeskyID ordering must not be empty"))
    sort(ordering) == 1:length(ordering) || throw(
        ArgumentError(
        "CholeskyID integer ordering must be a permutation of 1:$(length(ordering)), got $ordering"
    )
    )
    return Vector{Int}(ordering)
end

function _check_cholesky_ordering(ordering)
    throw(
        ArgumentError(
        "CholeskyID ordering must be `nothing`, a vector of variable names (Symbol), " *
        "or a vector of integer positions; got $(repr(ordering))"
    )
    )
end

CholeskyID() = CholeskyID(nothing)

"""
    SignRestriction <: AbstractIdentification

Sign restriction identification scheme.

    SignRestriction(restrictions, horizon = 0)

# Fields
- `restrictions::Matrix{Int}`: Square sign restriction matrix, variables in rows
  and shocks in columns; `+1` (positive), `-1` (negative), `0` (unrestricted).
  The constructor also accepts the functions `+` and `-` as entries.
- `horizon::Int`: Restrictions apply at horizons `0:horizon` (`0` = impact only)
"""
struct SignRestriction <: AbstractIdentification
    restrictions::Matrix{Int}
    horizon::Int

    function SignRestriction(restrictions, horizon)
        R = _normalize_sign_matrix(restrictions)
        _check_sign_restriction_args(R, horizon)
        return new(R, horizon)
    end
end

SignRestriction(restrictions) = SignRestriction(restrictions, 0)

function _check_sign_restriction_args(R::AbstractMatrix, horizon)
    size(R, 1) == size(R, 2) || throw(
        ArgumentError("sign restriction matrix must be square (variables × shocks), got size $(size(R))")
    )
    horizon >= 0 ||
        throw(ArgumentError("sign restriction horizon must be ≥ 0, got $horizon"))
    return nothing
end

"""
    IVIdentification{I} <: AbstractIdentification

Identification via external instruments, also known as SVAR-IV or proxy SVAR
(Stock & Watson 2018, Mertens & Ravn 2013 — same estimator, different names).

```julia
id = IVIdentification(proxy_vector, target_shock)
result = irf(model, id; horizon=20)
```

# Constructors
- `IVIdentification(Z, target_shock)` — `Z` is a vector or matrix; `target_shock`
  is an index or a variable name
- `IVIdentification(Z; target_shock = 1)`
- `IVIdentification(instrument::ExternalInstrument)`

See [`ExternalInstrument`](@ref) for the row-alignment rule on `Z`.
`IVIdentification()` (no instrument, taking it from a model fitted with
`fit(IVSVAR, ...)`) is deprecated.
"""
struct IVIdentification{I <: Union{Nothing, AbstractInstrument}} <: AbstractIdentification
    instrument::I
end

function IVIdentification()
    Base.depwarn(
        "`IVIdentification()` is deprecated; pass the instrument, " *
        "as in `IVIdentification(Z, target_shock)`.",
        :IVIdentification
    )
    return IVIdentification{Nothing}(nothing)
end

function IVIdentification(Z::AbstractVecOrMat; target_shock::Union{Integer, Symbol} = 1)
    return IVIdentification(ExternalInstrument(Z, target_shock))
end

function IVIdentification(Z::AbstractVecOrMat, target_shock::Union{Integer, Symbol})
    return IVIdentification(ExternalInstrument(Z, target_shock))
end

IVIdentification(inst::AbstractInstrument) = IVIdentification{typeof(inst)}(inst)

# ============================================================================
# Inference Types
# ============================================================================

"""
    InferenceType

Abstract type for statistical inference methods used to compute confidence bands for IRFs.

Concrete subtypes:
- `Analytic`: Asymptotic inference using delta method (Lütkepohl)
- `WildBootstrap`: Wild bootstrap with Rademacher weights
- `Bootstrap`: Standard i.i.d. bootstrap
- `BlockBootstrap`: Moving block bootstrap for time series dependence
"""
abstract type InferenceType end

"""
    Analytic <: InferenceType

Asymptotic inference using the delta method.

Based on Lütkepohl (2005) analytical formulas for IRF standard errors.
Supported only for `CholeskyID()` in the model's variable order; any other
identification throws an `ArgumentError`.

# Example
```julia
irf(model, CholeskyID(); inference=Analytic())
```
"""
struct Analytic <: InferenceType end

"""
    WildBootstrap <: InferenceType

Wild bootstrap inference using Rademacher weights.

Resamples residuals by multiplying with random ±1 weights, preserving
conditional heteroskedasticity while maintaining independence across equations.

# Fields
- `reps::Int`: Number of bootstrap replications
- `save_draws::Bool`: Whether to save all bootstrap IRF draws
- `max_failure_rate::Float64`: Largest fraction of failed replications tolerated
  (default 0.1); above it the call throws. Failed replications are excluded from
  bands and standard errors.

# References
- Liu (1988): "Bootstrap Procedures under Some Non-I.I.D. Models"
- Gonçalves and Kilian (2004): "Bootstrapping autoregressions with
  conditional heteroskedasticity of unknown form"

# Example
```julia
# Save draws for post-hoc band computation
irf(model, id; inference=WildBootstrap(reps=1000, save_draws=true))

# Memory-efficient: don't save draws
irf(model, id; inference=WildBootstrap(reps=1000, save_draws=false))
```
"""
struct WildBootstrap <: InferenceType
    reps::Int
    save_draws::Bool
    max_failure_rate::Float64

    function WildBootstrap(reps::Int = 1000; save_draws::Bool = false, max_failure_rate::Real = 0.1)
        reps > 0 || throw(ArgumentError("reps must be positive"))
        return new(reps, save_draws, _check_max_failure_rate(max_failure_rate))
    end
end

"""
    Bootstrap <: InferenceType

Standard i.i.d. bootstrap inference.

Resamples residuals with replacement, treating them as independent draws.
Appropriate when residuals can be assumed i.i.d. (homoskedastic and uncorrelated).

# Fields
- `reps::Int`: Number of bootstrap replications
- `save_draws::Bool`: Whether to save all bootstrap IRF draws
- `max_failure_rate::Float64`: Largest fraction of failed replications tolerated
  (default 0.1); above it the call throws. Failed replications are excluded from
  bands and standard errors.

# References
- Efron (1979): "Bootstrap methods: Another look at the jackknife"
- Freedman (1981): "Bootstrapping regression models"

# Note
For time series with dependence or heteroskedasticity, wild bootstrap or
block bootstrap may be more appropriate.

# Example
```julia
irf(model, id; inference=Bootstrap(reps=1000, save_draws=true))
```
"""
struct Bootstrap <: InferenceType
    reps::Int
    save_draws::Bool
    max_failure_rate::Float64

    function Bootstrap(reps::Int = 1000; save_draws::Bool = false, max_failure_rate::Real = 0.1)
        reps > 0 || throw(ArgumentError("reps must be positive"))
        return new(reps, save_draws, _check_max_failure_rate(max_failure_rate))
    end
end

"""
    BlockBootstrap <: InferenceType

Moving block bootstrap for time series with temporal dependence.

Resamples blocks of consecutive residuals to preserve temporal dependence structure.
Uses position-specific centering to ensure resampled residuals have approximately zero mean.

# Fields
- `reps::Int`: Number of bootstrap replications
- `block_length::Int`: Length of each block (rule of thumb: T^(1/3))
- `save_draws::Bool`: Whether to save all bootstrap IRF draws
- `max_failure_rate::Float64`: Largest fraction of failed replications tolerated
  (default 0.1); above it the call throws. Failed replications are excluded from
  bands and standard errors.

# References
- Künsch (1989): "The jackknife and the bootstrap for general stationary observations"
- Carlstein (1986): "The use of subseries values for estimating the variance of
  a general statistic from a stationary sequence"
- Paparoditis and Politis (2001): "Tapered block bootstrap"

# Block Length Selection
Rule of thumb: ℓ ≈ T^(1/3) for moderate dependence.
For stronger persistence, use larger blocks (e.g., ℓ = 10-20 for quarterly data).

# Example
```julia
irf(model, id; inference=BlockBootstrap(reps=1000, block_length=15, save_draws=true))
```
"""
struct BlockBootstrap <: InferenceType
    reps::Int
    block_length::Int
    save_draws::Bool
    max_failure_rate::Float64

    function BlockBootstrap(
            reps::Int = 1000; block_length::Int = 10, save_draws::Bool = false,
            max_failure_rate::Real = 0.1
    )
        reps > 0 || throw(ArgumentError("reps must be positive"))
        block_length > 0 || throw(ArgumentError("block_length must be positive"))
        return new(reps, block_length, save_draws, _check_max_failure_rate(max_failure_rate))
    end
end

# Bootstraps that resample the VAR residuals and refit the model.
const _ResidualBootstrap = Union{WildBootstrap, Bootstrap, BlockBootstrap}

"""
    ProxySVARMBB <: InferenceType

Jentsch & Lunsford (2022) corrected moving block bootstrap for proxy-SVARs.

Jointly resamples (residuals, proxy) in blocks with position-specific centering.
This is asymptotically valid under heteroskedasticity and serial correlation,
unlike the standard wild or block bootstrap applied to proxy-SVARs.

Optionally computes Anderson-Rubin confidence sets (robust to weak instruments)
via grid-based test inversion.

# Fields
- `reps::Int`: Number of bootstrap replications (default: 2000)
- `block_length::Int`: Block size ℓ (default: 4)
- `compute_ar::Bool`: Compute Anderson-Rubin confidence sets (default: false)
- `ar_grid::Union{Nothing, Vector{Float64}}`: Grid for AR sets
- `norm_scale::Float64`: Normalization scale for normalized IRFs (default: -1.0)
- `save_draws::Bool`: Save all bootstrap draws
- `max_failure_rate::Float64`: Largest fraction of failed replications tolerated
  (default 0.1); above it the call throws. Failed replications are excluded from
  bands and standard errors.

# References
- Jentsch & Lunsford (2022): "Asymptotically Valid Bootstrap Inference for
  Proxy SVARs"
- Anderson & Rubin (1949): "Estimation of the parameters of a single equation
  in a complete system of stochastic equations"

# Example
```julia
model = fit(OLSVAR, Y, p)
result = irf(model, IVIdentification(Z, 1);
    inference=ProxySVARMBB(2000; block_length=4))
```
"""
struct ProxySVARMBB <: InferenceType
    reps::Int
    block_length::Int
    compute_ar::Bool
    ar_grid::Union{Nothing, Vector{Float64}}
    norm_scale::Float64
    save_draws::Bool
    max_failure_rate::Float64

    function ProxySVARMBB(
            reps::Int = 2000;
            block_length::Int = 4,
            compute_ar::Bool = false,
            ar_grid::Union{Nothing, Vector{Float64}} = nothing,
            norm_scale::Float64 = -1.0,
            save_draws::Bool = false,
            max_failure_rate::Real = 0.1
    )
        reps > 0 || throw(ArgumentError("reps must be positive"))
        block_length > 0 || throw(ArgumentError("block_length must be positive"))
        # When compute_ar=true and ar_grid=nothing, a default grid [-10, 10] × 201
        # is applied at runtime in proxy_svar_mbb.
        return new(
            reps, block_length, compute_ar, ar_grid, norm_scale, save_draws,
            _check_max_failure_rate(max_failure_rate)
        )
    end
end

# ============================================================================
# IRF Structure
# ============================================================================

"""
    AbstractIRFResult{T}

Abstract supertype for all impulse response function result types.

Concrete subtypes:
- `IRFResult{T}`: Point-identified IRF results (Cholesky, IV, etc.)
- `SignRestrictedIRFResult{T}`: Set-identified IRF results (sign restrictions)
- `BayesianIRFResult{T}`: Posterior draws (produced by BayesianVAR.jl)
- `LocalProjectionIRFResult{T}`: Local projection estimates (produced by LocalProjections.jl)

# Interface

Accessors, `show`, and plotting (`irfplot`, `irfplot!`, and the Plots recipe)
use only the methods below, so a new subtype works with them once it
implements the required methods.

Required:

| Method                | Returns                                                         |
|:----------------------|:----------------------------------------------------------------|
| `point_estimate(irf)` | array indexed `(variable, shock, horizon)`, horizons `0:H`      |
| `lowerbounds(irf)`    | vector of arrays shaped like `point_estimate`, one per coverage |
| `upperbounds(irf)`    | vector of arrays shaped like `point_estimate`, one per coverage |
| `coverages(irf)`      | vector of coverage levels, in the order of the bands            |
| `varnames(irf)`       | vector of variable names (first dimension)                      |
| `shocknames(irf)`     | vector of shock names (second dimension)                        |

Optional (defaults in parentheses):

| Method           | Returns                                                          |
|:-----------------|:-----------------------------------------------------------------|
| `has_draws(irf)` | whether draws are stored (`false`)                               |
| `get_draws(irf)` | array indexed `(draw, variable, shock, horizon)` (throws)        |
| `n_draws(irf)`   | number of draws (`size(get_draws(irf), 1)`, or `0` without draws) |

Derived from the above: `horizon`, `n_vars`, `n_shocks`, `size`, and `show`.
A subtype may override `horizon` when `point_estimate` is expensive to compute.

Accessors that return stored data return it without copying. Axis names
(`:variable`, `:response`, ...) are not part of the interface; the dimension
order is.
"""
abstract type AbstractIRFResult{T <: AbstractFloat} end

"""
    IRFResult{T, A<:AxisArray}

Impulse response function results for point-identified systems.

Data is stored as AxisArray with dimensions:
- `:variable` - response variable (Symbol)
- `:shock` - structural shock (Symbol)
- `:horizon` - IRF horizon (0:H)

# Example Access
```julia
irf.irf[:GDP, :MonetaryShock, 0:12]  # GDP response to monetary shock, horizons 0-12
```

The trailing type parameters `D`, `I`, `N` are the types of `bootstrap_draws`,
`identification`, and `inference`; the constructors infer them.

# Fields
- `irf::A`: 3D AxisArray (variable × shock × horizon) — point estimates
- `stderr::A`: Standard errors (same shape)
- `bootstrap_draws::D`: Bootstrap IRF draws (draw × variable × shock × horizon), or `nothing`
- `lower::Vector{A}`: Lower confidence bands (one per coverage level)
- `upper::Vector{A}`: Upper confidence bands
- `coverage::Vector{Float64}`: Coverage levels
- `identification::I`: Identification scheme used
- `inference::N`: Inference method used, or `nothing`
- `metadata::NamedTuple`: Additional information; see below

# Metadata
- `horizon`, `names`, `timestamp`, `inference_type`: descriptive
- `normalization` (type), `scale` (per-shock vector), `impact_diagonal`: read by
  [`get_scale`](@ref) and updated by `rescale`
- `cumulated_vars`: `nothing`, or the names of the cumulated variables; set by the
  `cumulate` keyword of `irf` and by [`cumulative_irf`](@ref); shown by `show`
"""
struct IRFResult{
    T <: AbstractFloat, A <: AxisArray, D <: Union{Nothing, AxisArray},
    I <: AbstractIdentification, N <: Union{Nothing, InferenceType}
} <: AbstractIRFResult{T}
    irf::A
    stderr::A
    bootstrap_draws::D
    lower::Vector{A}
    upper::Vector{A}
    coverage::Vector{Float64}
    identification::I
    inference::N
    metadata::NamedTuple
end

function IRFResult(
        irf::AxisArray, stderr, bootstrap_draws, lower, upper, coverage,
        identification, inference, metadata
)
    return IRFResult{eltype(irf), typeof(irf)}(
        irf, stderr, bootstrap_draws, lower, upper,
        coverage, identification, inference, metadata
    )
end

function IRFResult{T, A}(
        irf, stderr, bootstrap_draws, lower, upper, coverage,
        identification, inference, metadata
) where {T, A}
    return IRFResult{
        T, A, typeof(bootstrap_draws), typeof(identification), typeof(inference)}(
        irf, stderr, bootstrap_draws, lower, upper, coverage, identification, inference,
        metadata
    )
end

"""
    SignRestrictedIRFResult{T, A<:AxisArray}

Impulse response function results for sign restriction identification (set-identified).

Data is stored as AxisArray with dimensions:
- `:variable` - response variable (Symbol)
- `:shock` - structural shock (Symbol)
- `:horizon` - IRF horizon (0:H)

# Example Access
```julia
irf.irf_median[:GDP, :MonetaryShock, 0:12]  # Median GDP response, horizons 0-12
```

The trailing type parameters `D` and `I` are the types of `irf_draws` and
`identification`; the constructors infer them.

# Fields
- `irf_median::A`: Median IRF across draws — 3D AxisArray (variable × shock × horizon)
- `irf_draws::D`: All drawn IRFs — 4D AxisArray (draw × variable × shock × horizon)
- `lower::Vector{A}`: Pointwise lower quantile bands
- `upper::Vector{A}`: Pointwise upper quantile bands
- `coverage::Vector{Float64}`: Coverage levels for quantile bands
- `rotation_matrices::Vector{Matrix{T}}`: All rotation matrices satisfying restrictions
- `identification::I`: Identification scheme used (sign or narrative restrictions)
- `metadata::NamedTuple`: Additional information; the keys are those of
  [`IRFResult`](@ref) with `n_draws` in place of `inference_type`
"""
struct SignRestrictedIRFResult{
    T <: AbstractFloat, A <: AxisArray, D <: AxisArray,
    I <: AbstractIdentification
} <: AbstractIRFResult{T}
    irf_median::A
    irf_draws::D
    lower::Vector{A}
    upper::Vector{A}
    coverage::Vector{Float64}
    rotation_matrices::Vector{Matrix{T}}
    identification::I
    metadata::NamedTuple
end

function SignRestrictedIRFResult(
        irf_median::AxisArray, irf_draws, lower, upper, coverage,
        rotation_matrices, identification, metadata
)
    return SignRestrictedIRFResult{eltype(irf_median), typeof(irf_median)}(
        irf_median, irf_draws,
        lower, upper, coverage, rotation_matrices, identification, metadata
    )
end

function SignRestrictedIRFResult{T, A}(
        irf_median, irf_draws, lower, upper, coverage,
        rotation_matrices, identification, metadata
) where {T, A}
    return SignRestrictedIRFResult{T, A, typeof(irf_draws), typeof(identification)}(
        irf_median,
        irf_draws, lower, upper, coverage, rotation_matrices, identification, metadata
    )
end

# ============================================================================
# AxisArray-based IRF Result Types (Hub types for Spoke integration)
# ============================================================================

using AxisArrays: AxisArrays, AxisArray, Axis

"""
    BayesianIRFResult{T, A<:AxisArray, I} <: AbstractIRFResult{T}

IRF result from Bayesian estimation with full posterior draws.

Produced by `BayesianVAR.jl`; this hub package defines only the shared result
type so that plotting and accessors work uniformly across estimators.

Data is stored as AxisArray with dimensions:
- `:draw` - posterior draw index
- `:variable` - response variable (Symbol or Int)
- `:shock` - structural shock (Symbol or Int)
- `:horizon` - IRF horizon (0:H)

# Example Access
```julia
irf.data[:, :GDP, :MonetaryPolicy, 0:12]  # GDP response to monetary shock, horizons 0-12
```

The trailing type parameter `B` is the element type of `lower` and `upper`;
`BayesianIRFResult(data, lower, upper, coverage, identification, metadata)` and
`BayesianIRFResult{T, A, I}(...)` infer it from the bands.

# Fields
- `data::A`: 4D AxisArray (draw × variable × shock × horizon)
- `lower::Vector{B}`: Credible bands (one per coverage level)
- `upper::Vector{B}`: Credible bands (one per coverage level)
- `coverage::Vector{Float64}`: Coverage levels (e.g., [0.68, 0.90, 0.95])
- `identification::I`: Identification scheme used (any identification type)
- `metadata::NamedTuple`: Additional information, defined by the producing package
  and not read by this one
"""
struct BayesianIRFResult{T <: AbstractFloat, A <: AxisArray, I, B <: AxisArray} <:
       AbstractIRFResult{T}
    data::A                         # 4D AxisArray (draw × variable × shock × horizon)
    lower::Vector{B}                # Credible bands (one per coverage level)
    upper::Vector{B}
    coverage::Vector{Float64}
    identification::I               # Any identification type
    metadata::NamedTuple
end

function BayesianIRFResult(
        data::AxisArray, lower, upper, coverage, identification,
        metadata
)
    return BayesianIRFResult{eltype(data), typeof(data), typeof(identification)}(
        data, lower,
        upper, coverage, identification, metadata
    )
end

function BayesianIRFResult{T, A, I}(
        data, lower, upper, coverage, identification,
        metadata
) where {T, A, I}
    return BayesianIRFResult{T, A, I, _band_eltype(lower, upper)}(
        data, lower, upper, coverage,
        identification, metadata
    )
end

"""
    LocalProjectionIRFResult{T, A<:AxisArray} <: AbstractIRFResult{T}

IRF result from local projection estimation.

Produced by `LocalProjections.jl`; this hub package defines only the shared
result type so that plotting and accessors work uniformly across estimators.

Data is stored as AxisArray with dimensions:
- `:response` - response variable
- `:shock` - shock variable
- `:horizon` - IRF horizon (0:H)

# Example Access
```julia
irf.data[:y, :x, 0:6]  # Response of y to shock x, horizons 0-6
```

The trailing type parameters `S` and `B` are the type of `stderr` and the element
type of `lower` and `upper`; `LocalProjectionIRFResult(data, stderr, lower, upper,
coverage, metadata)` and `LocalProjectionIRFResult{T, A}(...)` infer them.

# Fields
- `data::A`: 3D AxisArray (response × shock × horizon)
- `stderr::S`: Standard errors (same shape as data)
- `lower::Vector{B}`: Confidence bands (one per coverage level)
- `upper::Vector{B}`: Confidence bands (one per coverage level)
- `coverage::Vector{Float64}`: Coverage levels (e.g., [0.68, 0.90, 0.95])
- `metadata::NamedTuple`: Additional information, defined by the producing package
  and not read by this one
"""
struct LocalProjectionIRFResult{
    T <: AbstractFloat, A <: AxisArray, S <: AxisArray,
    B <: AxisArray
} <: AbstractIRFResult{T}
    data::A                         # 3D AxisArray (response × shock × horizon)
    stderr::S                       # Standard errors (same shape)
    lower::Vector{B}                # Confidence bands
    upper::Vector{B}
    coverage::Vector{Float64}
    metadata::NamedTuple
end

function LocalProjectionIRFResult(
        data::AxisArray, stderr, lower, upper, coverage,
        metadata
)
    return LocalProjectionIRFResult{eltype(data), typeof(data)}(
        data, stderr, lower, upper,
        coverage, metadata
    )
end

function LocalProjectionIRFResult{T, A}(
        data, stderr, lower, upper, coverage,
        metadata
) where {T, A}
    return LocalProjectionIRFResult{T, A, typeof(stderr), _band_eltype(lower, upper)}(
        data,
        stderr, lower, upper, coverage, metadata
    )
end

# Narrowest common type of the band arrays in `lower` and `upper`; the declared
# element type when both are empty.
function _band_eltype(lower, upper)
    isempty(lower) && isempty(upper) && return typejoin(eltype(lower), eltype(upper))
    return mapreduce(typeof, typejoin, Iterators.flatten((lower, upper)))
end

# ============================================================================
# Narrative Identification Types
# ============================================================================

"""
    NarrativeShockRestriction{D}

Single narrative restriction constraining a shock at a specific date.

`D` can be any date type: `Int` (index), `Dates.Date`, `Dates.DateTime`, or types from
`PeriodicalDates.jl` (e.g., `MonthlyDate`, `QuarterlyDate`).

# Fields
- `shock::Int`: Shock index (1-based)
- `date::D`: Date of the restriction (flexible type)
- `sign::Int`: +1 (positive) or -1 (negative)

# Constructors
```julia
# Using integer index
NarrativeShockRestriction(1, 50, 1)   # Shock 1 positive at index 50
NarrativeShockRestriction(1, 50, -1)  # Shock 1 negative at index 50

# Using Date
using Dates
NarrativeShockRestriction(1, Date(2008, 9, 15), -1)  # Lehman shock
```
"""
struct NarrativeShockRestriction{D}
    shock::Int       # Shock index (1-based)
    date::D          # Date of the restriction (flexible type)
    sign::Int        # +1 (positive) or -1 (negative)

    function NarrativeShockRestriction{D}(shock::Int, date::D, sign::Int) where {D}
        sign ∈ (-1, 1) || throw(ArgumentError("sign must be +1 or -1, got $sign"))
        shock > 0 || throw(ArgumentError("shock index must be positive"))
        return new{D}(shock, date, sign)
    end
end

# Convenience constructor
function NarrativeShockRestriction(shock::Int, date::D, sign::Int) where {D}
    return NarrativeShockRestriction{D}(shock, date, sign)
end

# Ergonomic constructors using the `+` / `-` functions for the sign
function NarrativeShockRestriction(shock::Int, date::D, ::typeof(+)) where {D}
    return NarrativeShockRestriction{D}(shock, date, 1)
end

function NarrativeShockRestriction(shock::Int, date::D, ::typeof(-)) where {D}
    return NarrativeShockRestriction{D}(shock, date, -1)
end

function Base.show(io::IO, nr::NarrativeShockRestriction)
    sign_str = nr.sign == 1 ? "+" : "-"
    return print(
        io, "NarrativeShockRestriction(", nr.shock, ", ", nr.date, ", ", sign_str, ")")
end

"""
    NarrativeRestriction{D} <: AbstractIdentification

Narrative sign restrictions combining sign restrictions with historical shock constraints.

# Fields
- `sign_restrictions::Matrix{Int}`: Square sign restriction matrix, as in
  [`SignRestriction`](@ref) (`0` = unrestricted)
- `narrative_shocks::Vector{NarrativeShockRestriction{D}}`: Historical shock
  constraints; each shock index must be at most the number of shocks
- `horizon::Int`: Sign restrictions apply at horizons `0:horizon` (`0` = impact only)

# Constructors

- `NarrativeRestriction(sign_restrictions, narrative_shocks, horizon)`
- `NarrativeRestriction(sign_restrictions, narrative_shocks; horizon = 0)`
- `NarrativeRestriction(narrative_shocks, n_vars; horizon = 0)`
- `NarrativeRestriction(sign_restrictions, table; horizon = 0, shock_col, date_col, sign_col)`
- `NarrativeRestriction(table; n_vars, horizon = 0, ...)`

All forms validate their arguments. From explicit restrictions:
```julia
NarrativeRestriction(
    [1 -1 0; 0 1 -1; 0 0 1],
    [NarrativeShockRestriction(1, Date(2008, 9, 15), -1)];
    horizon = 0
)
```

Narrative constraints only (no sign restrictions), for a 3-variable model:
```julia
NarrativeRestriction([NarrativeShockRestriction(1, Date(2008, 9, 15), -1)], 3)
```

From a Tables.jl-compatible table:
```julia
using DataFrames, Dates
narrative_df = DataFrame(
    shock = [1, 2],
    date = [Date(2008, 9, 15), Date(2020, 3, 1)],
    sign = [-1, -1]
)
NarrativeRestriction([1 -1; 0 1], narrative_df; horizon=0)
```
"""
struct NarrativeRestriction{D} <: AbstractIdentification
    sign_restrictions::Matrix{Int}
    narrative_shocks::Vector{NarrativeShockRestriction{D}}
    horizon::Int

    function NarrativeRestriction{D}(sign_restrictions, narrative_shocks, horizon) where {D}
        R = _normalize_sign_matrix(sign_restrictions)
        _check_sign_restriction_args(R, horizon)
        n_shocks = size(R, 2)
        for nr in narrative_shocks
            nr.shock <= n_shocks || throw(
                ArgumentError(
                "narrative restriction on shock $(nr.shock), but the model has $n_shocks shocks"
            )
            )
        end
        return new{D}(R, narrative_shocks, horizon)
    end
end

# Date type of a vector of narrative restrictions. An empty vector with an
# abstract element type carries no dates, so any `D` is consistent with it.
_narrative_date_type(::AbstractVector{NarrativeShockRestriction{D}}) where {D} = D
function _narrative_date_type(v::AbstractVector{<:NarrativeShockRestriction})
    return isempty(v) ? Any : mapreduce(nr -> typeof(nr.date), typejoin, v)
end

function NarrativeRestriction(
        sign_restrictions::AbstractMatrix,
        narrative_shocks::AbstractVector{<:NarrativeShockRestriction},
        horizon::Integer
)
    D = _narrative_date_type(narrative_shocks)
    return NarrativeRestriction{D}(sign_restrictions, narrative_shocks, horizon)
end

function NarrativeRestriction(
        sign_restrictions::AbstractMatrix,
        narrative_shocks::AbstractVector{<:NarrativeShockRestriction};
        horizon::Integer = 0
)
    return NarrativeRestriction(sign_restrictions, narrative_shocks, horizon)
end

# Constructor from Tables.jl-compatible table
function NarrativeRestriction(
        sign_restrictions::AbstractMatrix, table;
        horizon::Integer = 0,
        shock_col::Symbol = :shock,
        date_col::Symbol = :date,
        sign_col::Symbol = :sign
)
    Tables.istable(table) ||
        throw(ArgumentError("table must implement Tables.jl interface"))

    shocks = [NarrativeShockRestriction(
                  Int(Tables.getcolumn(row, shock_col)),
                  Tables.getcolumn(row, date_col),
                  Int(Tables.getcolumn(row, sign_col))
              )
              for row in Tables.rows(table)]
    isempty(shocks) && throw(ArgumentError("table must have at least one row"))
    return NarrativeRestriction(sign_restrictions, shocks; horizon)
end

# Convenience: construct from just a table (no sign restrictions)
function NarrativeRestriction(table; n_vars::Integer, horizon::Integer = 0, kwargs...)
    sign_restrictions = zeros(Int, n_vars, n_vars)  # No sign restrictions
    return NarrativeRestriction(sign_restrictions, table; horizon, kwargs...)
end

# Convenience: construct from just narrative_shocks and n_vars (no sign restrictions)
function NarrativeRestriction(
        narrative_shocks::AbstractVector{<:NarrativeShockRestriction},
        n_vars::Integer;
        horizon::Integer = 0
)
    sign_restrictions = zeros(Int, n_vars, n_vars)  # No sign restrictions
    return NarrativeRestriction(sign_restrictions, narrative_shocks; horizon)
end

# ============================================================================
# AbstractIRFResult interface
# ============================================================================

"""
    point_estimate(irf::AbstractIRFResult)

Point estimate of the impulse responses, a 3-dimensional array indexed
`(variable, shock, horizon)`.

- `IRFResult`, `LocalProjectionIRFResult`: the stored estimate (not a copy).
- `SignRestrictedIRFResult`: the stored pointwise median across accepted draws.
- `BayesianIRFResult`: the pointwise posterior median, computed from the draws.
"""
function point_estimate end

point_estimate(irf::IRFResult) = irf.irf
point_estimate(irf::SignRestrictedIRFResult) = irf.irf_median
point_estimate(irf::BayesianIRFResult) = _median_over_draws(irf.data)
point_estimate(irf::LocalProjectionIRFResult) = irf.data

# Helper: compute median over first dimension for AxisArray
function _median_over_draws(data::AxisArray)
    med = dropdims(median(Array(data); dims = 1); dims = 1)
    axes_without_draw = AxisArrays.axes(data)[2:end]
    return AxisArray(med, axes_without_draw...)
end

"""
    mean_estimate(irf::BayesianIRFResult)

Extract mean IRF (for Bayesian results).
"""
function mean_estimate(irf::BayesianIRFResult)
    return _mean_over_draws(irf.data)
end

# Helper: compute mean over first dimension for AxisArray
function _mean_over_draws(data::AxisArray)
    m = dropdims(mean(Array(data); dims = 1); dims = 1)
    axes_without_draw = AxisArrays.axes(data)[2:end]
    return AxisArray(m, axes_without_draw...)
end

"""
    lowerbounds(irf::AbstractIRFResult)

Lower confidence or credible bands: a vector with one `(variable, shock, horizon)`
array per entry of `coverages(irf)`. Returns the stored vector, not a copy.
"""
function lowerbounds end

lowerbounds(irf::IRFResult) = irf.lower
lowerbounds(irf::SignRestrictedIRFResult) = irf.lower
lowerbounds(irf::BayesianIRFResult) = irf.lower
lowerbounds(irf::LocalProjectionIRFResult) = irf.lower

"""
    upperbounds(irf::AbstractIRFResult)

Upper confidence or credible bands: a vector with one `(variable, shock, horizon)`
array per entry of `coverages(irf)`. Returns the stored vector, not a copy.
"""
function upperbounds end

upperbounds(irf::IRFResult) = irf.upper
upperbounds(irf::SignRestrictedIRFResult) = irf.upper
upperbounds(irf::BayesianIRFResult) = irf.upper
upperbounds(irf::LocalProjectionIRFResult) = irf.upper

"""
    coverages(irf::AbstractIRFResult) -> Vector{Float64}

Coverage levels of the bands returned by `lowerbounds` and `upperbounds`, in the
same order.
"""
function coverages end

coverages(irf::IRFResult) = irf.coverage
coverages(irf::SignRestrictedIRFResult) = irf.coverage
coverages(irf::BayesianIRFResult) = irf.coverage
coverages(irf::LocalProjectionIRFResult) = irf.coverage

# Values of axis `d` of an AxisArray, as a Vector.
_axis_values(A::AxisArray, d::Integer) = collect(AxisArrays.axisvalues(A)[d])

"""
    varnames(irf::AbstractIRFResult) -> Vector

Names of the response variables, in the order of the first dimension of
`point_estimate(irf)`.
"""
varnames(irf::IRFResult) = _axis_values(irf.irf, 1)
varnames(irf::SignRestrictedIRFResult) = _axis_values(irf.irf_median, 1)
varnames(irf::BayesianIRFResult) = _axis_values(irf.data, 2)
varnames(irf::LocalProjectionIRFResult) = _axis_values(irf.data, 1)

"""
    shocknames(irf::AbstractIRFResult) -> Vector

Names of the structural shocks, in the order of the second dimension of
`point_estimate(irf)`.
"""
function shocknames end

shocknames(irf::IRFResult) = _axis_values(irf.irf, 2)
shocknames(irf::SignRestrictedIRFResult) = _axis_values(irf.irf_median, 2)
shocknames(irf::BayesianIRFResult) = _axis_values(irf.data, 3)
shocknames(irf::LocalProjectionIRFResult) = _axis_values(irf.data, 2)

"""
    has_draws(irf::AbstractIRFResult) -> Bool

Whether the result stores bootstrap, posterior, or accepted-rotation draws.
Defaults to `false`.
"""
has_draws(::AbstractIRFResult) = false
has_draws(irf::IRFResult) = irf.bootstrap_draws !== nothing
has_draws(::SignRestrictedIRFResult) = true
has_draws(::BayesianIRFResult) = true

"""
    n_draws(irf::AbstractIRFResult) -> Int

Number of stored draws; `0` when `has_draws(irf)` is `false`.
"""
n_draws(irf::AbstractIRFResult) = has_draws(irf) ? size(get_draws(irf), 1) : 0

"""
    get_draws(irf::AbstractIRFResult)

Stored draws, a 4-dimensional array indexed `(draw, variable, shock, horizon)`.
Returns the stored array, not a copy. Throws an `ArgumentError` when
`has_draws(irf)` is `false`.
"""
function get_draws(irf::AbstractIRFResult)
    throw(
        ArgumentError(
        "$(nameof(typeof(irf))) does not store draws; " *
        "check `has_draws(irf)` before calling `get_draws`"
    )
    )
end
function get_draws(irf::IRFResult)
    irf.bootstrap_draws === nothing &&
        throw(
            ArgumentError(
            "this IRFResult has no bootstrap draws; compute it with " *
            "a bootstrap `inference` to store them"
        )
        )
    return irf.bootstrap_draws
end
get_draws(irf::SignRestrictedIRFResult) = irf.irf_draws
get_draws(irf::BayesianIRFResult) = irf.data

"""
    horizon(irf::AbstractIRFResult) -> Int

Maximum horizon `H` of the IRF; horizons run `0:H`.
"""
horizon(irf::AbstractIRFResult) = size(point_estimate(irf), 3) - 1
horizon(irf::BayesianIRFResult) = size(irf.data, 4) - 1

"""
    n_vars(irf::AbstractIRFResult) -> Int

Number of response variables.
"""
n_vars(irf::AbstractIRFResult) = length(varnames(irf))

"""
    n_shocks(irf::AbstractIRFResult) -> Int

Number of structural shocks.
"""
n_shocks(irf::AbstractIRFResult) = length(shocknames(irf))

"""
    size(irf::AbstractIRFResult) -> (n_vars, n_shocks, horizon + 1)

Size of the `(variable, shock, horizon)` point-estimate array.
"""
Base.size(irf::AbstractIRFResult) = (n_vars(irf), n_shocks(irf), horizon(irf) + 1)

function Base.show(io::IO, irf::AbstractIRFResult)
    return print(
        io, nameof(typeof(irf)), "(", n_vars(irf), " variables × ", n_shocks(irf),
        " shocks, horizon ", horizon(irf), ")"
    )
end

function Base.show(io::IO, ::MIME"text/plain", irf::AbstractIRFResult{T}) where {T}
    println(io, nameof(typeof(irf)), "{", T, "}")
    println(io, "  Horizon: ", horizon(irf))
    println(io, "  Variables: ", join(varnames(irf), ", "))
    println(io, "  Shocks: ", join(shocknames(irf), ", "))
    has_draws(irf) && println(io, "  Draws: ", n_draws(irf))
    cvgs = coverages(irf)
    return isempty(cvgs) || print(io, "  Coverage: ", join(cvgs .* 100, "%, "), "%")
end

# ============================================================================
# Normalization (needed here so IRFScale can reference AbstractNormalization)
# ============================================================================

"""
    AbstractNormalization

Abstract type for shock normalization schemes.
"""
abstract type AbstractNormalization end

"""
    UnitStd <: AbstractNormalization

Normalize shocks to have unit variance (standard normalization).
"""
struct UnitStd <: AbstractNormalization end

"""
    UnitEffect <: AbstractNormalization

Normalize shocks to have unit effect on diagonal (on impact).
"""
struct UnitEffect <: AbstractNormalization end

# ============================================================================
# Scaling information
# ============================================================================

"""
    IRFScale{T, N <: AbstractNormalization}

Describes the scaling state of an IRF result.

# Fields
- `normalization::N`: normalization scheme (`UnitStd()` or `UnitEffect()`)
- `scale::Vector{T}`: per-shock multiplicative scale factors applied to the IRF.
  Entry `j` is the cumulative factor for shock `j`.
- `impact_diagonal::Vector{T}`: diagonal of the (normalized, unscaled) impact matrix P —
  the on-impact effect of shock j on variable j before any `scale` multiplication.
  For `UnitEffect`, these are all ≈ 1 (by construction). For `UnitStd`, each entry is
  the standard deviation of the structural shock's immediate effect on its own variable.
  For sign-restricted IRFs, this is the median across accepted draws.
- `names::Vector{Symbol}`: variable/shock names (same ordering as `impact_diagonal`)
"""
struct IRFScale{T <: Real, N <: AbstractNormalization}
    normalization::N
    scale::Vector{T}
    impact_diagonal::Vector{T}
    names::Vector{Symbol}
end

function Base.show(io::IO, s::IRFScale)
    println(io, "IRFScale")
    println(io, "  normalization: ", s.normalization)
    println(io, "  per-shock state (impact_diag × scale = effective on-impact):")
    for (nm, d, sc) in zip(s.names, s.impact_diagonal, s.scale)
        eff = d * sc
        println(
            io, "    ", nm, "  impact_diag = ", round(d; digits = 6),
            "  scale = ", round(sc; digits = 6),
            "  effective = ", round(eff; digits = 6)
        )
    end
    return
end

"""
    get_scale(irf::AbstractIRFResult) -> IRFScale

Return the scaling information for an IRF result: the normalization scheme,
the per-shock `scale` factors, and the on-impact diagonal of the structural
impact matrix (before scaling).

# Example
```julia
s = get_scale(result)
s.normalization    # UnitStd() or UnitEffect()
s.scale            # per-shock scale factors [1.0, -1.0, 100.0, ...]
s.impact_diagonal  # diag(P) — what "1 unit of shock j" means
s.names            # variable/shock names
```
"""
function get_scale(irf::Union{IRFResult, SignRestrictedIRFResult})
    md = irf.metadata
    norm = md.normalization()   # reconstruct instance from stored type
    T = eltype(md.impact_diagonal)
    sc = md.scale isa AbstractVector ? T.(md.scale) :
         fill(T(md.scale), length(md.impact_diagonal))
    return IRFScale{T, typeof(norm)}(norm, sc, md.impact_diagonal, Symbol.(md.names))
end
