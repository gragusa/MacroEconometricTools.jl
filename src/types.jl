# ============================================================================
# Core Type Hierarchy
# ============================================================================

# First, define abstract types that will be used as type parameters

"""
    AbstractPrior

Abstract type for prior specifications in Bayesian VAR models.
"""
abstract type AbstractPrior end

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
- `BayesianVAR`: Bayesian estimation with priors
- `IVSVAR`: Instrumental variable structural VAR
- `LocalProjection`: Local projection estimation
"""
abstract type AbstractVARSpec end

"""
    OLSVAR <: AbstractVARSpec

Ordinary least squares VAR estimation specification.
"""
struct OLSVAR <: AbstractVARSpec end

"""
    BayesianVAR{P} <: AbstractVARSpec

Bayesian VAR estimation specification with prior `P`.

# Fields
- `prior::P`: Prior specification (e.g., Minnesota, Normal-Inverse-Wishart)
"""
struct BayesianVAR{P <: AbstractPrior} <: AbstractVARSpec
    prior::P
end

"""
    IVSVAR{I} <: AbstractVARSpec

Instrumental Variable Structural VAR specification.

# Fields
- `instrument::I`: Instrument specification (external, proxy, high-frequency)
"""
struct IVSVAR{I <: AbstractInstrument} <: AbstractVARSpec
    instrument::I
end

"""
    LocalProjection <: AbstractVARSpec

Local projection estimation specification.

# Fields
- `lags::Union{Int,Symbol}`: Number of lags or `:auto` for automatic selection
- `lag_selection::Symbol`: Criterion for lag selection (`:aic`, `:bic`, `:cv`)
"""
struct LocalProjection <: AbstractVARSpec
    lags::Union{Int, Symbol}
    lag_selection::Symbol

    function LocalProjection(lags::Union{Int, Symbol} = :auto; lag_selection::Symbol = :aic)
        lag_selection ∈ [:aic, :bic, :cv] ||
            throw(ArgumentError("lag_selection must be :aic, :bic, or :cv"))
        return new(lags, lag_selection)
    end
end

# ============================================================================
# Prior Types (for Bayesian estimation)
# ============================================================================

"""
    MinnesotaPrior <: AbstractPrior

Minnesota/Litterman prior for Bayesian VAR.

# Fields
- `λ₁::Float64`: Overall tightness
- `λ₂::Float64`: Cross-variable tightness
- `λ₃::Float64`: Lag decay
- `λ₄::Float64`: Exogenous variable tightness (if applicable)
"""
struct MinnesotaPrior <: AbstractPrior
    λ₁::Float64
    λ₂::Float64
    λ₃::Float64
    λ₄::Float64

    function MinnesotaPrior(λ₁ = 0.2, λ₂ = 0.5, λ₃ = 1.0, λ₄ = 1.0)
        all([λ₁, λ₂, λ₃, λ₄] .> 0) ||
            throw(ArgumentError("All hyperparameters must be positive"))
        return new(λ₁, λ₂, λ₃, λ₄)
    end
end

# Placeholder for future priors
struct NormalWishartPrior <: AbstractPrior end

# ============================================================================
# Instrument Types (for IV-SVAR)
# ============================================================================

"""
    ExternalInstrument{T} <: AbstractInstrument

External instrumental variable for identification.

# Fields
- `Z::Matrix{T}`: Instrument matrix (T × k)
- `target_shock::Int`: Index of the shock to be identified
- `method::Symbol`: Estimation method (`:tsls`, `:liml`, `:fuller`)
"""
struct ExternalInstrument{T <: AbstractFloat} <: AbstractInstrument
    Z::Matrix{T}
    target_shock::Int
    method::Symbol

    function ExternalInstrument(Z::Matrix{T}, target_shock::Int; method::Symbol = :tsls) where {T}
        method ∈ [:tsls, :liml, :fuller] ||
            throw(ArgumentError("method must be :tsls, :liml, or :fuller"))
        target_shock > 0 || throw(ArgumentError("target_shock must be positive"))
        return new{T}(Z, target_shock, method)
    end
end

"""
    ProxyIV{T} <: AbstractInstrument

Proxy instrumental variable for identification (Mertens-Ravn, Stock-Watson).

# Fields
- `proxies::Matrix{T}`: Proxy variables (T × k)
- `target_shocks::Vector{Int}`: Indices of shocks to be identified
- `relevance_threshold::Float64`: Threshold for weak instrument testing
"""
struct ProxyIV{T <: AbstractFloat} <: AbstractInstrument
    proxies::Matrix{T}
    target_shocks::Vector{Int}
    relevance_threshold::Float64

    function ProxyIV(proxies::Matrix{T}, target_shocks::Vector{Int};
            relevance_threshold::Float64 = 10.0) where {T}
        all(target_shocks .> 0) ||
            throw(ArgumentError("All target_shocks must be positive"))
        relevance_threshold > 0 ||
            throw(ArgumentError("relevance_threshold must be positive"))
        return new{T}(proxies, target_shocks, relevance_threshold)
    end
end

# ============================================================================
# VAR Coefficient Structure
# ============================================================================

"""
    VARCoefficients{T}

Storage structure for VAR coefficients with constraint information.

# Fields
- `intercept::Vector{T}`: Intercept coefficients (n_vars)
- `lags::Array{T,3}`: Lag coefficients (n_vars, n_vars, n_lags)
- `constraints::Any`: Applied constraints (Vector{<:AbstractConstraint} or nothing)
"""
struct VARCoefficients{T <: AbstractFloat}
    intercept::Vector{T}
    lags::Array{T, 3}
    constraints::Any  # Will be Vector{<:AbstractConstraint} or nothing
end

function VARCoefficients(intercept::Vector{T}, lags::Array{T, 3}) where {T}
    return VARCoefficients{T}(intercept, lags, nothing)
end

# ============================================================================
# VAR Model Structure
# ============================================================================

"""
    VARModel{T,S}

Vector Autoregression model.

# Type Parameters
- `T<:AbstractFloat`: Numeric type for computations
- `S<:AbstractVARSpec`: Model specification type

# Fields
- `spec::S`: Model specification
- `Y::Matrix{T}`: Original data matrix (T × n_vars)
- `X::Matrix{T}`: Lagged data matrix for estimation
- `coefficients::VARCoefficients{T}`: Estimated coefficients
- `residuals::Matrix{T}`: Residuals (T-p × n_vars)
- `Σ::Symmetric{T,Matrix{T}}`: Residual covariance matrix
- `companion::Matrix{T}`: Companion form matrix
- `names::Vector{Symbol}`: Variable names
- `metadata::NamedTuple`: Additional model information
"""
struct VARModel{T <: AbstractFloat, S <: AbstractVARSpec}
    spec::S
    Y::Matrix{T}
    X::Matrix{T}
    coefficients::VARCoefficients{T}
    residuals::Matrix{T}
    Σ::Symmetric{T, Matrix{T}}
    companion::Matrix{T}
    names::Vector{Symbol}
    metadata::NamedTuple
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
- `ordering::Union{Nothing,Vector{Symbol}}`: Variable ordering (nothing = use data order)
"""
struct CholeskyID <: AbstractIdentification
    ordering::Union{Nothing, Vector{Symbol}}
end

CholeskyID() = CholeskyID(nothing)

"""
    SignRestriction <: AbstractIdentification

Sign restriction identification scheme.

# Fields
- `restrictions::Matrix{Int}`: Sign restriction matrix (+1, -1, 0, NaN for unrestricted)
- `horizon::Int`: Horizon over which restrictions apply
"""
struct SignRestriction <: AbstractIdentification
    restrictions::Matrix{Int}
    horizon::Int
end

"""
    IVIdentification <: AbstractIdentification

Identification via instrumental variables (for IV-SVAR).
"""
struct IVIdentification <: AbstractIdentification end

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
Valid only for Cholesky (triangular) identification schemes.

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

    function WildBootstrap(reps::Int = 1000; save_draws::Bool = false)
        reps > 0 || throw(ArgumentError("reps must be positive"))
        return new(reps, save_draws)
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

    function Bootstrap(reps::Int = 1000; save_draws::Bool = false)
        reps > 0 || throw(ArgumentError("reps must be positive"))
        return new(reps, save_draws)
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

    function BlockBootstrap(reps::Int = 1000; block_length::Int = 10, save_draws::Bool = false)
        reps > 0 || throw(ArgumentError("reps must be positive"))
        block_length > 0 || throw(ArgumentError("block_length must be positive"))
        return new(reps, block_length, save_draws)
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
"""
abstract type AbstractIRFResult{T <: AbstractFloat} end

"""
    IRFResult{T}

Impulse response function results for point-identified systems.

# Fields
- `irf::Array{T,3}`: IRF array (horizon, n_vars, n_shocks)
- `stderr::Array{T,3}`: Standard errors (if computed)
- `bootstrap_draws::Union{Nothing, Array{T,4}}`: Bootstrap IRF draws (reps, horizon, n_vars, n_shocks) if saved
- `lower::Vector{Array{T,3}}`: Lower confidence bands (one per coverage level)
- `upper::Vector{Array{T,3}}`: Upper confidence bands
- `coverage::Vector{Float64}`: Coverage levels
- `identification::AbstractIdentification`: Identification scheme used
- `inference::Union{Nothing, InferenceType}`: Inference method used
- `metadata::NamedTuple`: Additional information
"""
struct IRFResult{T <: AbstractFloat} <: AbstractIRFResult{T}
    irf::Array{T, 3}
    stderr::Array{T, 3}
    bootstrap_draws::Union{Nothing, Array{T, 4}}
    lower::Vector{Array{T, 3}}
    upper::Vector{Array{T, 3}}
    coverage::Vector{Float64}
    identification::AbstractIdentification
    inference::Union{Nothing, InferenceType}
    metadata::NamedTuple
end

"""
    SignRestrictedIRFResult{T}

Impulse response function results for sign restriction identification (set-identified).

# Fields
- `irf_median::Array{T,3}`: Median IRF across draws (horizon, n_vars, n_shocks)
- `irf_draws::Array{T,4}`: All drawn IRFs (n_draws, horizon, n_vars, n_shocks)
- `lower::Vector{Array{T,3}}`: Pointwise lower quantile bands
- `upper::Vector{Array{T,3}}`: Pointwise upper quantile bands
- `coverage::Vector{Float64}`: Coverage levels for quantile bands
- `rotation_matrices::Vector{Matrix{T}}`: All rotation matrices satisfying restrictions
- `identification::SignRestriction`: Identification scheme used
- `metadata::NamedTuple`: Additional information (n_draws, etc.)
"""
struct SignRestrictedIRFResult{T <: AbstractFloat} <: AbstractIRFResult{T}
    irf_median::Array{T, 3}
    irf_draws::Array{T, 4}
    lower::Vector{Array{T, 3}}
    upper::Vector{Array{T, 3}}
    coverage::Vector{Float64}
    rotation_matrices::Vector{Matrix{T}}
    identification::SignRestriction
    metadata::NamedTuple
end

# ============================================================================
# AxisArray-based IRF Result Types (Hub types for Spoke integration)
# ============================================================================

using AxisArrays: AxisArrays, AxisArray, Axis

"""
    BayesianIRFResult{T, A<:AxisArray, I} <: AbstractIRFResult{T}

IRF result from Bayesian estimation with full posterior draws.

Data is stored as AxisArray with dimensions:
- `:draw` - posterior draw index
- `:variable` - response variable (Symbol or Int)
- `:shock` - structural shock (Symbol or Int)
- `:horizon` - IRF horizon (0:H)

# Example Access
```julia
irf.data[:, :GDP, :MonetaryPolicy, 0:12]  # GDP response to monetary shock, horizons 0-12
```

# Fields
- `data::A`: 4D AxisArray (draw × variable × shock × horizon)
- `lower::Vector{AxisArray}`: Credible bands (one per coverage level)
- `upper::Vector{AxisArray}`: Credible bands (one per coverage level)
- `coverage::Vector{Float64}`: Coverage levels (e.g., [0.68, 0.90, 0.95])
- `identification::I`: Identification scheme used (any identification type)
- `metadata::NamedTuple`: Additional information (max_horizon, n_draws, etc.)
"""
struct BayesianIRFResult{T <: AbstractFloat, A <: AxisArray, I} <: AbstractIRFResult{T}
    data::A                         # 4D AxisArray (draw × variable × shock × horizon)
    lower::Vector{AxisArray}        # Credible bands (one per coverage level)
    upper::Vector{AxisArray}
    coverage::Vector{Float64}
    identification::I               # Any identification type
    metadata::NamedTuple
end

"""
    LocalProjectionIRFResult{T, A<:AxisArray} <: AbstractIRFResult{T}

IRF result from local projection estimation.

Data is stored as AxisArray with dimensions:
- `:response` - response variable
- `:shock` - shock variable
- `:horizon` - IRF horizon (0:H)

# Example Access
```julia
irf.data[:y, :x, 0:6]  # Response of y to shock x, horizons 0-6
```

# Fields
- `data::A`: 3D AxisArray (response × shock × horizon)
- `stderr::AxisArray`: Standard errors (same shape as data)
- `lower::Vector{AxisArray}`: Confidence bands (one per coverage level)
- `upper::Vector{AxisArray}`: Confidence bands (one per coverage level)
- `coverage::Vector{Float64}`: Coverage levels (e.g., [0.68, 0.90, 0.95])
- `metadata::NamedTuple`: Additional information (horizon, formula, term, etc.)
"""
struct LocalProjectionIRFResult{T <: AbstractFloat, A <: AxisArray} <: AbstractIRFResult{T}
    data::A                         # 3D AxisArray (response × shock × horizon)
    stderr::AxisArray               # Standard errors (same shape)
    lower::Vector{AxisArray}        # Confidence bands
    upper::Vector{AxisArray}
    coverage::Vector{Float64}
    metadata::NamedTuple
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
        new{D}(shock, date, sign)
    end
end

# Convenience constructor
function NarrativeShockRestriction(shock::Int, date::D, sign::Int) where {D}
    NarrativeShockRestriction{D}(shock, date, sign)
end

"""
    NarrativeRestriction{D} <: AbstractIdentification

Narrative sign restrictions combining sign restrictions with historical shock constraints.

# Fields
- `sign_restrictions::Matrix{Int}`: Impact sign restrictions (+1, -1, 0)
- `narrative_shocks::Vector{NarrativeShockRestriction{D}}`: Historical shock constraints
- `horizon::Int`: Horizon over which sign restrictions apply (0 = impact only)

# Constructors

From explicit restrictions:
```julia
NarrativeRestriction(
    sign_restrictions = [1 -1 0; 0 1 -1],
    narrative_shocks = [
        NarrativeShockRestriction(1, Date(2008, 9, 15), -1),
    ],
    horizon = 0
)
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
end

# Constructor from explicit arguments
function NarrativeRestriction(sign_restrictions::Matrix{Int},
        narrative_shocks::Vector{NarrativeShockRestriction{D}};
        horizon::Int = 0) where {D}
    NarrativeRestriction{D}(sign_restrictions, narrative_shocks, horizon)
end

# Constructor from Tables.jl-compatible table
using Tables

function NarrativeRestriction(sign_restrictions::Matrix{Int}, table;
        horizon::Int = 0,
        shock_col::Symbol = :shock,
        date_col::Symbol = :date,
        sign_col::Symbol = :sign)
    Tables.istable(table) ||
        throw(ArgumentError("table must implement Tables.jl interface"))

    rows = Tables.rows(table)
    shocks = NarrativeShockRestriction[]

    for row in rows
        shock_idx = Int(Tables.getcolumn(row, shock_col))
        date_val = Tables.getcolumn(row, date_col)
        sign_val = Int(Tables.getcolumn(row, sign_col))

        sign_val ∈ (-1, 1) || throw(ArgumentError("sign must be +1 or -1, got $sign_val"))
        push!(shocks, NarrativeShockRestriction(shock_idx, date_val, sign_val))
    end

    # Infer date type from first element
    isempty(shocks) && throw(ArgumentError("table must have at least one row"))
    D = typeof(first(shocks).date)
    typed_shocks = Vector{NarrativeShockRestriction{D}}(shocks)

    return NarrativeRestriction{D}(sign_restrictions, typed_shocks, horizon)
end

# Convenience: construct from just a table (no sign restrictions)
function NarrativeRestriction(table; n_vars::Int, horizon::Int = 0, kwargs...)
    sign_restrictions = zeros(Int, n_vars, n_vars)  # No sign restrictions
    return NarrativeRestriction(sign_restrictions, table; horizon, kwargs...)
end

# Convenience: construct from just narrative_shocks and n_vars (no sign restrictions)
function NarrativeRestriction(narrative_shocks::Vector{NarrativeShockRestriction{D}},
        n_vars::Int;
        horizon::Int = 0) where {D}
    sign_restrictions = zeros(Int, n_vars, n_vars)  # No sign restrictions
    return NarrativeRestriction{D}(sign_restrictions, narrative_shocks, horizon)
end

# ============================================================================
# Accessor Functions for IRF Results
# ============================================================================

"""
    point_estimate(irf::AbstractIRFResult)

Extract point estimate IRF.
- For point-identified (IRFResult, LocalProjectionIRFResult): returns the data directly
- For set-identified/Bayesian: returns median across draws
"""
point_estimate(irf::IRFResult) = irf.irf
point_estimate(irf::LocalProjectionIRFResult) = irf.data

function point_estimate(irf::BayesianIRFResult)
    # Median across draw dimension, preserving other axes
    _median_over_draws(irf.data)
end

function point_estimate(irf::SignRestrictedIRFResult)
    irf.irf_median
end

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
    _mean_over_draws(irf.data)
end

# Helper: compute mean over first dimension for AxisArray
function _mean_over_draws(data::AxisArray)
    m = dropdims(mean(Array(data); dims = 1); dims = 1)
    axes_without_draw = AxisArrays.axes(data)[2:end]
    return AxisArray(m, axes_without_draw...)
end

"""
    has_draws(irf::AbstractIRFResult) -> Bool

Check if IRF result contains posterior/bootstrap draws.
"""
has_draws(irf::IRFResult) = irf.bootstrap_draws !== nothing
has_draws(irf::SignRestrictedIRFResult) = true
has_draws(irf::BayesianIRFResult) = true
has_draws(irf::LocalProjectionIRFResult) = false

"""
    n_draws(irf::AbstractIRFResult) -> Int

Get number of posterior/bootstrap draws.
"""
n_draws(irf::IRFResult) = irf.bootstrap_draws === nothing ? 0 : size(irf.bootstrap_draws, 1)
n_draws(irf::SignRestrictedIRFResult) = size(irf.irf_draws, 1)
n_draws(irf::BayesianIRFResult) = size(irf.data, 1)
n_draws(irf::LocalProjectionIRFResult) = 0

"""
    get_draws(irf::AbstractIRFResult)

Get the full draws (for Bayesian/set-identified).
"""
get_draws(irf::BayesianIRFResult) = irf.data
get_draws(irf::SignRestrictedIRFResult) = irf.irf_draws
get_draws(irf::IRFResult) = irf.bootstrap_draws

"""
    lowerbounds(irf::AbstractIRFResult)

Get lower confidence/credible bands.
"""
lowerbounds(irf::AbstractIRFResult) = irf.lower

"""
    upperbounds(irf::AbstractIRFResult)

Get upper confidence/credible bands.
"""
upperbounds(irf::AbstractIRFResult) = irf.upper

"""
    coverages(irf::AbstractIRFResult)

Get coverage levels for bands.
"""
coverages(irf::AbstractIRFResult) = irf.coverage

"""
    horizon(irf::AbstractIRFResult) -> Int

Maximum horizon of the IRF.
"""
function horizon(irf::IRFResult)
    size(irf.irf, 1) - 1
end

function horizon(irf::SignRestrictedIRFResult)
    size(irf.irf_median, 1) - 1
end

function horizon(irf::BayesianIRFResult)
    ax = AxisArrays.axes(irf.data, Axis{:horizon})
    horizons = AxisArrays.axisvalues(ax)[1]
    return maximum(horizons)
end

function horizon(irf::LocalProjectionIRFResult)
    ax = AxisArrays.axes(irf.data, Axis{:horizon})
    horizons = AxisArrays.axisvalues(ax)[1]
    return maximum(horizons)
end

"""
    n_vars(irf::AbstractIRFResult) -> Int

Number of variables in the IRF.
"""
n_vars(irf::IRFResult) = size(irf.irf, 2)
n_vars(irf::SignRestrictedIRFResult) = size(irf.irf_median, 2)

function n_vars(irf::BayesianIRFResult)
    ax = AxisArrays.axes(irf.data, Axis{:variable})
    return length(AxisArrays.axisvalues(ax)[1])
end

function n_vars(irf::LocalProjectionIRFResult)
    ax = AxisArrays.axes(irf.data, Axis{:response})
    return length(AxisArrays.axisvalues(ax)[1])
end

"""
    varnames(irf::BayesianIRFResult) -> Vector

Variable names from AxisArray axes.
"""
function varnames(irf::BayesianIRFResult)
    ax = AxisArrays.axes(irf.data, Axis{:variable})
    return collect(AxisArrays.axisvalues(ax)[1])
end

function varnames(irf::LocalProjectionIRFResult)
    ax = AxisArrays.axes(irf.data, Axis{:response})
    return collect(AxisArrays.axisvalues(ax)[1])
end
