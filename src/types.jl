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
struct BayesianVAR{P<:AbstractPrior} <: AbstractVARSpec
    prior::P
end

"""
    IVSVAR{I} <: AbstractVARSpec

Instrumental Variable Structural VAR specification.

# Fields
- `instrument::I`: Instrument specification (external, proxy, high-frequency)
"""
struct IVSVAR{I<:AbstractInstrument} <: AbstractVARSpec
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
    lags::Union{Int,Symbol}
    lag_selection::Symbol

    function LocalProjection(lags::Union{Int,Symbol}=:auto; lag_selection::Symbol=:aic)
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

    function MinnesotaPrior(λ₁=0.2, λ₂=0.5, λ₃=1.0, λ₄=1.0)
        all([λ₁, λ₂, λ₃, λ₄] .> 0) || throw(ArgumentError("All hyperparameters must be positive"))
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
struct ExternalInstrument{T<:AbstractFloat} <: AbstractInstrument
    Z::Matrix{T}
    target_shock::Int
    method::Symbol

    function ExternalInstrument(Z::Matrix{T}, target_shock::Int; method::Symbol=:tsls) where T
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
struct ProxyIV{T<:AbstractFloat} <: AbstractInstrument
    proxies::Matrix{T}
    target_shocks::Vector{Int}
    relevance_threshold::Float64

    function ProxyIV(proxies::Matrix{T}, target_shocks::Vector{Int};
                     relevance_threshold::Float64=10.0) where T
        all(target_shocks .> 0) || throw(ArgumentError("All target_shocks must be positive"))
        relevance_threshold > 0 || throw(ArgumentError("relevance_threshold must be positive"))
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
struct VARCoefficients{T<:AbstractFloat}
    intercept::Vector{T}
    lags::Array{T,3}
    constraints::Any  # Will be Vector{<:AbstractConstraint} or nothing
end

function VARCoefficients(intercept::Vector{T}, lags::Array{T,3}) where T
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
struct VARModel{T<:AbstractFloat,S<:AbstractVARSpec}
    spec::S
    Y::Matrix{T}
    X::Matrix{T}
    coefficients::VARCoefficients{T}
    residuals::Matrix{T}
    Σ::Symmetric{T,Matrix{T}}
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
    ordering::Union{Nothing,Vector{Symbol}}
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
# IRF Structure
# ============================================================================

"""
    AbstractIRFResult{T}

Abstract supertype for all impulse response function result types.

Concrete subtypes:
- `IRFResult{T}`: Point-identified IRF results (Cholesky, IV, etc.)
- `SignRestrictedIRFResult{T}`: Set-identified IRF results (sign restrictions)
"""
abstract type AbstractIRFResult{T<:AbstractFloat} end

"""
    IRFResult{T}

Impulse response function results for point-identified systems.

# Fields
- `irf::Array{T,3}`: IRF array (horizon, n_vars, n_shocks)
- `stderr::Array{T,3}`: Standard errors (if computed)
- `lower::Vector{Array{T,3}}`: Lower confidence bands (one per coverage level)
- `upper::Vector{Array{T,3}}`: Upper confidence bands
- `coverage::Vector{Float64}`: Coverage levels
- `identification::AbstractIdentification`: Identification scheme used
- `inference::Symbol`: Inference method (`:bootstrap`, `:delta`, etc.)
- `metadata::NamedTuple`: Additional information
"""
struct IRFResult{T<:AbstractFloat} <: AbstractIRFResult{T}
    irf::Array{T,3}
    stderr::Array{T,3}
    lower::Vector{Array{T,3}}
    upper::Vector{Array{T,3}}
    coverage::Vector{Float64}
    identification::AbstractIdentification
    inference::Symbol
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
struct SignRestrictedIRFResult{T<:AbstractFloat} <: AbstractIRFResult{T}
    irf_median::Array{T,3}
    irf_draws::Array{T,4}
    lower::Vector{Array{T,3}}
    upper::Vector{Array{T,3}}
    coverage::Vector{Float64}
    rotation_matrices::Vector{Matrix{T}}
    identification::SignRestriction
    metadata::NamedTuple
end
