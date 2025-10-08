"""
`VectorAutoRegression` a vector autoregressive (VAR) model.

# Fields

  - `rawdata`: The actual data in its original form, before any transformations or processing.
  - `Y`: The matrix of data potentially demeaned `p+1:end`, where `p` is the
    number of lags in the model.
  - `X`: Matrix of lagged values of `Y`.
  - `A`: Matrix of estimated coefficients for the VAR model of size `(m*p, m)`
    or `(m*p+1, m)` depending on whether the model is fitted with an intercept,
    where `m` is the number of variables and `p` is the number of lags. The intercept is
    the first row of the matrix.
  - `Sigma_e`: Covariance matrix of residuals (m*m).
  - `F`: Companion matrix, used in the analysis and computation of VAR models.
  - `hasintercept`: Indicates whether the model is fitted with an intercept.
  - `longrunmean`: The long-run mean of the VAR model. (m*1). This is given by
    `(I - A₁ - A₂ - ... - Aₚ)⁻¹ α`, where `Aⱼ` are the matrix of coefficients
    corresponding to lag `j` and `α` is the intercept.
  - `names`: Names of variables in the model. It helps in identifying
    and accessing model variables by name.
  - `timeindex`: Time index, representing the temporal aspect of the data.
"""
struct VectorAutoRegression{M,V}
    "The actual data - untouched"
    rawdata::M
    "Matrix of data - could be demeaned from p+1:end"
    Y::M
    "Matrix of lagged value of Y"
    X::M
    "Matrix of estimated coefficient"
    A::M  ## Matrix Coefficient
    "Residuals"
    residuals::M
    "Covariance Matrix of residuals"
    Sigma_e::M  ## Residual variance
    "Companion Matrix"
    F::M  ## Companion Matrix
    "If fitted with intercept"
    hasintercept::Bool
    "longrunmean"
    longrunmean::V
    "Names of variables"
    names::Vector{Symbol}
    "Time index"
    timeindex::Any
end

names(v::VectorAutoRegression) = v.names

function Base.show(io::IO, v::VectorAutoRegression)
    println(io, "Vector AutoRegression Model")
    println(io, "--------------------------")
    println(io, "Number of Variables: $(nvars(v))")
    println(io, "Number of Lags: $(nlags(v))")
    println(io, "Number of Observations: $(nobs(v))")
    println(io, "Has Intercept: $(hasintercept(v))")
    println(io, "Variable Names: $(varnames(v))")
    return println(io, "Time Index: $(timeindex(v))")
end

const VAR = VectorAutoRegression

function VAR(n::Int64, nlags, nvars, type::Type{FF}=Float64;
             intercept::Bool=true,
             names=Symbol.("Y_" .* string.(1:nvars)),
             index=1:n) where {FF<:AbstractFloat}
    parent = Array{FF}(undef, n, nvars)
    Y = similar(parent)
    X = similar(parent, (n, nvars * nlags + Int(intercept)))
    A = similar(parent, (nvars * nlags + Int(intercept), nvars))
    F = zeros(FF, nvars * nlags, nvars * nlags)
    E = similar(Y, (n - nlags, nvars))
    @inbounds for j ∈ (nvars + 1):(nvars * nlags)
        F[j, j - nvars] = one(FF)
    end
    μ = Array{FF}(undef, nvars)
    Σ = similar(parent, (nvars, nvars))
    return VAR(parent, Y, X, A, E, Σ, F,
               intercept,
               μ,
               names,
               index)
end

## This is used for the simulation of the bootstrap. Takes the VAR
## and use the information to fit the VAR with different values
## WARNING: The original VAR should be copied before using this function
@propagate_inbounds function noninvasivefit!(v::VectorAutoRegression, Y)
    _, p, m = sizes(v)
    copy!(v.Y, Y)
    delag!(v.X, v.Y, p, hasintercept(v))
    X = @view v.X[(p + 1):end, :]
    Y = @view v.Y[(p + 1):end, :]
    v.A .= X \ Y
    mul!(v.residuals, X, v.A)
    v.residuals .-= Y
    df = size(v.residuals, 1) - Int(hasintercept(v)) - m * p
    mul!(v.Sigma_e, v.residuals', v.residuals)
    rmul!(v.Sigma_e, 1 // df)
    update_companion!(v)
    return v
end

@propagate_inbounds function StatsBase.fit!(v::VectorAutoRegression;
                                            estimatecoef::Bool=true,
                                            copyrawdata::Bool=true)
    _, p, m = sizes(v)
    delag!(v.X, v.Y, p, hasintercept(v))
    if estimatecoef
        X = @view v.X[(p + 1):end, :]
        Y = @view v.Y[(p + 1):end, :]
        v.A .= X \ Y
        mul!(v.residuals, X, v.A)
        v.residuals .-= Y
        df = size(v.residuals, 1) - Int(hasintercept(v)) - m * p
        mul!(v.Sigma_e, v.residuals', v.residuals)
        rmul!(v.Sigma_e, 1 // df)
        update_companion!(v)
        α = v.A[1, :]
        A = coef(v)
        AA = dropdims(sum(A; dims=3); dims=3)
        μ = (I - AA) \ α
        v.longrunmean .= μ
    end
    return v
end

function VAR(Y::Matrix, p::Int; kwargs...)
    n, m = size(Y)
    v = VAR(n, p, m, eltype(Y); kwargs...)
    copy!(v.rawdata, Y)
    copy!(v.Y, Y)
    fit!(v)
    return v
end

"""
rawnobs(v::VectorAutoRegression)
nobs(v::VectorAutoRegression)
nvars(v::VectorAutoRegression)
nlags(v::VectorAutoRegression)

These methods retrieve the sizes of the VAR model.

# Details

  - `rawnobs` returns the total number of observations in the original data, i.e. `T`
  - `nlags` returns the number of lags used in the VAR model, i.e. `p`
  - `nobs` returns the total number of observations used, i.e. `T-p`
  - `nvars` returns the number of variables in the VAR model, i.e. `m`

# Returns

  - Each function return an integer.
"""
nvars(v::VectorAutoRegression) = size(v.Y, 2)
nlags(v::VectorAutoRegression) = Int((size(v.X, 2) - Int(v.hasintercept)) // nvars(v))
StatsBase.nobs(v::VectorAutoRegression) = Int(size(v.Y, 1) - nlags(v))
rawnobs(v::VectorAutoRegression) = size(v.Y, 1)
sizes(v::VAR) = (nobs(v), nlags(v), nvars(v))
@doc (@doc nvars) nlags, nobs, rawnobs

StatsModels.hasintercept(v::VectorAutoRegression) = v.hasintercept
intercept(v::VectorAutoRegression) = rawcoef(v)[1, :]

longrunmean(v::VectorAutoRegression) = v.longrunmean
longruneffect(v::VectorAutoRegression) = inv(I - dropdims(sum(coef(v), dims = 3), dims = 3))

companionmatrix(v::VectorAutoRegression) = v.F
laggedexogenous(l::VectorAutoRegression) = l.X
varnames(v::VectorAutoRegression) = v.names
timeindex(v::VectorAutoRegression) = v.timeindex
roots(v::VectorAutoRegression) = inv.(eigvals(companionmatrix(v)))
function marepresentation(v::VectorAutoRegression, j)
    out = Array{eltype(v.A),3}(undef, nvars(v), nvars(v), j+1) 
    out[:,:,1] = Matrix(float(I(nvars(v))))
    out[:,:,2] = v.F[1:nvars(v), 1:nvars(v)]
    if j == 1
        return out
    end
    for i ∈ 2:j
        out[:,:,i+1] = (v.F^i)[1:nvars(v), 1:nvars(v)]
    end
    return out
end

"""
coef(v::VectorAutoRegression)
rawcoef(v::VectorAutoRegression)

Extracts the coefficients of a (VAR) model.

# Arguments

  - `v::VectorAutoRegression`: A fitted VAR model.

# Returns

  - `coef` returns a 3-dimensional array of coefficients with dimensions `(m, m, p)`,
    where `m` is the number of variables in the VAR model and `p` is the number of lags.
  - `rawcoef` returns the raw coefficient matrix `A` of the VAR model of size `(m*p, m)`,
    where `m` is the number of variables and `p` is the number of lags.

# Details

Each `m x m` matrix slice of the array returned by `coef` corresponds to A_j. The element
`(r, c)` of A_j is the coefficient of the `r`th variable's `j`th lag on the `c`th variable.
"""
function StatsBase.coef(v::VectorAutoRegression)
    p, m = nlags(v), nvars(v)
    return reshape(view(rawcoef(v), 2:(m * p + Int(hasintercept(v))), :)', (m, m, p))
end

rawcoef(v::VectorAutoRegression) = v.A
@doc (@doc coef) rawcoef

"""
    update_companion!(v::VectorAutoRegression)

Updates the companion matrix `F` of a `VectorAutoRegression` object in-place based on the
current estimated coefficients stored in `A`.

# Arguments

  - `v::VectorAutoRegression`: The fitted VAR whose companion matrix `F` is to be updated.

# Note

Internal API
"""
@propagate_inbounds function update_companion!(v::VectorAutoRegression)
    p, m = nlags(v), nvars(v)
    tmp = coef(v)
    for j ∈ 1:p
        v.F[1:m, (1 + (j - 1)m):(m + (j - 1)m)] .= tmp[:, :, j]
    end
    return v.F
end

"""
    StatsBase.residuals(v::VectorAutoRegression)

Calculates the residuals of a `VectorAutoRegression` (VAR) model, i.e. Y - X * A.

# Arguments

  - `v::VectorAutoRegression`: A fitted VAR.

# Returns

  - A matrix (T-p, m) of residuals for the VAR model, where each column
    corresponds to a variable in the model and each row corresponds
    to an observation.
"""
StatsBase.residuals(v::VectorAutoRegression) = v.residuals

"""
residualsvariance(v::VectorAutoRegression)

Retrieves the (estimated) variance of the residuals of a VAR model.
"""
function residualsvariance(v::VectorAutoRegression; type=:ols)
    type ∈ (:ols, :mle) || throw(ArgumentError("Unknown type: $type"))
    if type == :ols
        return Symmetric(v.Sigma_e)
    else
        df = nobs(v) - hasintercept(v) - nvars(v) * nlags(v)
        return Symmetric(v.Sigma_e * df / (df + nvars(v) * nlags(v) + hasintercept(v)))
    end
end

"""
    isstable(var::VectorAutoRegression; verbose::Bool=false)

Check if a given Vector AutoRegression (VAR) model is stable by examining the eigenvalues
of the VAR(1) representation.

# Parameters

  - `var`: The VAR model to check for stability.
  - `verbose`: Optional boolean flag for verbose output. If `true`, the function will
    print the eigenvalues of the VAR model. Default is `false`.

# Returns

Returns `true` if the VAR model is stable (i.e., all eigenvalues of the companion
matrix have absolute values less than or equal to 1), `false` otherwise.
"""
function isstable(v::VectorAutoRegression; verbose::Bool=false)
    eigs = eigvals(companionmatrix(v))
    if verbose
        println("Eigenvalues of VAR($(nlags(v)))")
        for val ∈ sort(abs.(eigs); rev=true)
            println(val)
        end
    end
    return all(abs.(eigs) .<= 1)
end

"""
loglikelihood(v::VectorAutoRegression)

Calculate the log-likelihood of a VAR(p) model.

# Arguments

  - `v::VectorAutoRegression`: A fitted VAR.

# Returns

  - The value of the log-likelihood function for the fitted VAR model.

# Notes

The log-likelihood function for the VAR(p) model is defined as:
-\\left(\\frac{T-p}{2}\\right)\\left(\\log\\left|\\Sigma\\right|-m\\log\\left(2\\pi\\right)-m\\right)
where T is the number of observations in the original series, `Σ` is the matrix of estimated variance of
the residuals, and `m` is the number of variables in the VAR model.
"""
function StatsBase.loglikelihood(v::VectorAutoRegression)
    n, _, m = sizes(v)
    logd = logdet(residualsvariance(v; type = :mle))
    part1 = -(n * m / 2) * log(2 * π)
    part2 = -(n / 2) * (logd + m)
    return part1 + part2
end

## Lagorder Selection

abstract type LagOrderSelector end

struct VARLagOrderSelector <: LagOrderSelector
    results::Array{@NamedTuple{bic::Float64,aic::Float64,hqic::Float64},1}
end

abstract type InformationCriterion end
struct BIC <: InformationCriterion end
struct AIC <: InformationCriterion end
struct HQIC <: InformationCriterion end

function BIC(v::VectorAutoRegression)
    T = nobs(v)
    m = nvars(v)
    p = nlags(v)
    freepar = p * m^2 + m * Int(hasintercept(v))
    ld = logdet(residualsvariance(v; type=:mle))
    return ld + (log(T) / T) * freepar
end

function AIC(v::VectorAutoRegression)
    T = nobs(v)
    m = nvars(v)
    p = nlags(v)
    freepar = p * m^2 + m * Int(hasintercept(v))
    ld = logdet(residualsvariance(v; type=:mle))
    return ld + (2.0 / T) * freepar
end

function HQIC(v::VectorAutoRegression)
    T = nobs(v)
    m = nvars(v)
    p = nlags(v)
    freepar = p * m^2 + m * Int(hasintercept(v))
    ld = logdet(residualsvariance(v; type=:mle))
    return ld + (2.0 * log(log(T)) / T) * freepar
end

ics(v::VectorAutoRegression) = (bic=BIC(v), aic=AIC(v), hqic=HQIC(v))

function selectorder(v::VectorAutoRegression, maxlags::Int=12)
    ## Fit 0 order VAR
    YY = v.rawdata
    vv = VAR(YY[maxlags:end, :], 0)
    fit!(vv)
    cics = [ics(vv)]
    for p ∈ 0:(maxlags - 1)
        offset = maxlags - p
        vv = VAR(YY[offset:end, :], p)
        fit!(vv)
        push!(cics, ics(vv))
    end

    return LagOrderSelector(cics)
end

function Base.show(io::IO, c::LagOrderSelector)
    println(io, "Lag Order Selection (maxlags: $(length(c.results)-1))")
    # df = DataFrame(Tables.columntable(c.results))
    # df.lag = 0:(size(df, 1) - 1)
    # df = df[:, [:lag, :bic, :aic, :hqic]]
    # min_values = combine(df, DataFrames.names(df) .=> minimum .=> DataFrames.names(df))

    # hl_pv = Highlighter((data, i, j) -> j > 1 && data[i, j] == min_values[!, j][1],
    #                     crayon"bg:white fg:blue")
    # return pretty_table(df;
    #                     header=["lag", "BIC", "AIC", "HQIC"],
    #                     highlighters=hl_pv)
end

function pickorder(c::LagOrderSelector, ::BIC)
    return argmin([c.results[i].bic for i ∈ 1:length(c.results)]) + 1
end
function pickorder(c::LagOrderSelector, ::AIC)
    return argmin([c.results[i].aic for i ∈ 1:length(c.results)]) + 1
end
function pickorder(c::LagOrderSelector, ::HQIC)
    return argmin([c.results[i].hqic for i ∈ 1:length(c.results)]) + 1
end

## ------------------------------------------------------
## Short Run Restrictions
## ------------------------------------------------------
abstract type ShortRunRestrictions end
abstract type AbstractNormalization end
struct UnitStd <: AbstractNormalization end
struct UnitEffect <: AbstractNormalization end

struct TriangularRestriction{C,M} <: ShortRunRestrictions
    R::C
    normalization::M
    function TriangularRestriction(v::VectorAutoRegression, ::UnitEffect)
        Aldl = ldl(v.Sigma_e)
        return new{typeof(Aldl),UnitEffect}(Aldl, UnitEffect())
    end

    function TriangularRestriction(v::VectorAutoRegression, ::UnitStd)
        Achol = cholesky(v.Sigma_e)
        return new{typeof(Achol),UnitStd}(Achol, UnitStd())
    end
end

TriangularRestriction(v::VectorAutoRegression) = new(v, UnitEffect())

identifyingmatrix(r::TriangularRestriction{T}) where {T<:Cholesky} = r.R.L
function identifyingmatrix(r::TriangularRestriction{T}) where {T<:LDLFactorizations.LDLFactorization}
    return Matrix(r.R.L) + I
end

function GeneralRestriction(v::VectorAutoRegression, R::Matrix)
    if size(R, 1) != nvars(v)
        throw(ArgumentError("The number of rows in the restriction matrix must be equal to the number of variables in the VAR model"))
    end
    if size(R, 2) != nvars(v)
        throw(ArgumentError("The number of columns in the restriction matrix must be equal to the number of variables in the VAR model"))
    end
    ## To do: implement this
    ## The fixed values are float, the missing are free parameters?
end

init(x::TriangularRestriction) = identifyingmatrix(x)

## ------------------------------------------------------
## Impulse Response Functions
## ------------------------------------------------------
"""
    VARIRF{F,T,L}

A type representing the (Structural) Impulse Response Function (IRF) of a 
time series model, which is used to analyze the dynamic effects 
of a one-time shock to one of the variables in the model on the 
future values of all variables in the system.


# Fields
- `irf`: The impulse response function values, typically stored as a matrix or a 3D array. 
         Each element represents the response of the system's variables to a shock in another 
         variable at a given time lag.
- `R`: The orthogonalizing matrix used to compute the impulse responses.
- `lb`: The lower bounds of the impulse response function values. 
- `ub`: The upper bounds of the impulse response function values. providing 
   the upper end of the confidence interval for each IRF estimate.
- `coverage`: The coverage probabilities of the confidence intervals 
  (e.g., 0.95 for 95% coverage).
- `names`: An index or list of the variable names in the model, facilitating the 
   interpretation of the IRF by associating each row and column with a specific variable.
- `boundsmethod`: A named tuple containing the method used to compute the confidence intervals, 
                  the number of replications, the block width, and the horizon of the IRF.
"""

abstract type AbstractIRF end

struct VARIRF{F,T,L} <: AbstractIRF
    irf::T
    R::F
    σ::T
    lb::L
    ub::L
    ql::L
    qu::L
    H::Int
    varsizes::@NamedTuple{nobs::Int64,nlags::Int64,nvars::Int64}
    coverage::Array{Float64}
    names::Vector{Symbol}
    boundsmethod::NamedTuple
end

nvars(virf::VARIRF) = virf.varsizes.nvars
nlags(virf::VARIRF) = virf.varsizes.nlags
nhorizon(virf::VARIRF) = virf.H
lowerbounds(virf::VARIRF) = virf.lb
upperbounds(virf::VARIRF) = virf.ub
Base.size(virf::VARIRF) = size(virf.irf)
impulseresponse(virf::VARIRF) = virf.irf

function impulseresponse(v::VectorAutoRegression,
                         r::TriangularRestriction;
                         horizon::Int=24,
                         ci_type=:wildbootstrap,
                         bootreps=999,
                         block_width=10,
                         coverage::Vector=[0.68, 0.90, 0.95],
                         initialvalues=:initialobs)
    if any(coverage .> 1) && all(coverage .< 0)
        throw(ArgumentError("Coverage probabilities must be in the range [0, 1]"))
    end
    if horizon < 0
        throw(ArgumentError("The horizon must be a positive integer"))
    end
    coverage = sort(coverage)
    θ = irf(v, r, horizon)
    if ci_type == :delta
        σ = irf_se_asy(v, r, θ)
        lb, ub = calculatebounds_se(θ, σ, coverage)
        ql = similar(lb)
        qu = similar(ub)
    elseif ci_type == :wildbootstrap
        θᵇ = irfwildboot(v, r, horizon, bootreps, initialvalues)
        lb, ub, ql, qu = calculatebounds_boot(θᵇ, θ, coverage)
        σ = similar(θ, (0,0,0))
    elseif ci_type == :bootstrap
        θᵇ = irfboot(v, r, horizon, bootreps, initialvalues)
        lb, ub, ql, qu = calculatebounds_boot(θᵇ, θ, coverage)
        σ = similar(θ, (0,0,0))
    elseif ci_type == :blockbootstrap
        θᵇ = irfblockboot(v, r, block_width, horizon, bootreps, initialvalues)
        lb, ub, ql, qu = calculatebounds_boot(θᵇ, θ, coverage)
        σ = similar(θ, (0,0,0))
    else
        throw(ArgumentError("Unknown CI type. CI ∈ [:wildbootstrap, :bootstrap, :blockbootstrap]"))
    end
    boundsmethod = (ci_type=ci_type, rep=bootreps, block_width=block_width, H=horizon)
    return VARIRF(θ, r, σ, lb, ub, ql, qu, horizon, (nobs=nobs(v), nlags=nlags(v), nvars=nvars(v)),
                  coverage, varnames(v), boundsmethod)
end

"""
    getbounds_boot(θ, θ̂, coverage=[0.67, 0.90, 0.95])

Calculate the lower and upper bounds for a given bootstrapped IRF.

# Parameters

  - `θ`: A 3-dimensional array representing the bootstrapped IRF. The dimensions are
    [h, m^2, rep], where h is the horizon, m^2 is the number of moments squared, and
    rep is the number of repetitions.
  - `θ̂`: A 2-dimensional array representing the IRF estimated on the sample.
  - `coverage`: An optional array of coverage probabilities. The default values
    are [0.67, 0.90, 0.95].

# Process

The function first transforms the coverage probabilities into quantiles `q_{1-γ/2}`
and `q_{γ/2}`. It then calculates the difference between each element of `θ` and `θ̂`
across the first two dimensions. The lower and upper bounds are then calculated for
each quantile by taking the respective quantile of the differences across the third
dimension.

# Returns

The function returns two 3-dimensional arrays `lb` and `ub`, representing the lower and
upper bounds for each quantile.
"""
function calculatebounds_boot(θ, θ̂, coverage=[0.67, 0.90, 0.95])
    ## Note: θ is the bootstrapped irf [h, m^2, rep]
    ## thetahat is the IRF estimated on the sample
    ## Transform coverage into q_{1-γ/2} and q_{γ/2}
    α = (1 .- coverage) ./ 2
    Z = mapslices(x -> x .- θ̂, θ; dims=(2, 3, 4))
    ql = map(g -> dropdims(mapslices(x -> quantile(x, g), Z; dims=1); dims = 1), α)
    qu = map(g -> dropdims(mapslices(x -> quantile(x, 1 .- g), Z; dims=1); dims = 1), α)
    lb = map(x -> θ̂ + x, ql)
    ub = map(x -> θ̂ + x, qu)
    #return map(x->dropdims(x, dims=1), lb), map(x->dropdims(x, dims=1), ub)
    return lb, ub, ql, qu
end

function calculatebounds_se(θ̂, σ, coverage)
    α = (1.0 .- coverage) ./ 2
    Z = norminvcdf.(1.0 .- α)
    lb = map(z -> θ̂ .- z .* σ, Z)
    ub = map(z -> θ̂ .+ z .* σ, Z)
    return ub, lb
end

@propagate_inbounds function irfwildboot(var::VAR,
                                         r::ShortRunRestrictions,
                                         H=24,
                                         bootreps=100,
                                         initialvalues=:initialobs)
    m = nvars(var)
    wvar = deepcopy(var)
    u = residuals(var)
    ū = similar(u)
    ω = similar(u, size(u, 1))
    cirf = similar(u, (bootreps, H + 1, m, m))
    out = similar(u, (H + 1, m, m))
    𝕐 = similar(var.Y)
    ## Container to be passed to irf!
    out[1, :, :] .= init(r)
    for j ∈ 1:bootreps
        randn!(ω)
        ū .= ω .* u
        bootsimulate!(𝕐, ū, var, initialvalues)
        noninvasivefit!(wvar, 𝕐)
        irf!(out, wvar, r, H)
        cirf[j, :, :, :] .= out
    end
    return cirf
end

@propagate_inbounds function irfblockboot(var::VAR, r::ShortRunRestrictions, block_width=10,
                                          H=24, bootreps=999, initialvalues=:longrunmean)
    T, m = nobs(var), nvars(var)
    wvar = deepcopy(var)
    ℓ = block_width
    u = residuals(var)
    cirf = similar(u, (bootreps, H + 1, m, m))
    𝕐 = similar(var.Y)
    N = ceil(Int, T / ℓ)
    𝒰 = DiscreteUniform(0, T - ℓ)
    𝒾 = Vector{Int64}(undef, T)
    uᵇ = similar(u, (N * ℓ, m))
    out = similar(u, (H + 1, m, m))
    out[1, :, :] .= init(r)
    for jj ∈ 1:bootreps
        rand!(𝒰, 𝒾)
        for j ∈ 1:N
            i = 𝒾[j]
            uᵇ[(1 + ℓ * (j - 1)):(j * ℓ), :] .= @view u[(i + 1):(i + ℓ), :]
        end
        for j ∈ 0:(N - 1)
            for s ∈ 1:ℓ
                uᵇ[j * ℓ + s, :] .-= vec(mean(view(u, s:(s + T - ℓ), :); dims=1))
            end
        end
        # Simulate the VAR
        bootsimulate!(𝕐, uᵇ, var, initialvalues)
        noninvasivefit!(wvar, 𝕐)
        irf!(out, wvar, r, H)
        cirf[jj, :, :, :] .= out
    end
    return cirf
end

function irfboot(var::VAR, r::ShortRunRestrictions, H=24, bootreps=100, initialvalues=:mean)
    n, m = nobs(var), nvars(var)
    wvar = deepcopy(var)
    𝕐 = similar(var.Y)
    u = residuals(var)
    cirf = similar(u, (bootreps, H + 1, m, m))
    out = similar(u, (H + 1, m, m))
    ū = similar(u)
    ω = Array{Int64,1}(undef, size(u, 1))
    out[1, :, :] .= init(r)
    for j ∈ 1:bootreps
        sample!(1:n, ω)
        for j ∈ Base.axes(ū, 2)
            for i ∈ Base.axes(ū, 1)
                ū[i, j] = u[ω[i], j]
            end
        end
        simulate!(𝕐, ū, var, initialvalues)
        noninvasivefit!(wvar, 𝕐)
        irf!(out, wvar, r, H)
        cirf[j, :, :, :] .= out
    end
    return cirf
end

function irf(v::VectorAutoRegression, r::ShortRunRestrictions, H::Int=24)
    m = nvars(v)
    out = similar(companionmatrix(v), (H + 1, m, m))
    out[1, :, :] .= init(r)
    irf!(out, v, r, H)
    return out
end

function irf!(out, v::VectorAutoRegression, r::ShortRunRestrictions, H::Int=24)
    _, p, m = sizes(v)
    R = identifyingmatrix(r)
    F = companionmatrix(v)
    tmp = diagm(0 => ones(eltype(F), m * p))
    @inbounds @fastmath for j ∈ 1:H
        tmp = tmp * F
        mygemm!(view(out, j + 1, :, :), tmp, R)
    end
end

function mygemm!(C, A, B)
    m = size(B, 1)
    ix = Base.axes(A, 1)[1:m]
    @inbounds @fastmath for m ∈ ix, n ∈ Base.axes(B, 2)
        Cmn = zero(eltype(C))
        for k ∈ ix
            Cmn += A[m, k] * B[k, n]
        end
        C[m, n] = Cmn
    end
end

## ------------------------------------------------------
## Utils (demean)
## ------------------------------------------------------

"""
    delag(X, nlags)

Creates a new matrix from the input matrix `X` by generating lagged versions of
each column for a specified number of lags `nlags`. This function is useful in time
series analysis, particularly for preparing lagged predictors in autoregressive models.

# Arguments

  - `X`: The input matrix containing the time series data. Each column represents a
    separate variable or time series.
  - `nlags`: The number of lags to generate for each column in `X`.

# Returns

  - A matrix containing the lagged versions of the columns in `X`. For each column
    in `X`, there will be `nlags` columns in the output matrix, each representing
    the column lagged by 1 to `nlags` periods. Missing values for lags that go beyond
    the start of the series are filled with `NaN`.
"""
function delag(X, nlags)
    delagged = map(j -> map(x -> lag(x, j; default=NaN), eachcol(X)), 1:nlags)
    return reduce(hcat, collect(Base.Iterators.flatten(delagged)))
end

"""
    delag!(dest, X, p::Int64, addintercept=true)

In-place modifies the matrix `dest` to contain lagged versions of the matrix `X`
for a specified number of lags `p`. This function is designed to efficiently prepare
lagged predictor matrices for time series analysis, especially in the context of
constructing autoregressive models.

# Arguments

  - `dest`: The output matrix to be modified in-place. It should be pre-allocated with
    the appropriate dimensions to hold the lagged versions of `X`. The number of rows should
    match `X`, and the number of columns should be `size(X, 2) * p` to accommodate `p` lags
    for each column in `X`.
  - `X`: The input matrix containing the original time series data. Each column represents a separate variable or time series.
  - `p::Int64`: The number of lags to generate for each column in `X`.

# Behavior

For each column `j` in `X`, and for each lag `ℓ` from `1` to `p`, the function shifts the elements in column `j` downwards by `ℓ` positions in the corresponding section of `Xₗ`. It effectively copies the value from `X[t-ℓ, j]` to `Xₗ[t, m*(ℓ-1)+j]` for each time point `t` in `X`, starting from `1+ℓ` to `n`, where `n` is the number of observations and `m` is the number of variables in `X`.
"""
Base.@propagate_inbounds function delag!(dest, Y, p::Int64, addintercept::Bool=true)
    n, m = size(Y, 1), size(Y, 2)
    offset = addintercept ? 1 : 0
    for j ∈ Base.axes(Y, 2)
        for ℓ ∈ 1:p
            for t ∈ (1 + ℓ):n
                dest[t, offset + m * (ℓ - 1) + j] = Y[t - ℓ, j]
            end
        end
    end
    addintercept && fill!(view(dest, :, 1), 1)
    fill!(view(dest, 1:p, :), NaN)
    return dest
end

"""
    demean!(Y::Matrix)

In-place subtracts the mean of each column from the corresponding elements in the matrix `Y`.

# Arguments

  - `Y::Matrix`: The matrix to be demeaned.

# Behavior

This function calculates the mean of each column in `Y` and then subtracts in-place
this column mean from every element within the respective column.

```
```
"""
@propagate_inbounds function demean!(Y::Matrix)
    μ = mean(Y; dims=1)
    for j ∈ Base.axes(Y, 2)
        for t ∈ Base.axes(Y, 1)
            Y[t, j] -= μ[j]
        end
    end
end

"""
demean!(dest::Matrix, Y::Matrix)

Subtracts the mean of each column from the corresponding elements in the matrix `Y`
and write in `dest`.
"""
@propagate_inbounds function demean!(dest::Matrix, Y::Matrix)
    μ = mean(Y; dims=1)
    for j ∈ Base.axes(Y, 2)
        for t ∈ Base.axes(Y, 1)
            dest[t, j] = Y[t, j] - μ[j]
        end
    end
end

"""
    demean_from_p!(Y::Matrix, p)

In-place subtracts the mean of each column calculated omitting the first `p` observations
from the corresponding elements in the matrix `Y`.
"""
@propagate_inbounds function demean_from_p!(Y::Matrix, p)
    Yv = @view Y[(p + 1):end, :]
    μ = mean(Yv; dims=1)
    for j ∈ Base.axes(Y, 2)
        for t ∈ Base.axes(Yv, 1)
            Yv[t, j] -= μ[j]
        end
    end
end

## ------------------------------------------------------
## Asymptotic Variance of IRF
## ------------------------------------------------------

const ⊗ = kron

function varsigma(var::VAR)
    ## Return the variance of the coefficient
    ## Σ_{α} = Σ_{ε} ⊗ (X'X)^{-1}
    ## n = nobs(var)
    m = nvars(var)
    Σₑ = residualsvariance(var)
    𝔻 = duplicationmatrix
    𝔻ₘ = 𝔻(m)
    𝔻⁺ₘ = 𝔻ₘ'𝔻ₘ \ 𝔻ₘ'
    return 2 * 𝔻⁺ₘ * (Σₑ ⊗ Σₑ) * 𝔻⁺ₘ'
end

function varcoef(var::VAR)
    ## Return the variance of the coefficient
    ## Σ_{α} = Σ_{ε} ⊗ (X'X)^{-1}
    n, p, m = sizes(var)
    Σ = residualsvariance(var)
    X = var.X[(p + 1):end, :]
    XXinv = (X'X) \ I
    return kron(XXinv, Σ)
end

function StatsBase.stderror(var::VAR)
    _, p, m = sizes(var)
    Σa = varcoef(var)
    return reshape(sqrt.(diag(Σa)), (m * p, m))
end

function G(v::VAR, ir)
    _, p, m = sizes(v)
    H = size(ir, 1)
    F = companionmatrix(v)
    memoization = Dict{Int,Matrix{eltype(ir)}}()
    for h ∈ 0:H
        memoization[h] = ((F')^h)
    end
    J = zeros(eltype(ir), (m, p * m))
    J[1:m, 1:m] .= I(m)
    ## Calculate G_i
    function G_i(i::Int, ir, memoization)
        G = zeros(eltype(ir), (m^2, p * m^2))
        for j ∈ 0:(i - 1)
            A = memoization[i - 1 - j]
            G += J * A ⊗ view(F^j, 1:m, 1:m)
        end
        return G
    end
    return map(x -> G_i(x, ir, memoization), 1:H)
end

function effect_cov(v::VAR, R, ir)
    ## The diagonal elements are the 
    ## (standard errors) of the IRF
    ##  ===============
    n, p, m = sizes(v)
    horizons = size(ir, 1)
    P = IRFs.identifyingmatrix(R)
    P⁻¹ = inv(P)
    Iₘ = I(m)
    𝔾 = IRFs.G(v, ir)
    ## Φ is the memoization of (F^h)[1:m, 1:m]
    𝕃 = IRFs.eliminationmatrix
    𝕂 = IRFs.commutationmatrix
    𝕃ₘ = 𝕃(m)
    𝕂ₘₘ = 𝕂(m, m)

    H = 𝕃ₘ' * inv(𝕃ₘ * (I(m^2) + 𝕂ₘₘ) * (P ⊗ Iₘ) * 𝕃ₘ')
    Σα = varcoef(v) #.* (n - hasintercept(v) - m * p)
    Σα = hasintercept(v) ? Σα[(m + 1):end, (m + 1):end] : Σα
    Σσ = varsigma(v)
    ## The variance is 
    ## CᵢΣαCᵢ + C̄ᵢΣαC̄ᵢ
    ## C₀ = O, Cᵢ = (P'⊗Iₘ)Gᵢ, C̄ᵢ = (Iₘ⊗Φᵢ)H
    ## where Φᵢ are the coefficient of the wald representation
    ## P is the identification matrix
    ## I let Aᵢ = CᵢΣαCᵢ and Bᵢ = C̄ᵢΣαC̄ᵢ
    A = Array{eltype(ir)}(undef, horizons, m^2, m^2)
    A[1, :, :] .= zero(eltype(ir))
    B = similar(A)
    for h ∈ 1:horizons
        if h > 1
            Cᵢ = (P' ⊗ Iₘ)*𝔾[h-1]
            A[h, :, :] .= Cᵢ * Σα * Cᵢ'
        end
        C̄ᵢ = (Iₘ ⊗ (ir[h, :, :]*P⁻¹)) * H
        B[h, :, :] .= C̄ᵢ * Σσ * C̄ᵢ'./nobs(v)
    end
    return A .+ B
end

function irf_se_asy(v::VAR, R, ir)
    n, p, m = sizes(v)
    V = effect_cov(v, R, ir)
    ## V[h, :, :] is the variance of the IRF at horizon h
    ## The standard error is the square root of the diagonal
    ## elements of V[h, :, :]
    return mapslices(x -> reshape(sqrt.(diag(x)), (m, m)), V; dims=(2, 3))
end

function Base.summary(v::VAR)
    _, p, m = sizes(v)
    A = coef(v)
    σ = stderror(v)
    nms = mapreduce(x -> x .* String.(v.names), vcat, "L" .* string.(1:p) .* ".")
    ## Get est tables
    function tbl(j, A, σ, nms)
        α = vec(A[j, :, :])
        se = σ[:, j]
        tt = α ./ se
        pv = 2 * (1 .- StatsFuns.normcdf.(abs.(tt)))
        return CoefTable(hcat(α, se, tt, pv),
                         ["Coef.", "Std. Error", "t", "Pr(>|z|)"], nms, 4, 3)
    end
    stbl = map(j -> tbl(j, A, σ, nms), 1:m)
    hl_pv = Highlighter((data, i, j) -> j == 5 && data.cols[i][j - 2] < 0.05,
                        crayon"bg:white fg:blue")
    #hl_lags = Highlighter(data, i,j) -> j == 1 ? "background-color: #f44336; color: white" : ""

    # return map(x -> pretty_table(DataFrames.DataFrame(x);
    #                              alignment=[:l, :r, :r, :r, :r],
    #                              #highlighters=hl_pv,
    #                              header_crayon=crayon"yellow",
    #                              formatters=ft_printf("%5.3f", 2:5)), stbl)
end
