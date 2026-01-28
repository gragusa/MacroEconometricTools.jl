"""
MacroEconometricTools.jl

A modern Julia package for macroeconometric analysis, featuring:
- Vector Autoregressions (VAR) with OLS and Bayesian estimation
- Instrumental Variable Structural VAR (IV-SVAR)
- Local Projections (LP)
- Flexible constraint systems
- Multiple identification schemes
- Bootstrap and asymptotic inference
"""
module MacroEconometricTools

using LinearAlgebra
using Statistics
using Random
using Dates
using Distributed
using StatsBase
using StatsBase: fit
using StatsFuns
using Distributions
using CovarianceMatrices
using GLM
using AxisArrays
using Tables

# Core type hierarchy
include("types.jl")
include("constraints.jl")
include("utilities.jl")

# VAR estimation and analysis
include("var/estimation.jl")
include("var/inference.jl")
include("var/identification.jl")
include("var/irfs.jl")

# Local projections
# include("localprojection/estimation.jl")
# include("localprojection/inference.jl")

# IV-SVAR
include("ivsvar/instruments.jl")
include("ivsvar/estimation.jl")
include("ivsvar/diagnostics.jl")

# Simulation and bootstrap
include("simulation.jl")
include("bootstrap.jl")

# Parallel computing utilities
include("parallel.jl")

# Plotting recipes (will work when Plots.jl is loaded)
include("plots_recipes.jl")

# Export main types
export AbstractVARSpec, OLSVAR, BayesianVAR, IVSVAR, LocalProjection
export VARModel, VARCoefficients
export VAR
export AbstractConstraint, ZeroConstraint, FixedConstraint, BlockExogeneity
export AbstractIdentification, CholeskyID, SignRestriction, IVIdentification
export AbstractInstrument, ExternalInstrument, ProxyIV
export AbstractIRFResult, IRFResult, SignRestrictedIRFResult
export BayesianIRFResult, LocalProjectionIRFResult
export NarrativeShockRestriction, NarrativeRestriction
export point_estimate, mean_estimate, has_draws, n_draws, get_draws
export lowerbounds, upperbounds, coverages, horizon

# Export inference types
export InferenceType, Analytic, WildBootstrap, Bootstrap, BlockBootstrap

# Export main functions
export fit
export irf, impulse_response
export coef, vcov, residuals, fitted
export n_vars, n_lags, effective_obs, varnames, intercept
# StatsBase methods (not exported, use via StatsBase): nobs, dof, dof_residual, modelmatrix, rss
export log_likelihood, aic, bic, hqic
export bootstrap_irf
export forecast
export rotation_matrix, normalize, normalize!
export lag, create_lags, create_lags!
export companion_form
export variance_decomposition, historical_decomposition
export cumulative_irf
export is_stable, long_run_effect, long_run_mean
export confidence_bands

end # module
