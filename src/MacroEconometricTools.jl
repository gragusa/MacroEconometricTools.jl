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
include("ivsvar/bootstrap.jl")
include("ivsvar/msw.jl")

# Simulation and bootstrap
include("simulation.jl")
include("bootstrap.jl")

# Parallel computing utilities
include("parallel.jl")

# Plotting recipes (will work when Plots.jl is loaded)
include("plots_recipes.jl")

# Stub functions for Makie extension
"""
    irfplot(irf::AbstractIRFResult; kwargs...) -> Figure

Plot impulse response functions using Makie.  Returns a `Makie.Figure`.
Requires a Makie backend (e.g., `CairoMakie` or `GLMakie`) to be loaded.

All scaling should be applied to the IRF result *before* plotting
using `rescale` / `rescale!`.

# Selection
- `vars = :all`: variables to plot (`:all`, or `Vector{Symbol}`)
- `shocks = :all`: shocks to plot
- `pretty_vars = nothing`: custom labels for variables
- `pretty_shocks = nothing`: custom labels for shocks

# Appearance
- `bandcolor = :steelblue`: fill colour for confidence / credible bands
- `bandalpha = 0.2`: base opacity for the widest band (narrower bands are darker)
- `linecolor = :black`: colour of the point-estimate line
- `linewidth = 2.0`: width of the point-estimate line
- `drawzero = true`: draw a dashed zero reference line
- `zerolinecolor = :gray60`: colour of the zero line
- `zerolinestyle = :dash`: line style of the zero line
- `xtickstep = 4`: spacing of x-axis ticks (set `0` to disable)

# Paths (SignRestrictedIRFResult, BayesianIRFResult only)
- `plot_type = :quantiles`: `:quantiles`, `:paths`, or `:both`
- `path_alpha = 0.02`: opacity of individual draw lines
- `path_color = :gray`: colour of individual draw lines
- `path_linewidth = 0.5`: width of individual draw lines

# Layout
- `figure = (;)`: keyword arguments forwarded to `Makie.Figure`
- `size = nothing`: shorthand for figure size, e.g. `(900, 600)`
- `title = nothing`: super-title above the grid
- `title_fontsize = 20`: super-title font size
- `linkxaxes = true`: link x-axes across all panels
- `linkyaxes = :row`: link y-axes (`:row`, `:all`, or `false`)
- `colgap = 10`: horizontal gap between panels
- `rowgap = 10`: vertical gap between panels

# Examples
```julia
using MacroEconometricTools, CairoMakie

# Basic usage
fig = irfplot(result)

# Rescale to percentage points, then plot with custom labels
result_pct = rescale(result, 100)
fig = irfplot(result_pct;
    pretty_vars  = ["Output", "Prices"],
    pretty_shocks = ["Demand", "Supply"],
    size = (1000, 700),
    title = "Structural IRFs")
save("irfs.pdf", fig)
```
"""
function irfplot end

"""
    irfplot!(ax, irf::AbstractIRFResult; var::Symbol, shock::Symbol, kwargs...)

Draw a single IRF panel (one variable–shock pair) onto an existing `Makie.Axis`.
Returns `ax`.  Requires a Makie backend (e.g., `CairoMakie` or `GLMakie`).

All scaling should be applied to the IRF result *before* plotting
using `rescale` / `rescale!`.

Use this to compose custom multi-panel figures:

```julia
using CairoMakie, MacroEconometricTools

result_pct = rescale(result, 100)
fig = Figure(size = (1200, 400))
ax1 = Axis(fig[1, 1]; title = "GDP")
ax2 = Axis(fig[1, 2]; title = "Inflation")
irfplot!(ax1, result_pct; var = :GDP, shock = :MP, bandcolor = :steelblue)
irfplot!(ax2, result_pct; var = :Inflation, shock = :MP, bandcolor = :indianred)
linkxaxes!(ax1, ax2)
save("composed.pdf", fig)
```

Accepts the same appearance keywords as `irfplot`:
`bandcolor`, `bandalpha`, `linecolor`, `linewidth`,
`drawzero`, `zerolinecolor`, `zerolinestyle`, `xtickstep`.

For set-identified / Bayesian results, also accepts:
`plot_type` (`:quantiles`, `:paths`, `:both`), `path_alpha`, `path_color`,
`path_linewidth`.
"""
function irfplot! end

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
export IRFScale, get_scale, rescale, rescale!

# Export inference types
export InferenceType, Analytic, WildBootstrap, Bootstrap, BlockBootstrap, ProxySVARMBB
export AbstractNormalization, UnitStd, UnitEffect

# Export main functions
export fit
export irf, impulse_response
export coef, vcov, residuals, fitted
export n_vars, n_lags, effective_obs, varnames, intercept
# StatsBase methods (not exported, use via StatsBase): nobs, dof, dof_residual, modelmatrix, rss
export log_likelihood, aic, bic, hqic
export bootstrap_irf, bootstrap_irf_wild, bootstrap_irf_standard, bootstrap_irf_block
export forecast
export rotation_matrix, normalize, normalize!
export lag, create_lags, create_lags!
export companion_form
export variance_decomposition, historical_decomposition
export cumulative_irf
export is_stable, long_run_effect, long_run_mean
export confidence_bands
export irfplot, irfplot!
export first_stage_F, iv_summary, refit_for_bootstrap

end # module
