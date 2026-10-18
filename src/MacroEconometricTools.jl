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

using LinearAlgebra: LinearAlgebra, Diagonal, I, Symmetric, cholesky, cholesky!,
                     diag, dot, eigen, eigvals, ldiv!, logdet, mul!, pinv, qr,
                     transpose!, LAPACK, SingularException
using FastLapackInterface: FastLapackInterface, QRWs, QROrmWs
using Statistics: Statistics
using Random: Random, AbstractRNG, randn!
using Dates: Dates, now
using Distributed: Distributed
using StatsBase: StatsBase, aic, bic, confint, dof_residual, fit, loglikelihood
using StatsFuns: StatsFuns, norminvcdf
using Distributions: Distributions, Chisq, mean, median, quantile, scale, std
using GLM: GLM, coef, dof, fitted, nobs, residuals, stderror, vcov
using AxisArrays: AxisArrays, AxisArray, Axis
using Tables: Tables

# Core type hierarchy
include("types.jl")
include("constraints.jl")
include("utilities.jl")

# VAR estimation and analysis
include("var/structural_core.jl")
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

# Plotting recipes (will work when Plots.jl is loaded)
include("plots_recipes.jl")

# Stub functions for Makie extension
"""
    irfplot(irf::AbstractIRFResult; kwargs...) -> Figure

Plot impulse response functions using Makie.  Returns a `Makie.Figure`.
Requires a Makie backend (e.g., `CairoMakie` or `GLMakie`) to be loaded.

All scaling should be applied to the IRF result *before* plotting
using `rescale`.

# Selection
- `vars = :all`: variables to plot (`:all`, a vector of names from `varnames(irf)`,
  or a vector of positions)
- `shocks = :all`: shocks to plot (`:all`, names from `shocknames(irf)`, or positions)
- `pretty_vars = nothing`: labels for the selected variables, in selection order
- `pretty_shocks = nothing`: labels for the selected shocks, in selection order

# Appearance
- `bandcolor = :steelblue`: fill colour for confidence / credible bands
- `bandalpha = 0.2`: base opacity for the widest band (narrower bands are darker)
- `linecolor = :black`: colour of the point-estimate line
- `linewidth = 2.0`: width of the point-estimate line
- `drawzero = true`: draw a dashed zero reference line
- `zerolinecolor = :gray60`: colour of the zero line
- `zerolinestyle = :dash`: line style of the zero line
- `xtickstep = 4`: spacing of x-axis ticks (set `0` to disable)

# Paths (results with stored draws, see `has_draws`)
- `plot_type = :quantiles`: `:quantiles`, `:paths`, or `:both`; `:paths` and `:both`
  throw an `ArgumentError` when the result has no draws
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
    irfplot!(ax, irf::AbstractIRFResult; var, shock, kwargs...)

Draw a single IRF panel (one variable–shock pair) onto an existing `Makie.Axis`.
`var` and `shock` are a name from `varnames(irf)` / `shocknames(irf)` or a position.
Returns `ax`.  Requires a Makie backend (e.g., `CairoMakie` or `GLMakie`).

All scaling should be applied to the IRF result *before* plotting
using `rescale`.

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

For results with stored draws, also accepts:
`plot_type` (`:quantiles`, `:paths`, `:both`), `path_alpha`, `path_color`,
`path_linewidth`.
"""
function irfplot! end

# Without a Makie backend, `irfplot`/`irfplot!` have no methods; the hint says
# which package to load.
function _irfplot_hint(io, exc, argtypes, kwargs)
    exc.f === irfplot || exc.f === irfplot! || return
    Base.get_extension(@__MODULE__, :MacroEconometricToolsMakieExt) === nothing || return
    print(
        io, "\n`$(exc.f)` is implemented by the Makie extension; load a Makie ",
        "backend first, e.g. `using CairoMakie` or `using GLMakie`."
    )
    return
end

function __init__()
    Base.Experimental.register_error_hint(_irfplot_hint, MethodError)
    return
end

# Exported: the end-user API. Deprecated names are reachable as
# `MacroEconometricTools.name` but not exported.
export AbstractVARSpec, OLSVAR
export VARModel, VARCoefficients
export VAR
export AbstractConstraint, ZeroConstraint, FixedConstraint, BlockExogeneity
export AbstractIdentification, CholeskyID, SignRestriction, IVIdentification
export AbstractInstrument, ExternalInstrument
export AbstractIRFResult, IRFResult, SignRestrictedIRFResult
export BayesianIRFResult, LocalProjectionIRFResult
export NarrativeShockRestriction, NarrativeRestriction
export point_estimate, mean_estimate, has_draws, n_draws, get_draws
export lowerbounds, upperbounds, coverages, horizon
export IRFScale, get_scale, rescale
export InferenceType, Analytic, WildBootstrap, Bootstrap, BlockBootstrap, ProxySVARMBB
export AbstractNormalization, UnitStd, UnitEffect
export fit, irf
export coef, vcov, stderror, confint, residuals, fitted, dates, residual_covariance
export nobs, dof, dof_residual
export n_vars, n_lags, varnames, shocknames, intercept
export loglikelihood, aic, bic, hqic, select_lags
export forecast
export rotation_matrix, normalize_impact, normalize_impact!
export create_lags
export variance_decomposition, historical_decomposition
export cumulative_irf
export is_stable, long_run_effect, long_run_mean
export confidence_bands
export irfplot, irfplot!
export first_stage_F, iv_summary

# Public, not exported: building blocks for spoke packages and advanced use.
# Matrix-level structural core: operates on raw (B, Σ_L) matrices rather than a
# fitted VARModel, for loops over reduced-form draws (e.g. posterior draws).
public structural_impact, compute_ma_matrices_from_B!, compute_ma_matrices,
       generate_random_orthogonal!, check_sign_restrictions, compute_structural_shocks,
       check_narrative_restrictions, compute_irf_point, n_shocks
public iv_impact, iv_instrument, estimate_proxy_svar, proxy_svar_dynamics
public bootstrap_irf, refit_for_bootstrap, proxy_svar_mbb, msw_confidence_set
public lag, create_lags!, companion_form
public apply_constraints!, build_selection_matrix, check_constraints
public coefficient_covariance, sigma_covariance, irf_jacobian_matrices,
       irf_effect_covariance, irf_asymptotic_stderror
public duplication_matrix, elimination_matrix, commutation_matrix
public simulate_var, simulate_structural

end # module
