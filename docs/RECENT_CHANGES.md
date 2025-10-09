# Recent Changes Since Last Commit

## StatsBase.jl API Alignment (2025-01)

### **BREAKING CHANGES**

#### Function Renaming
- **`estimate()` → `StatsBase.fit()`**: All model estimation now uses the standard StatsBase.jl convention
  - **Migration**: Replace `estimate(OLSVAR, Y, 4)` with `fit(OLSVAR, Y, 4)`
  - **Rationale**: Aligns with Julia ecosystem standards (StatsBase, GLM, etc.)
  - The `VAR()` convenience wrapper remains unchanged for backward compatibility

#### Accessor Method Renaming
- **`n_obs()` → `effective_obs()`**: Number of observations used in estimation (after accounting for lags)
  - **Migration**: Replace `n_obs(model)` with `effective_obs(model)`
  - **Rationale**: More descriptive name; distinguishes from StatsBase.jl `nobs()`

- **`raw_nobs()` → `StatsBase.nobs()`**: Total observations in original data
  - **Migration**: Replace `raw_nobs(model)` with `nobs(model)`
  - **Rationale**: Follows StatsBase.jl convention

#### New StatsBase.jl Interface Methods
- **`StatsBase.dof(model)`**: Degrees of freedom (number of estimated parameters)
  - For VAR(p) with n variables: `n × (1 + n × p)` parameters

- **`StatsBase.dof_residual(model)`**: Residual degrees of freedom
  - Calculated as: `effective_obs(model) - dof(model)`

- **`StatsBase.modelmatrix(model)`**: Design matrix X used in estimation
  - Returns matrix of size `(T × (1 + n_vars × n_lags))`

- **`StatsBase.rss(model)`**: Residual sum of squares
  - Sum of squared residuals across all equations

#### New Custom Accessors
- **`intercept(model)`**: Extract intercept coefficients
  - Returns vector of length `n_vars(model)`

### Summary of API Changes

| Old API | New API | Type |
|---------|---------|------|
| `estimate(OLSVAR, Y, 4)` | `fit(OLSVAR, Y, 4)` | **BREAKING** |
| `n_obs(model)` | `effective_obs(model)` | **BREAKING** |
| `raw_nobs(model)` | `nobs(model)` | **BREAKING** |
| N/A | `dof(model)` | New |
| N/A | `dof_residual(model)` | New |
| N/A | `modelmatrix(model)` | New |
| N/A | `rss(model)` | New |
| N/A | `intercept(model)` | New |

### Migration Example

```julia
# Old API
var = estimate(OLSVAR, Y, 4)
n_eff = n_obs(var)
n_tot = raw_nobs(var)

# New API
var = fit(OLSVAR, Y, 4)
n_eff = effective_obs(var)
n_tot = nobs(var)

# New functionality
params = dof(var)                    # Number of parameters
df_resid = dof_residual(var)         # Residual degrees of freedom
X = modelmatrix(var)                 # Design matrix
rss_val = rss(var)                   # Residual sum of squares
c = intercept(var)                   # Intercept vector
```

## Type System Refactoring (2025-01)
- **BREAKING**: Renamed `identify()` → `rotation_matrix()` for clarity - the method returns the structural impact matrix P, not the full identification
- Introduced `AbstractIRFResult{T}` supertype to enable polymorphic dispatch on IRF results
- `IRFResult{T}` and `SignRestrictedIRFResult{T}` now share a common abstract parent
- Updated all accessor methods (`horizon()`, `n_vars()`, `n_shocks()`, `size()`) to dispatch on `AbstractIRFResult`
- Added `cumulative_irf()` method for `SignRestrictedIRFResult`
- Unified plotting recipe helpers to work with `AbstractIRFResult`

## API Enhancements
- Added `varnames(model)` accessor to retrieve variable names
- Added `raw_nobs(model)` accessor for total observations before lags
- Enhanced `irf()` for sign restrictions with `n_draws` and `max_attempts` parameters
- `SignRestrictedIRFResult` now stores all rotation matrices and IRF draws for full set identification

## Documentation Overhaul
- Updated all tutorials to use `rotation_matrix()` instead of `identify()`
- Added comprehensive sign restriction plotting guide (`:paths`, `:quantiles`, `:both` modes)
- Documented the type hierarchy design rationale in TECHNICAL.md
- Updated README and QUICK_REFERENCE with new accessor methods
- Added model properties section to getting_started tutorial

## Benchmarking & Contributor Guidance
- Added a dedicated benchmarking environment with reusable suites and an AirspeedVelocity runner (`benchmark/benchmarks.jl`, `benchmark/run_asv.jl`, `benchmark/Project.toml:1`).
- Documented contributor expectations and benchmark usage in `AGENTS.md` and the README benchmark section (`README.md:145`).

## Deterministic Randomness & Bootstrap Fixes
- Threaded serial/distributed bootstrap paths now accept caller-provided RNGs; residual resampling respects those seeds (`src/bootstrap.jl:38`, `src/bootstrap.jl:170`).
- Sign-restriction search (serial + distributed) inherits RNG semantics and respects constraint counts (`src/var/identification.jl:114`).

## Metadata & Compatibility
- Reintroduced the `VAR` convenience wrapper to mirror the legacy IRFs interface (`src/var/estimation.jl:185`).
- IRF results carry variable names in metadata for downstream tooling (`src/var/irfs.jl:79`).

## Plotting Recipes
- Refreshed the Plots.jl recipe to work with `AbstractIRFResult` and metadata-driven labels (`src/plots_recipes.jl:1`).
- Added an optional Makie extension with an `irfplot` recipe for grid layouts and confidence bands (`ext/MacroEconometricToolsMakieExt.jl:1`).

## Tests & Dependency Updates
- Trimmed `test/runtests.jl` to load only the maintained smoke suite; added deterministic bootstrap coverage in `test/test_basic.jl:67`.
- Promoted `StableRNGs` and `Distributed` to core dependencies; registered `Makie` as an optional extension (`Project.toml:10`).
- Optional plotting recipes are now loaded via conditional includes (`src/MacroEconometricTools.jl:48`).
