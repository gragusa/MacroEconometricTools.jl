# Recent Changes Since Last Commit

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
