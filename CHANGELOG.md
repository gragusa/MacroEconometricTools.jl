# Changelog

## Unreleased

### Added

- **`irfplot!` mutating function**: New exported function that draws a single (variable, shock) IRF panel onto an existing `Makie.Axis`. Supports all four IRF result types (`IRFResult`, `SignRestrictedIRFResult`, `BayesianIRFResult`, `LocalProjectionIRFResult`). Enables composing custom multi-panel figures with per-panel styling.

- **`vars`/`shocks` selection for `LocalProjectionIRFResult` (Makie)**: The `irfplot` dispatch for `LocalProjectionIRFResult` now accepts `vars`, `shocks`, `pretty_vars`, and `pretty_shocks` keyword arguments, matching the other three IRF result type dispatches.

- **`vars`/`shocks` selection for `LocalProjectionIRFResult` (Plots.jl)**: The RecipesBase recipe for `LocalProjectionIRFResult` now accepts `vars`, `shocks`, `pretty_vars`, and `pretty_shocks` keyword arguments for subsetting and relabelling.

- **Per-shock `rescale` and `rescale!`**: Rescale IRF results per shock via `Pair{Symbol, Real}` arguments. Example: `rescale(irf, :MP => -1, :Supply => 100)`. Under `UnitStd`, use `=> -1` to fix the sign of a shock. Under `UnitEffect`, use `=> 0.25` for a 25bp shock. When bootstrap draws are available, confidence bands are recomputed from rescaled draws. Works on `IRFResult` and `SignRestrictedIRFResult`.

- **`get_scale` accessor**: Returns an `IRFScale` struct with the normalization scheme, per-shock scale factors, and the impact diagonal (diag of the structural impact matrix P before scaling).

- **`IRFScale` type**: Exported struct with per-shock scaling information.

- **Per-shock `metadata.scale`**: The `scale` field in IRF metadata is now a `Vector` (one entry per shock) instead of a scalar, tracking per-shock cumulative scale factors.

- **`impact_diagonal` in metadata**: The `irf()` function now stores the diagonal of the normalized impact matrix in `metadata.impact_diagonal`, capturing what "one unit of shock j" means for each variable.

### Removed

- **`irf_scale` and `flipshock` from all plotting functions**: The `irfplot`, `irfplot!`, and Plots.jl recipe functions no longer accept `irf_scale` or `flipshock` keyword arguments. All scaling must be done on the IRF result before plotting using `rescale` / `rescale!`.
