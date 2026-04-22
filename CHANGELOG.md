# Changelog

## Unreleased

### Added

- **Proxy-SVAR `target_shock` works for any variable position**: `IVIdentification(Z, target)`, `ExternalInstrument(Z; target_shock=...)`, and `ProxyIV(proxies; target_shock=...)` now produce correct IRFs, MSW confidence sets, and Jentsch-Lunsford MBB inference regardless of which variable is instrumented. Previously the instrumented variable had to be ordered first; bootstrap dynamics, AR anchoring, band placement, and MSW quadratic coefficients all hardcoded position 1. The estimation layer was already parametric — the fix propagated `target` through `proxy_svar_dynamics`, `_proxy_svar_mbb_impl`, `_ar_confidence_sets`, the `compute_inference_bands` dispatch, and `msw_confidence_set`.

- **`ProxyIV` constructor parity with `ExternalInstrument`**: `ProxyIV` now accepts `target_shock` as an `Int` or `Symbol` via keyword (`ProxyIV(proxies; target_shock=:FFR)`) or positional (`ProxyIV(proxies, 2)`) arguments, matching `ExternalInstrument`'s surface. The struct is parameterized as `ProxyIV{T, S}` with a scalar `target_shock::S` field (`Vector{Int}` storage dropped; single-shock identification was already enforced downstream).

- **`estimate_proxy_svar!` and `proxy_svar_dynamics!`**: In-place variants of the Jentsch-Lunsford inner-loop kernels, plus `ProxySvarBuffers` and `ProxyDynamicsBuffers` workspace structs. Used inside `proxy_svar_mbb` to eliminate per-draw allocations.

- **`irfplot!` mutating function**: New exported function that draws a single (variable, shock) IRF panel onto an existing `Makie.Axis`. Supports all four IRF result types (`IRFResult`, `SignRestrictedIRFResult`, `BayesianIRFResult`, `LocalProjectionIRFResult`). Enables composing custom multi-panel figures with per-panel styling.

- **`vars`/`shocks` selection for `LocalProjectionIRFResult` (Makie)**: The `irfplot` dispatch for `LocalProjectionIRFResult` now accepts `vars`, `shocks`, `pretty_vars`, and `pretty_shocks` keyword arguments, matching the other three IRF result type dispatches.

- **`vars`/`shocks` selection for `LocalProjectionIRFResult` (Plots.jl)**: The RecipesBase recipe for `LocalProjectionIRFResult` now accepts `vars`, `shocks`, `pretty_vars`, and `pretty_shocks` keyword arguments for subsetting and relabelling.

- **Per-shock `rescale` and `rescale!`**: Rescale IRF results per shock via `Pair{Symbol, Real}` arguments. Example: `rescale(irf, :MP => -1, :Supply => 100)`. Under `UnitStd`, use `=> -1` to fix the sign of a shock. Under `UnitEffect`, use `=> 0.25` for a 25bp shock. When bootstrap draws are available, confidence bands are recomputed from rescaled draws. Works on `IRFResult` and `SignRestrictedIRFResult`.

- **`get_scale` accessor**: Returns an `IRFScale` struct with the normalization scheme, per-shock scale factors, and the impact diagonal (diag of the structural impact matrix P before scaling).

- **`IRFScale` type**: Exported struct with per-shock scaling information.

- **Per-shock `metadata.scale`**: The `scale` field in IRF metadata is now a `Vector` (one entry per shock) instead of a scalar, tracking per-shock cumulative scale factors.

- **`impact_diagonal` in metadata**: The `irf()` function now stores the diagonal of the normalized impact matrix in `metadata.impact_diagonal`, capturing what "one unit of shock j" means for each variable.

### Changed

- **Renamed `estimate_proxy_svar_python` → `estimate_proxy_svar`**: The `_python` suffix was a stale port artifact. Internal function; no public API impact.

### Fixed

- **Proxy-SVAR: hardcoded position-1 assumptions**: The Jentsch-Lunsford MBB bootstrap dynamics, AR confidence set anchor, band placement in `compute_inference_bands`, and the MSW confidence set quadratic (Wald statistic, `a`/`b` coefficients, impact-normalization row) all hardcoded `[1]` / `[1, 1]`. Any `target_shock ≠ 1` silently produced wrong numbers. Now parameterized on `target`.

### Performance

- **Proxy-SVAR MBB bootstrap: −96% memory, −98% allocations, −17% wall time**. The inner loop now reuses `ProxySvarBuffers` / `ProxyDynamicsBuffers` across draws, and the in-place `estimate_proxy_svar!` solves OLS via `cholesky!(X'X)` + `ldiv!` (instead of `xx \ yy`), computes residuals via `mul!(U, X, A, -1, 1)`, and uses `mul!` everywhere for the covariance and identification formulas. Benchmark on a 3-variable VAR, `horizon=20`, `reps=2000`: **226 MiB → 9.9 MiB** total allocation, **145 k → 3 k** total allocs, **72 → 1.5** allocs per draw, **14% → 0.2%** GC fraction, **90.6 ms → 75.4 ms** wall time. Verified bit-exact against the 170-assertion Python cross-validation suite.

### Removed

- **`irf_scale` and `flipshock` from all plotting functions**: The `irfplot`, `irfplot!`, and Plots.jl recipe functions no longer accept `irf_scale` or `flipshock` keyword arguments. All scaling must be done on the IRF result before plotting using `rescale` / `rescale!`.
