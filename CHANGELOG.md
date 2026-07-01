# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
(with the [Julia 0.x convention](https://pkgdocs.julialang.org/v1/compatibility/#Version-specifier-format):
while `0.x`, minor bumps may carry breaking changes and patch bumps are
backward-compatible).

## [Unreleased]

### Added

### Changed

### Deprecated

### Removed

### Fixed

### Performance

## [0.1.1] - 2026-06-30

First registered release. Establishes the structural-identification core
(Cholesky, sign, narrative, and IV/proxy schemes), the shock-normalization and
rescaling API, and the fast bootstrap paths, and documents the hub/spoke type
surface.

### Added

- **Narrative-restriction IRF identification.** A constructed `NarrativeRestriction`
  now produces an IRF result. `irf(model, ::NarrativeRestriction)` runs a rejection
  search — sign restrictions on the impact/IRF combined with narrative sign
  constraints on the structural shocks — and returns a `SignRestrictedIRFResult`.
  Narrative restrictions reference shocks by residual-row index or by date; dates
  are supplied once at fit time via `fit(OLSVAR, Y, p; dates=…)`, stored lag-trimmed
  so they align with residual rows, and read automatically. New exported accessor
  `dates(model)` returns the stored dates (or `nothing`). A date-based restriction
  on a date-less model fails fast.

- **Shock-normalization API.** Two-layer control over IRF units — pick a *scheme*
  at estimation time, then optionally apply per-shock *rescaling* afterwards.
  - *Normalization scheme* — `irf(model, id; normalization=…)` takes an
    `AbstractNormalization`. `UnitStd()` (default): unit-variance shocks
    (`P*P' = Σ` under Cholesky; on impact, shock `j` moves variable `j` by one
    residual standard deviation). `UnitEffect()`: unit own-variable impact
    (`P[j, j] = 1`), natural for "1 percentage point of the monetary policy shock"
    under a recursive ordering. The scheme is fixed at estimation time and stored
    in `metadata.normalization`. `UnitStd` / `UnitEffect` / `AbstractNormalization`
    are exported, as are the low-level `normalize` / `normalize!` utilities that
    apply a scheme to an impact matrix `P`.
  - *Per-shock rescaling* — `rescale(irf, :MP => -1, :Supply => 100)` multiplies
    each named shock by a scalar; `rescale!` mutates in place. When bootstrap draws
    are available (`save_draws=true`), bands are recomputed from rescaled draws;
    otherwise bands are rescaled approximately with a warning. Works on `IRFResult`
    and `SignRestrictedIRFResult`.
  - *Introspection* — `get_scale(irf)` returns an exported `IRFScale` struct
    (`normalization`, per-shock `scale`, `impact_diagonal`, `names`) with a custom
    `show` printing `impact_diag × scale = effective on-impact` per shock.

- **Per-shock `metadata.scale`.** The `scale` field in IRF metadata is now a
  `Vector` (one entry per shock) tracking per-shock cumulative scale factors
  composed by successive `rescale` calls.

- **`impact_diagonal` in metadata.** `irf()` stores the diagonal of the normalized
  impact matrix in `metadata.impact_diagonal` — what "one unit of shock `j`" means
  for each variable before any `rescale`.

- **Proxy-SVAR `target_shock` works for any variable position.**
  `IVIdentification(Z, target)`, `ExternalInstrument(Z; target_shock=…)`, and
  `ProxyIV(proxies; target_shock=…)` now produce correct IRFs, MSW confidence sets,
  and Jentsch-Lunsford MBB inference regardless of which variable is instrumented.
  The fix propagates `target` through `proxy_svar_dynamics`, `_proxy_svar_mbb_impl`,
  `_ar_confidence_sets`, the `compute_inference_bands` dispatch, and
  `msw_confidence_set`.

- **`ProxyIV` constructor parity with `ExternalInstrument`.** `ProxyIV` accepts
  `target_shock` as an `Int` or `Symbol` via keyword (`ProxyIV(proxies; target_shock=:FFR)`)
  or positional (`ProxyIV(proxies, 2)`) argument. The struct is parameterized as
  `ProxyIV{T, S}` with a scalar `target_shock::S` field.

- **Matrix-level `iv_impact` / `iv_instrument`.** Exported lower-level building
  blocks for per-draw IV identification: `iv_impact(ν, Σ, Z, target)` returns the
  impact matrix and first-stage F from raw reduced-form inputs, and
  `iv_instrument(id::IVIdentification, n_obs, n_lags, names)` resolves an
  identification to its instrument matrix and target index. Intended for spoke
  packages that loop over reduced-form draws.

- **`estimate_proxy_svar!` and `proxy_svar_dynamics!`.** In-place variants of the
  Jentsch-Lunsford inner-loop kernels, plus `ProxySvarBuffers` and
  `ProxyDynamicsBuffers` workspace structs, used inside `proxy_svar_mbb` to
  eliminate per-draw allocations.

- **`simulate_var!`.** In-place variant of `simulate_var` writing into a
  caller-supplied output matrix; `simulate_var` now delegates to it. Used across the
  Cholesky-SVAR bootstrap functions to reuse a single simulation buffer.

- **`_BootstrapWorkspace`, `_fast_refit_ols!`, `_fast_cholesky_irf!` (internal).**
  Pre-allocated workspace and in-place kernels replacing the
  `fit(OLSVAR)` + `rotation_matrix` + `compute_irf_point` chain on the
  Cholesky-identification fast path. Both kernels benchmark at 0 bytes per call and
  support `UnitStd` / `UnitEffect` normalization.

- **`irfplot!` mutating function.** Exported; draws a single (variable, shock) IRF
  panel onto an existing `Makie.Axis`. Supports all four IRF result types and
  enables composing custom multi-panel figures.

- **`vars` / `shocks` selection for `LocalProjectionIRFResult`.** Both the Makie
  `irfplot` dispatch and the Plots.jl RecipesBase recipe now accept `vars`,
  `shocks`, `pretty_vars`, and `pretty_shocks` for subsetting and relabelling,
  matching the other IRF result types.

### Changed

- **`SignRestrictedIRFResult.identification`** widened from `SignRestriction` to
  `AbstractIdentification`, so the field holds either a sign or a narrative
  restriction.

- **Hub/spoke docstrings.** Type docstrings for `BayesianVAR`, `MinnesotaPrior`,
  and `BayesianIRFResult` (provided by `BayesianVAR.jl`) and `LocalProjection` and
  `LocalProjectionIRFResult` (provided by `LocalProjections.jl`) note that this hub
  package defines only the shared type; estimation lives in the spoke. The
  `bootstrap_irf*` family docstrings clarify that they return raw bootstrap IRF
  draws, whereas `irf(model, id; inference=…)` runs the same bootstrap and returns
  an assembled `IRFResult` with bands.

### Removed

- **`irf_scale` and `flipshock` from all plotting functions.** `irfplot`,
  `irfplot!`, and the Plots.jl recipe no longer accept `irf_scale` or `flipshock`.
  All scaling must be done on the IRF result before plotting via `rescale` /
  `rescale!`.

### Fixed

- **Proxy-SVAR hardcoded position-1 assumptions.** The Jentsch-Lunsford MBB
  bootstrap dynamics, the AR confidence-set anchor, band placement in
  `compute_inference_bands`, and the MSW confidence-set quadratic all hardcoded
  `[1]` / `[1, 1]`, so any `target_shock ≠ 1` silently produced wrong numbers. Now
  parameterized on `target`.

- **Cumulative-IRF test coverage.** Added regression tests for the `cumulate=`
  keyword and `cumulative_irf` across sign-restricted and Cholesky IRF results,
  which previously had no coverage.

### Performance

- **Proxy-SVAR MBB bootstrap: −96% memory, −98% allocations, −17% wall time.** The
  inner loop reuses `ProxySvarBuffers` / `ProxyDynamicsBuffers` across draws, and
  in-place `estimate_proxy_svar!` solves OLS via `cholesky!(X'X)` + `ldiv!`,
  computes residuals via `mul!(U, X, A, -1, 1)`, and uses `mul!` throughout.
  Benchmark (3-variable VAR, `horizon=20`, `reps=2000`): 226 MiB → 9.9 MiB total
  allocation, 145 k → 3 k total allocs, 72 → 1.5 allocs per draw, 14% → 0.2% GC,
  90.6 ms → 75.4 ms. Verified bit-exact against the 170-assertion Python
  cross-validation suite.

- **Cholesky-SVAR bootstraps (`WildBootstrap`, `Bootstrap`, `BlockBootstrap`):
  ~1.6× faster, ~11× less memory.** For `CholeskyID` with default ordering, the
  inner loop uses an in-place pipeline over a single `_BootstrapWorkspace{T}`
  (`simulate_var!` → `_fast_refit_ols!` → `_fast_cholesky_irf!`) with no per-rep
  `VARModel`, lagged-X, residual, companion-form, or MA-coefficient allocation.
  Non-Cholesky identifications keep the original `fit(OLSVAR)` path unchanged.
  Benchmark (4-variable VAR(2), `T=150`, `horizon=20`, `reps=300`): 11.86 ms →
  7.33 ms (1.62×), 49.0 MB → 4.3 MB (11.4×) for `WildBootstrap`; per-rep residual
  allocation ≈160 KB → ≈240 B. Verified bit-identical across repeated seeded runs.

- **Bootstrap simulation-buffer reuse.** Even on the non-Cholesky generic path,
  `Y_boot` is allocated once outside the rep loop and written by `simulate_var!` in
  place, saving one T×K allocation per rep.

- **Proxy-SVAR MBB micro-cleanup.** Replaced three per-rep slice-broadcast
  assignments with `copyto!(view(…), …)`, removing a fixable source of per-rep
  allocation.

[Unreleased]: https://github.com/gragusa/MacroEconometricTools.jl/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/gragusa/MacroEconometricTools.jl/releases/tag/v0.1.1
