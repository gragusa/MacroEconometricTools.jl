# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
(with the [Julia 0.x convention](https://pkgdocs.julialang.org/v1/compatibility/#Version-specifier-format):
while `0.x`, minor bumps may carry breaking changes and patch bumps are
backward-compatible).

## [Unreleased]

### Added

- **`AbstractIRFResult` interface.** Every result type implements
  `point_estimate`, `lowerbounds`, `upperbounds`, `coverages`, `varnames`, and
  `shocknames` (new, exported); `has_draws`, `n_draws`, and `get_draws` are
  optional. `horizon`, `n_vars`, `n_shocks`, `size`, `show`, `irfplot`,
  `irfplot!`, and the Plots recipe are defined once for any subtype in terms
  of these methods, so a new result type works with them once it implements
  the required methods. The interface is documented in the
  `AbstractIRFResult` docstring.
- `BayesianIRFResult` and `LocalProjectionIRFResult` gain `size`, `show`, and
  `n_shocks`.
- Instruments observed on part of the sample. `ExternalInstrument` and
  `IVIdentification` accept `missing` entries for periods without an
  instrument value. The reduced-form VAR uses the full sample; the 2SLS impact
  column, the first-stage F, and `iv_impact` use only the periods in which the
  instrument is observed. `NaN`/`Inf` entries throw an `ArgumentError`.
  `ProxySVARMBB` and `msw_confidence_set` throw when the instrument has missing
  values.

### Removed (breaking)

- The estimation specifications `BayesianVAR`, `LocalProjection`,
  `AbstractPrior`, `MinnesotaPrior`, and `NormalWishartPrior`. Nothing in this
  package estimated them, and their exported names clashed with the names
  exported by BayesianVAR.jl and LocalProjections.jl, which define their own.
  The shared result types `BayesianIRFResult` and `LocalProjectionIRFResult`
  remain.

### Deprecated

- `fit(IVSVAR, Y, p; instrument)`, the `IVSVAR` specification, and
  `IVIdentification()` without an instrument. Fit the reduced form with
  `fit(OLSVAR, Y, p)` and pass the instrument to the identification:
  `irf(model, IVIdentification(Z, target_shock))`,
  `first_stage_F(model, id)`, `iv_summary(model, id)`.
- `ProxyIV` is a deprecated name for `ExternalInstrument`.

### Changed (breaking)

- One instrument type, `ExternalInstrument{T, S}` with fields `Z` and
  `target_shock`. `ProxyIV` (field `proxies`) is merged into it.
- The `method` keyword (`:tsls`, `:liml`, `:fuller`) of `ExternalInstrument`
  and `IVIdentification`, and the `relevance_threshold` keyword of `ProxyIV`,
  are removed: they were accepted but had no effect (the estimator is always
  2SLS). `iv_summary` reports the first-stage F and flags F < 10.
- An instrument must have as many rows as the data `Y` (the first `n_lags`
  rows are dropped) or as many rows as the VAR residuals; any other length
  throws a `DimensionMismatch`. Previously a longer instrument was silently cut
  to its last rows.
- `WildBootstrap`, `Bootstrap`, and `BlockBootstrap` inference with
  `IVIdentification` throw an `ArgumentError` pointing to `ProxySVARMBB`: they
  do not resample the instrument. `BlockBootstrap` was previously replaced by
  `ProxySVARMBB` without notice.
- `Analytic()` inference throws an `ArgumentError` unless the identification is
  `CholeskyID()` in the model's variable order; the delta method assumes a
  lower-triangular impact matrix.
- Bootstrap failure policy (`WildBootstrap`, `Bootstrap`, `BlockBootstrap`,
  `ProxySVARMBB`, `bootstrap_irf*`, `proxy_svar_mbb`). A replication counts as
  failed only when it raises `PosDefException`, `SingularException`,
  `LAPACKException`, or `DomainError`, or yields non-finite responses; any
  other exception propagates. Failed replications are dropped: the returned
  draws contain only successful replications, and bands, standard errors, and
  centering use only those. A warning reports the count. The call throws when
  more than `max_failure_rate` (new keyword of the four inference types and of
  `bootstrap_irf*`, default 0.1) of the replications fail, or when all fail.
- `ExternalInstrument(Z, :name)` and `IVIdentification(Z, :name)` accept a
  `Symbol` target positionally, for vector and matrix `Z`; integer-valued `Z`
  is converted to floating point.

- `get_draws` throws an `ArgumentError` for results without draws: an
  `IRFResult` computed without a bootstrap (previously returned `nothing`) and
  `LocalProjectionIRFResult` (previously a `MethodError`).
- Plotting (Plots recipe, `irfplot`, `irfplot!`):
  - `pretty_vars` / `pretty_shocks` label the *selected* variables and shocks,
    in selection order. For `IRFResult` and `SignRestrictedIRFResult` they
    previously had to label every variable.
  - Shocks are selected by `shocknames(irf)` instead of the variable names.
  - A selection entry that is not present throws an `ArgumentError`
    (previously an `ErrorException` for `IRFResult`/`SignRestrictedIRFResult`
    and a confusing downstream error for the other types). `vars`/`shocks`
    and `var`/`shock` also accept integer positions for every result type.
  - `plot_type = :paths` or `:both` works for any result with stored draws
    (including bootstrapped `IRFResult`) and throws an `ArgumentError` when
    there are none (previously ignored for `IRFResult` and
    `LocalProjectionIRFResult`).
  - The Plots recipe draws bands widest-first with the same colors and
    transparency for every result type; the point-estimate line color is
    `median_color` (default `:black`) unless `linecolor` is given.
- Type layout. Every field of `VARModel`, `IRFResult`, `SignRestrictedIRFResult`,
  `BayesianIRFResult`, and `LocalProjectionIRFResult` except `metadata` has a
  concrete type:
  - `VARModel{T, S, D}` has a `dates` field (`nothing` or a vector aligned with
    the residual rows, checked on construction); `dates(model)` reads it. The
    `metadata` key `dates` is removed.
  - `VARCoefficients{T}` drops its constraint type parameter; `constraints` is
    `nothing` or a `Vector{AbstractConstraint}`.
  - `IRFResult{T, A, D, I, N}` and `SignRestrictedIRFResult{T, A, D, I}` gain
    trailing parameters for the draws, identification, and inference types.
  - `BayesianIRFResult{T, A, I, B}` and `LocalProjectionIRFResult{T, A, S, B}`
    gain trailing parameters for the band (and standard-error) array types.
    `BayesianIRFResult{T, A, I}(...)` and `LocalProjectionIRFResult{T, A}(...)`
    still construct them, and new outer constructors without type parameters
    infer every parameter.

  The `metadata` keys of `IRFResult` and `SignRestrictedIRFResult` that the
  package reads are documented in their docstrings.
- IRF arrays use one layout, `(variable, shock, horizon)`, with the draw
  dimension first when present, matching the `IRFResult` axes and the MA
  arrays of `compute_ma_matrices_from_B!`. Affected:
  - `bootstrap_irf`, `bootstrap_irf_wild`, `bootstrap_irf_standard`, and
    `bootstrap_irf_block` return `(reps, n_vars, n_shocks, horizon+1)`
    (previously `(reps, horizon+1, n_vars, n_shocks)`).
  - `compute_irf_point` and `irf_asymptotic_stderror` return
    `(n_vars, n_shocks, horizon+1)`; `irf_effect_covariance` returns
    `(n_vars^2, n_vars^2, horizon+1)`.
- `variance_decomposition` returns an `AxisArray` with axes
  `(:variable, :shock, :horizon)` over forecast horizons `1:H+1`. Its keyword
  is `horizon` (as documented; previously `horizon_spec`). It throws an
  `ArgumentError` unless the shocks have unit variance (it rejects
  `normalization = UnitEffect()`, a non-unit `scale`, and rescaled results)
  and no variable is cumulated.
- Julia 1.13 or later is required.
- Model-level and matrix-level structural identification share one
  implementation. `rotation_matrix(model, ::CholeskyID)` calls
  `structural_impact`; `compute_ma_matrices` and the model-level
  `check_sign_restrictions` use `compute_ma_matrices_from_B!`;
  `generate_random_orthogonal`, `estimate_proxy_svar`, and
  `proxy_svar_dynamics` call their in-place forms. Results agree with the
  previous ones to rounding error; random draws are unchanged for a given RNG.
- `structural_impact(Σ_L, id)` applies an integer `id.ordering` when
  `ordering_idx` is not given. It throws an `ArgumentError` when
  `ordering_idx` contradicts an integer `id.ordering`, or when `id.ordering`
  is given by variable name and `ordering_idx` is missing.
- The matrix-level `check_sign_restrictions` throws a `DimensionMismatch` when
  `Φ` covers fewer horizons than the restrictions; `compute_ma_matrices_from_B!`
  checks the sizes of `Φ`, `B`, and `temp`.
- The proxy-SVAR estimator (`estimate_proxy_svar`, used by `ProxySVARMBB`)
  solves the least-squares step by Householder QR with preallocated LAPACK
  workspaces (FastLapackInterface.jl) instead of the normal equations, and
  throws `SingularException` for a numerically rank-deficient design. It
  accepts `Float32` and `Float64` data.

- Constraints are resolved by one routine shared by `fit`,
  `check_constraints`, `apply_constraints!`, and `build_selection_matrix`. An
  unknown variable name or a lag outside `1:n_lags` throws an `ArgumentError`
  on every path, and so do two constraints that set the same coefficient to
  different values. `build_selection_matrix` treats coefficients fixed by a
  `FixedConstraint` as not free.
- `irf(model, ::SignRestriction)` and `irf(model, ::NarrativeRestriction)`
  share one implementation and validate `horizon`, `coverage`, `n_draws`, and
  `max_attempts` like `irf` for the other identifications (`horizon = 0`
  now throws). Coverage levels are sorted. `irf(model, ::NarrativeRestriction;
  parallel = :distributed)` throws an `ArgumentError`.

### Fixed

- A single failed bootstrap replication made every band and standard error
  `NaN`: failed draws were filled with `NaN` and kept in the mean used for
  centering and in `std`. `proxy_svar_mbb` sorted them into its percentile
  intervals and counted them as non-rejections in its Anderson-Rubin sets.
- `compute_ma_matrices_from_B!` with `n_lags = 0` left `Φ[:, :, 2:end]`
  unset; these slices are now zero.
- `variance_decomposition` read the IRF array as horizon-first while it is
  stored variable-first, so it returned wrong shares without an error.
- `fit(OLSVAR, ...)` with a `FixedConstraint` set the fixed coefficient but did
  not re-estimate the other coefficients of that equation; with both fixed and
  zero constraints it threw a `DimensionMismatch`. Constrained estimation is now
  restricted least squares, equation by equation, for any mix of zero and fixed
  constraints.

## [0.1.2] - 2026-10-15

### Removed

- **CovarianceMatrices.jl dependency.** The package imported it without using it.

### Fixed

- **Cholesky impact matrix with an ordering.**
  `structural_impact(Σ_L, ::CholeskyID; ordering_idx)` returned a permuted
  copy of `Σ_L`, which is not a factor of Σ, so impulse responses were wrong
  whenever an ordering was given. It now returns the lower Cholesky factor of
  `Σ[ordering_idx, ordering_idx]` with rows and columns mapped back to the
  original variable order, matching
  `rotation_matrix(model, CholeskyID(ordering))`. An `ordering_idx` that is not
  a permutation of the variables throws an `ArgumentError`.

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

[Unreleased]: https://github.com/gragusa/MacroEconometricTools.jl/compare/v0.1.2...HEAD
[0.1.2]: https://github.com/gragusa/MacroEconometricTools.jl/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/gragusa/MacroEconometricTools.jl/releases/tag/v0.1.1
