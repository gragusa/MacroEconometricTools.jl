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
- `residual_covariance(model)` returns the residual covariance matrix `Σ`
  (formerly returned by `vcov`).
- `normalize_impact` and `normalize_impact!` (exported).
- `stderror`, `confint`, `nobs`, `dof`, and `dof_residual` are exported (the
  StatsAPI functions, so they do not clash with StatsBase or GLM).
- `==`, `isequal`, and `hash` compare `ZeroConstraint`, `FixedConstraint`,
  `BlockExogeneity`, `CholeskyID`, `SignRestriction`,
  `NarrativeShockRestriction`, and `NarrativeRestriction` by their fields, so
  two specifications built from equal arrays are equal.
- `select_lags(Y, max_lags; criterion = :aic | :bic | :hqic)` chooses the lag
  order of an OLS VAR. Every candidate order is fitted on the rows after the
  first `max_lags`, so all candidates have the same number of observations.
  It returns the selected order and a table of all three criteria.
- Calling `irfplot` or `irfplot!` before a Makie backend is loaded raises a
  `MethodError` whose message says to load one (e.g. `using CairoMakie`).

### Removed (breaking)

- The estimation specifications `BayesianVAR`, `LocalProjection`,
  `AbstractPrior`, `MinnesotaPrior`, and `NormalWishartPrior`. Nothing in this
  package estimated them, and their exported names clashed with the names
  exported by BayesianVAR.jl and LocalProjections.jl, which define their own.
  The shared result types `BayesianIRFResult` and `LocalProjectionIRFResult`
  remain.
- The `parallel` keyword of `bootstrap_irf`. It ran serially with a warning
  for every value; the bootstrap has no parallel option. `parallel` remains
  for the sign-restriction search (`rotation_matrix`, `irf` with
  `SignRestriction`).
- `rescale!`. It changed its argument's arrays in place but returned a new
  result carrying the updated scale metadata, so the argument was left
  inconsistent. Use `rescale`.
- The Makie recipes `MCMCTrace`, `MCMCDensity`, and `MCMCHistogram`. They
  were unrelated to impulse responses and unused by the packages built on this
  one.

### Deprecated

- `fit(IVSVAR, Y, p; instrument)`, the `IVSVAR` specification, and
  `IVIdentification()` without an instrument. Fit the reduced form with
  `fit(OLSVAR, Y, p)` and pass the instrument to the identification:
  `irf(model, IVIdentification(Z, target_shock))`,
  `first_stage_F(model, id)`, `iv_summary(model, id)`.
- `ProxyIV` is a deprecated name for `ExternalInstrument`.
- `log_likelihood(model)`; use `loglikelihood(model)`.
- `impulse_response`; use `irf`.
- `effective_obs(model)`; use `nobs(model)`.
- `normalize(P, norm)` and `normalize!(P, norm)`; use `normalize_impact` and
  `normalize_impact!`. The deprecated names are no longer exported, so they no
  longer clash with `LinearAlgebra.normalize`.
- `bootstrap_irf(model, identification, horizon, reps; method, block_length,
  max_failure_rate)`, `bootstrap_irf_wild`, `bootstrap_irf_standard`, and
  `bootstrap_irf_block`. Use `bootstrap_irf(model, identification, inference;
  horizon, normalization, rng)` with `inference` a `WildBootstrap`,
  `Bootstrap`, or `BlockBootstrap`, which carries the number of replications,
  the block length, and `max_failure_rate`. The same inference objects drive
  `irf(model, identification; inference)`, and both return the same draws for
  the same `rng`. `bootstrap_irf` throws an `ArgumentError` for other
  inference types and for `IVIdentification`.
- The `scale` keyword of `irf`. Use `rescale(irf(...), factor)`, which gives
  the same responses, draws, and bands.
- `confidence_bands(irf, WildBootstrap; coverage)` (an inference type that was
  not used); use `confidence_bands(irf; coverage)`.
- `confidence_bands(irf, model, identification, inference; coverage, rng)`;
  call `irf(model, identification; inference, coverage, rng)` again. The
  deprecated method forwards to `irf` with the result's horizon and
  normalization, and throws for rescaled or cumulated results. It previously
  always failed with a `MethodError`.
- The `zerolinecol` keyword of the Plots recipe; use `zerolinecolor`, the
  spelling `irfplot` uses.

### Changed (breaking)

- The Makie extension requires Makie 0.24 (previously 0.20).

- Export surface. Exported names are the end-user API. Building blocks for
  packages built on this one are `public` but not exported; call them as
  `MacroEconometricTools.name` or bring them in with
  `using MacroEconometricTools: name`. Newly public, no longer exported:
  `structural_impact`, `compute_ma_matrices_from_B!`,
  `generate_random_orthogonal!`, `check_sign_restrictions`,
  `compute_structural_shocks`, `check_narrative_restrictions`, `iv_impact`,
  `iv_instrument`, `refit_for_bootstrap`, `proxy_svar_dynamics`,
  `bootstrap_irf`, `lag` (avoids a clash with ShiftedArrays' `lag`),
  `create_lags!`, and `companion_form`. Newly public (previously internal):
  `compute_ma_matrices`, `compute_irf_point`, `n_shocks`,
  `estimate_proxy_svar`, `apply_constraints!`, `build_selection_matrix`,
  `check_constraints`, `coefficient_covariance`, `sigma_covariance`,
  `irf_jacobian_matrices`, `irf_effect_covariance`,
  `irf_asymptotic_stderror`, `duplication_matrix`, `elimination_matrix`,
  `commutation_matrix`, `simulate_var`, and `simulate_structural`.
- Deprecated names are not exported: `IVSVAR`, `ProxyIV`, `log_likelihood`,
  `bootstrap_irf_wild`, `bootstrap_irf_standard`, `bootstrap_irf_block`,
  `impulse_response`, and `effective_obs`. They still work, with a
  deprecation warning, when qualified with the module name.
- `cumulative_irf(::IRFResult)` throws an `ArgumentError` when the result has
  no stored draws, as `irf(...; cumulate)` does, instead of cumulating the
  bands themselves with a warning. Store the draws with `save_draws = true`.
  Both paths throw for results with Anderson-Rubin bands
  (`ProxySVARMBB(compute_ar = true)`), and `cumulative_irf` throws when a
  requested variable is already cumulated. The `cumulative` metadata key is
  gone; `cumulated_vars` lists every cumulated variable.
- `rescale` multiplies the bands directly instead of recomputing them from the
  draws, and no longer warns when the result has no draws. Every band type in
  the package scales exactly with the shock, so the bands are unchanged for
  percentile and delta-method results; Anderson-Rubin bands of `ProxySVARMBB`
  results were previously replaced by percentile bands and are now kept.
  `rescale(irf, factor)` scales the responses to every shock; non-finite
  factors throw.

- `VARModel` follows the StatsAPI contract:
  - `coef(model)` is the `(1 + n_vars * n_lags) × n_vars` coefficient matrix
    `B` of `Y = X B + U` (intercept in row 1, column = equation) instead of a
    NamedTuple. The lag array stays available as `model.coefficients.lags`.
  - `vcov(model)` is the covariance of `vec(coef(model))`,
    `kron(Σ, inv(X'X))`, instead of the residual covariance.
  - `stderror(model)` is shaped like `coef(model)`; `confint(model; level)` is
    a `length(coef(model)) × 2` matrix in `vec(coef(model))` order instead of
    a NamedTuple of matrices.
  - `nobs(model)` is the effective sample `size(Y, 1) - n_lags` instead of
    the total; `dof_residual(model)` is `nobs(model) - (1 + n_vars * n_lags)`,
    the divisor of `Σ`; `modelmatrix(model)` has `nobs(model)` rows.
  - `loglikelihood`, `aic`, and `bic` are methods of the StatsAPI functions
    (formulas unchanged), so they no longer clash with StatsBase's.
- `Base.size(::VARModel)` is removed; use `nobs`, `n_lags`, and `n_vars`.
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
- `CholeskyID`, `SignRestriction`, and `NarrativeRestriction` check their
  arguments on every construction path, including the positional forms.
  Sign matrices must be square, with entries in {-1, 0, +1} (or `+`/`-`), and
  `horizon` must be nonnegative. A narrative shock index must not exceed the
  number of shocks. A `CholeskyID` ordering must be a nonempty permutation
  of `1:n` or a nonempty list of distinct names. The stored sign matrix is a
  copy of the input. `NarrativeRestriction` accepts a vector of narrative
  restrictions with an abstract element type, including an empty
  `NarrativeShockRestriction[]`.
- Bad input raises `ArgumentError` instead of `ErrorException`. This covers
  unknown shock names in `rescale`, narrative dates missing from the model's
  dates (or used on a model fitted without dates), cumulation without
  bootstrap draws, and `confidence_bands` on a result without saved draws.
  A proxy whose length does not match the residuals raises
  `DimensionMismatch`.
- A `block_length` larger than the number of residual periods throws an
  `ArgumentError` in `BlockBootstrap` and `ProxySVARMBB`.
- `irf(model, ::SignRestriction; inference = ...)` (and the same call with
  `NarrativeRestriction`) throws an `ArgumentError` explaining that
  set-identified results take their bands from the accepted rotations.
- `proxy_svar_mbb` and `msw_confidence_set` are `public` and no longer
  exported (call them as `MacroEconometricTools.proxy_svar_mbb` or import
  them by name). Both take a `coverage` keyword (default
  `[0.68, 0.90, 0.95]`), and their results are vectors aligned with the
  sorted `coverage` instead of fixed 68%/90%/95% fields:
  - `proxy_svar_mbb`: `ci68_irf`/`ci95_irf`/… become `irf_bands`,
    `irf_norm_bands`, `fevd_bands`, and `halls_irf_norm_bands`, each a
    `(lower, upper)` pair of `(variable, horizon)` matrices; `ar.index68`/
    `index90`/`index95` become `ar.included`, and `ar.rates` and each entry of
    `ar.included` are `(grid, variable, horizon)`; `irf_store` and
    `irf_norm_store` become `irf_draws` and `irf_norm_draws` with layout
    `(draw, variable, horizon)`; the result also carries `coverage`.
  - `msw_confidence_set`: `cs68_irf_norm`/`cs95_irf_norm` and
    `bounded68`/`bounded95` become `cs_irf_norm` (arrays of size
    `(2 or 4, variable, horizon)`) and `bounded`. Critical values are
    `quantile(Chisq(1), α)` rather than the rounded constants 0.9889 and
    3.8415.
- `irf(...; inference = ProxySVARMBB(compute_ar = true))` computes the
  Anderson-Rubin sets at each requested coverage level. Before, a level was
  mapped to the nearest of the 68%, 90%, and 95% sets.
- The point-estimate impact column in `proxy_svar_mbb` is computed by the
  same Cholesky-based step as the bootstrap draws.

### Fixed

- `forecast(model, h)` returned `h` rows of which the first `n_lags` were
  initial data, so only `h - n_lags` periods were forecast. It now returns the
  `h` forecast periods. With `include_draws = true`, the `forecast` field is
  the conditional mean instead of the mean of the simulated paths, and the
  draws use the new `rng` keyword instead of the global RNG.
- `confidence_bands(irf; coverage)` recomputes percentile bands from saved
  draws, also for `ProxySVARMBB` results, and throws an `ArgumentError` for
  Anderson-Rubin bands, which are not percentile bands of the draws.
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
- `fitted(model)` and `confint(model)` rebuilt the coefficient matrix with a
  wrong reshape of the lag array, so for more than one variable they returned
  wrong values.

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
