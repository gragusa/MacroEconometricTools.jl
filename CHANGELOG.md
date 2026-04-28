# Changelog

## Unreleased

### Added

- **Shock normalization API**: Two-layer control over IRF units — pick a _scheme_ at estimation time, then (optionally) apply per-shock _rescaling_ afterwards.

  **1. Normalization scheme** — `irf(model, id; normalization=...)` takes an `AbstractNormalization`:
  - `UnitStd()` (default): shocks have unit variance. Under Cholesky, `P*P' = Σ`; on impact, shock `j` moves variable `j` by one residual standard deviation.
  - `UnitEffect()`: shocks have unit impact on their own variable — `P[j, j] = 1`. Natural for "1 percentage point of the monetary policy shock" under a recursive ordering.

  The scheme is fixed at estimation time and stored in `metadata.normalization`. `UnitStd` / `UnitEffect` / `AbstractNormalization` are exported, as are the low-level `normalize` / `normalize!` utilities that apply a scheme to an impact matrix `P`.

  **2. Per-shock rescaling** — `rescale(irf, :MP => -1, :Supply => 100)` multiplies each named shock by a scalar; `rescale!` mutates in place. Typical uses: `:MP => -1` flips a sign under `UnitStd` for interpretability; `:MP => 0.25` scales a `UnitEffect` shock to a 25 bp move. Unlisted shocks are unchanged. When bootstrap draws are available (`save_draws=true`), confidence bands are recomputed from rescaled draws (correct); otherwise the bands are rescaled approximately with a warning, and `min`/`max` are applied to handle sign flips. Works on `IRFResult` and `SignRestrictedIRFResult`; for the latter, draws are rescaled and median / quantile bands are recomputed.

  **3. Introspection** — `get_scale(irf)` returns an exported `IRFScale` struct with:
  - `normalization` — the scheme (`UnitStd()` or `UnitEffect()`),
  - `scale::Vector` — cumulative per-shock scale factors applied via `rescale`,
  - `impact_diagonal::Vector` — diagonal of the (normalized, unscaled) impact matrix `P`; i.e. what "1 unit of shock `j`" means for variable `j`. `≈ 1` everywhere under `UnitEffect`; under `UnitStd`, these are the own-variable impact SDs. For sign-restricted IRFs, this is the median across accepted draws.
  - `names::Vector{Symbol}` — variable/shock labels.

  `IRFScale` has a custom `show` method that prints `impact_diag × scale = effective on-impact` per shock.

- **Per-shock `metadata.scale`**: The `scale` field in IRF metadata is now a `Vector` (one entry per shock) instead of a scalar, tracking per-shock cumulative scale factors composed by successive `rescale` calls.

- **`impact_diagonal` in metadata**: The `irf()` function now stores the diagonal of the normalized impact matrix in `metadata.impact_diagonal`, capturing what "one unit of shock j" means for each variable before any `rescale` is applied.

- **Proxy-SVAR `target_shock` works for any variable position**: `IVIdentification(Z, target)`, `ExternalInstrument(Z; target_shock=...)`, and `ProxyIV(proxies; target_shock=...)` now produce correct IRFs, MSW confidence sets, and Jentsch-Lunsford MBB inference regardless of which variable is instrumented. Previously the instrumented variable had to be ordered first; bootstrap dynamics, AR anchoring, band placement, and MSW quadratic coefficients all hardcoded position 1. The estimation layer was already parametric — the fix propagated `target` through `proxy_svar_dynamics`, `_proxy_svar_mbb_impl`, `_ar_confidence_sets`, the `compute_inference_bands` dispatch, and `msw_confidence_set`.

- **`ProxyIV` constructor parity with `ExternalInstrument`**: `ProxyIV` now accepts `target_shock` as an `Int` or `Symbol` via keyword (`ProxyIV(proxies; target_shock=:FFR)`) or positional (`ProxyIV(proxies, 2)`) arguments, matching `ExternalInstrument`'s surface. The struct is parameterized as `ProxyIV{T, S}` with a scalar `target_shock::S` field (`Vector{Int}` storage dropped; single-shock identification was already enforced downstream).

- **`estimate_proxy_svar!` and `proxy_svar_dynamics!`**: In-place variants of the Jentsch-Lunsford inner-loop kernels, plus `ProxySvarBuffers` and `ProxyDynamicsBuffers` workspace structs. Used inside `proxy_svar_mbb` to eliminate per-draw allocations.

- **`simulate_var!`**: In-place variant of `simulate_var` that writes into a caller-supplied output matrix instead of allocating. Loop structure, scalar arithmetic, and `@inbounds` annotations are byte-identical to the allocating version — `simulate_var` now delegates to `simulate_var!`. Used inside the three Cholesky-SVAR bootstrap functions to reuse a single simulation buffer across all replications.

- **`_BootstrapWorkspace`**, **`_fast_refit_ols!`**, **`_fast_cholesky_irf!`** (internal): Pre-allocated workspace and in-place kernels that replace the `fit(OLSVAR)` + `rotation_matrix` + `compute_irf_point` chain for the Cholesky-identification fast path. `_fast_refit_ols!` solves unconstrained VAR(p) OLS via `mul!(G, X', X)` + `cholesky!(Symmetric(G))` + `ldiv!` directly into workspace buffers, with no `VARModel` construction, no lagged-X copy, no residual/covariance allocation, and no companion-form allocation. `_fast_cholesky_irf!` fuses the companion-power iteration (`F_power ← F_power · F`) with the IRF computation (`IRF_h = F^h[1:K, 1:K] · P`), avoiding the intermediate `(K, K, horizon+1)` MA-coefficient array. Both kernels benchmark at 0 bytes per call. Supports `UnitStd` and `UnitEffect` normalization.

- **`irfplot!` mutating function**: New exported function that draws a single (variable, shock) IRF panel onto an existing `Makie.Axis`. Supports all four IRF result types (`IRFResult`, `SignRestrictedIRFResult`, `BayesianIRFResult`, `LocalProjectionIRFResult`). Enables composing custom multi-panel figures with per-panel styling.

- **`vars`/`shocks` selection for `LocalProjectionIRFResult` (Makie)**: The `irfplot` dispatch for `LocalProjectionIRFResult` now accepts `vars`, `shocks`, `pretty_vars`, and `pretty_shocks` keyword arguments, matching the other three IRF result type dispatches.

- **`vars`/`shocks` selection for `LocalProjectionIRFResult` (Plots.jl)**: The RecipesBase recipe for `LocalProjectionIRFResult` now accepts `vars`, `shocks`, `pretty_vars`, and `pretty_shocks` keyword arguments for subsetting and relabelling.

### Changed

### Fixed

- **Proxy-SVAR: hardcoded position-1 assumptions**: The Jentsch-Lunsford MBB bootstrap dynamics, AR confidence set anchor, band placement in `compute_inference_bands`, and the MSW confidence set quadratic (Wald statistic, `a`/`b` coefficients, impact-normalization row) all hardcoded `[1]` / `[1, 1]`. Any `target_shock ≠ 1` silently produced wrong numbers. Now parameterized on `target`.

### Performance

- **Proxy-SVAR MBB bootstrap: −96% memory, −98% allocations, −17% wall time**. The inner loop now reuses `ProxySvarBuffers` / `ProxyDynamicsBuffers` across draws, and the in-place `estimate_proxy_svar!` solves OLS via `cholesky!(X'X)` + `ldiv!` (instead of `xx \ yy`), computes residuals via `mul!(U, X, A, -1, 1)`, and uses `mul!` everywhere for the covariance and identification formulas. Benchmark on a 3-variable VAR, `horizon=20`, `reps=2000`: **226 MiB → 9.9 MiB** total allocation, **145 k → 3 k** total allocs, **72 → 1.5** allocs per draw, **14% → 0.2%** GC fraction, **90.6 ms → 75.4 ms** wall time. Verified bit-exact against the 170-assertion Python cross-validation suite.

- **Cholesky-SVAR bootstraps (`WildBootstrap`, `Bootstrap`, `BlockBootstrap`): ~1.6× faster, ~11× less memory**. The inner loop of `bootstrap_irf_wild`, `bootstrap_irf_standard`, and `bootstrap_irf_block` now branches on `identification isa CholeskyID && ordering === nothing`; the common Cholesky path replaces the per-rep chain of `simulate_var` (allocating) → `refit_for_bootstrap` → `fit(OLSVAR)` → `rotation_matrix` → `compute_irf_point` with an in-place pipeline that reuses a single `_BootstrapWorkspace{T}` across every replication: `simulate_var!` writes the bootstrap panel into a preallocated buffer, `_fast_refit_ols!` estimates OLS coefficients and the innovation Cholesky factor directly into workspace fields via the normal-equations Cholesky (no `VARModel` struct, no lagged-X copy, no residual matrix, no companion-form allocation), and `_fast_cholesky_irf!` computes the IRF via fused companion-power iteration (no intermediate MA-coefficient array). Non-Cholesky identifications (sign restrictions, ordered Cholesky, etc.) continue to use the original `fit(OLSVAR)` path with no behavioural change. Benchmark on a 4-variable VAR(2), `T=150`, `horizon=20`, `reps=300`: **11.86 ms → 7.33 ms** wall time (1.62×), **49.0 MB → 4.3 MB** total allocation (11.4× drop) for `WildBootstrap`; similar for `Bootstrap` and `BlockBootstrap`. Per-rep residual allocation drops from ≈160 KB to ≈240 B. The three internal helpers (`simulate_var!`, `_fast_refit_ols!`, `_fast_cholesky_irf!`) individually benchmark at 0 bytes per call. Verified: zero bit-level difference between repeated calls with the same `StableRNG` seed; full test suite (1055 assertions across 39 test sets) passes, including the strict Python cross-validation references.

- **`bootstrap_irf_wild`, `bootstrap_irf_standard`, `bootstrap_irf_block`: simulation buffer reuse**. Even on the non-Cholesky generic path, the simulated bootstrap panel `Y_boot` is now allocated once outside the rep loop and written by `simulate_var!` in place, saving one T×K allocation per rep.

- **Proxy-SVAR MBB micro-cleanup**: Replaced three per-rep slice-broadcast assignments (`irf_store[:, :, b] .= dyn_buf.irf` and two analogues for `irf_norm_store`, `fevd_store`) with `copyto!(view(...), ...)`, shaving a few hundred bytes per MBB replication. Not a measurable wall-time change on the current benchmark but eliminates a fixable source of per-rep allocation.

### Removed

- **`irf_scale` and `flipshock` from all plotting functions**: The `irfplot`, `irfplot!`, and Plots.jl recipe functions no longer accept `irf_scale` or `flipshock` keyword arguments. All scaling must be done on the IRF result before plotting using `rescale` / `rescale!`.
