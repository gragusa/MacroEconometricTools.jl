# Recent Changes Since Last Commit

## Benchmarking & Contributor Guidance
- Added a dedicated benchmarking environment with reusable suites and an AirspeedVelocity runner (`benchmark/benchmarks.jl`, `benchmark/run_asv.jl`, `benchmark/Project.toml:1`).  
- Documented contributor expectations and benchmark usage in `AGENTS.md` and the README benchmark section (`README.md:145`).

## Deterministic Randomness & Bootstrap Fixes
- Threaded serial/distributed bootstrap paths now accept caller-provided RNGs; residual resampling respects those seeds (`src/bootstrap.jl:38`, `src/bootstrap.jl:170`).  
- Sign-restriction search (serial + distributed) inherits RNG semantics and respects constraint counts (`src/var/identification.jl:114`).

## API Compatibility & Metadata
- Reintroduced the `VAR` convenience wrapper to mirror the legacy IRFs interface (`src/var/estimation.jl:185`).  
- IRF results carry variable names in metadata for downstream tooling (`src/var/irfs.jl:79`).

## Plotting Recipes
- Refreshed the Plots.jl recipe to work with `IRFResult` and metadata-driven labels (`src/plots.jl:1`).  
- Added an optional Makie extension with an `irfplot` recipe for grid layouts and confidence bands (`ext/MacroEconometricToolsMakieExt.jl:1`).

## Tests & Docs
- Trimmed `test/runtests.jl` to load only the maintained smoke suite; added deterministic bootstrap coverage in `test/test_basic.jl:67`.  
- Updated bootstrap documentation for RNG-aware helpers (`docs/TECHNICAL.md:459`).

## Dependency Updates
- Promoted `StableRNGs` and `Distributed` to core dependencies; registered `Makie` as an optional extension (`Project.toml:10`).  
- Optional plotting recipes are now loaded via conditional includes (`src/MacroEconometricTools.jl:48`).

