# Repository Guidelines

## Project Structure & Module Organization
MacroEconometricTools.jl is a Julia package with its entry point in `src/MacroEconometricTools.jl`, which includes the domain modules. Core econometric routines live in `src/var` (VAR estimation, identification, inference), `src/ivsvar` (instrumental-variable specifications), and `src/localprojection`; shared utilities sit in `utilities.jl`, `types.jl`, `constraints.jl`, `bootstrap.jl`, and `parallel.jl`. Simulation helpers occupy `src/simulation.jl`. Tests reside in `test/`, where `runtests.jl` wires static analysis, regression datasets such as `kilian_kim_original_dataset.csv`, and focused suites like `test_basic.jl`. Documentation assets are under `docs/` with the Documenter build script in `docs/make.jl` and narrative pages inside `docs/src/`.

## Build, Test, and Development Commands
- `julia --project -e 'using Pkg; Pkg.instantiate()'`: resolve and download package and documentation dependencies.
- `julia --project -e 'using Pkg; Pkg.test()'`: runs unit suites and Aqua quality checks declared in `Project.toml`.
- `julia --project=docs docs/make.jl`: builds the docs; set Documenter credentials before deploying.
- `julia --project -e 'using Pkg; Pkg.precompile()'`: optional warm-up to avoid latency before exploratory work.

## Coding Style & Naming Conventions
Follow Julia style with 4-space indentation, no tabs, and keep lines under roughly 92 characters. Exported types use `CamelCase`; functions, macros, and local variables stay `lowercase_with_underscores`, while mutating methods carry a trailing `!`. Provide triple-quoted docstrings for exported APIs, place includes in dependency order inside `src/MacroEconometricTools.jl`, and document assumptions near numerical constants. When generating matrices or random draws, be explicit about element types and prefer deterministic seeds via StableRNGs.

## Testing Guidelines
Extend `test_basic.jl` or add new files included from `runtests.jl`, grouping related assertions inside `@testset` blocks. Reuse helpers such as `get_data()` and seed stochastic routines with `StableRNGs.Seed` to keep baseline expectations reproducible. Cover structural invariants—dimensions, stability diagnostics, identification checks—and guard regression values with tight tolerances. Run `julia --project -e 'using Pkg; Pkg.test()'` before pushing; add `Pkg.test(; coverage=true)` when gathering coverage for CI.

## Commit & Pull Request Guidelines
Write commit summaries in the imperative mood (e.g., “Add VAR stability checks”), keep them under ~72 characters, and add bodies when rationale or data sources need context. Each pull request should reference related issues, summarise behavioural impacts, and note any new datasets or doc pages touched. Attach reproducible command snippets rather than screenshots unless visual output is essential. Confirm both `Pkg.test()` and, when docs change, `julia --project=docs docs/make.jl` succeed before requesting review and mention the results in the PR description.

## Documentation & Examples
Update worked examples in `docs/src/` alongside code changes so the manual stays in sync with APIs. Add new exports to the Documenter navigation via the relevant `@docs` blocks, and regenerate Quarto notebooks such as `test/IRFs.qmd` if expected figures or tables change. Note any configuration prerequisites (credentials, environment variables) inside `docs/TECHNICAL.md` or inline comments to guide future contributors.
