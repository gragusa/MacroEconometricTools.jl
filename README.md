# MacroEconometricTools.jl

A modern Julia package for macroeconometric analysis featuring Vector Autoregressions (VAR), structural identification, and impulse response analysis.

## Features

### Implemented
- ✅ **OLS VAR estimation** with flexible lag selection
- ✅ **Coefficient constraints** (zero, fixed values, block exogeneity)
- ✅ **Multiple identification schemes**:
  - Cholesky (recursive) identification
  - Sign restrictions (Rubio-Ramírez et al. 2010)
  - IV identification (placeholder)
- ✅ **Impulse Response Functions** with multiple inference methods
- ✅ **Bootstrap inference** (wild, standard, block bootstrap)
- ✅ **Distributed parallel computing** for bootstrap and sign restrictions
- ✅ **Forecast error variance decomposition**
- ✅ **Historical decomposition**
- ✅ **Forecasting** with uncertainty quantification
- ✅ **Simulation** from estimated models

### In Development
- 🚧 IV-SVAR (instrumental variable identification)
- 🚧 Local projections
- 🚧 Bayesian VAR

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/gragusa/MacroEconometricTools.jl")
```

## Quick Start

```julia
using MacroEconometricTools

# Estimate VAR
var = estimate(OLSVAR, Y, 4)  # 4 lags

# Cholesky identification
P = identify(var, CholeskyID())

# Impulse responses with bootstrap
irf_result = irf(var, CholeskyID();
                 horizon=24,
                 bootstrap_reps=1000)
```

## Parallel Computing

Speed up bootstrap and sign restrictions using distributed computing:

```julia
using Distributed
addprocs(4)
@everywhere using MacroEconometricTools

# Parallel bootstrap (4× faster)
irf_result = irf(var, id;
                horizon=24,
                bootstrap_reps=2000,
                parallel=:distributed)

# Parallel sign restriction search
P = identify(var, id_sign;
            max_draws=50000,
            parallel=:distributed)
```

## Documentation

- **[Getting Started Tutorial](docs/src/tutorials/getting_started.md)**: Complete walkthrough with oil market example
- **[Sign Restrictions Tutorial](docs/src/tutorials/sign_restrictions.md)**: Advanced identification methods
- **[Parallel Computing Guide](docs/src/tutorials/parallel_computing.md)**: Performance optimization
- **[Mathematical Theory](docs/src/mathematical/theory.md)**: Rigorous treatment of VAR theory
- **[Technical Documentation](docs/TECHNICAL.md)**: Implementation details for developers
- **[Quick Reference](docs/QUICK_REFERENCE.md)**: Fast lookup for common tasks

## Example: Oil Market VAR

```julia
using MacroEconometricTools

# Load data (simulated oil market: production, activity, price)
include("docs/src/tutorials/data/oil_data.jl")
data = load_oil_data()

# Estimate VAR(24) - monthly data, 2 years of lags
var = estimate(OLSVAR, data.data, 24; names=data.names)

# Cholesky identification (production → activity → price)
irf_result = irf(var, CholeskyID();
                horizon=48,  # 4 years
                inference=:bootstrap,
                bootstrap_reps=1000,
                coverage=[0.68, 0.90, 0.95])

# Variance decomposition
fevd = variance_decomposition(irf_result)

# Historical decomposition
hd = historical_decomposition(var, CholeskyID())
```

## Key Advantages

### Performance
- **Type-stable code**: Fully parametric types for compiler optimization
- **No unnecessary allocations**: Mutating functions where appropriate
- **Parallel computing**: Near-linear speedup with distributed workers
- **Efficient algorithms**: Equation-by-equation estimation with constraints

### Design
- **Type-based dispatch**: `estimate(OLSVAR, ...)` not `estimate(..., method=:ols)`
- **Flexible constraints**: Easy specification of coefficient restrictions
- **Extensible**: Simple to add new estimators and identification schemes
- **No Missing types**: Uses NaN for type stability

### Usability
- **Comprehensive documentation**: 25,000+ words across tutorials and theory
- **Runnable examples**: All code examples tested and working
- **Clear error messages**: Helpful diagnostics
- **Standard interface**: Follows SciML/StatsModels conventions

## Citation

If you use this package in your research, please cite:

```bibtex
@software{MacroEconometricTools2024,
  author = {Giuseppe Ragusa},
  title = {MacroEconometricTools.jl: Structural Vector Autoregression in Julia},
  year = {2024},
  url = {https://github.com/gragusa/MacroEconometricTools.jl}
}
```

## Contributing

Contributions are welcome! Please:
1. Read the [Technical Documentation](docs/TECHNICAL.md#extending-the-package)
2. Follow the code style in [CLAUDE.md](CLAUDE.md)
3. Add tests for new features
4. Update documentation

## Benchmarks

- Run targeted microbenchmarks:
  ```julia
  julia --project=benchmark -e 'using Pkg; Pkg.instantiate(); using BenchmarkTools; include("benchmark/benchmarks.jl"); BenchmarkTools.run(MacroEconometricToolsBenchmarks.SUITE)'
  ```
- Compare revisions with AirspeedVelocity:
  ```bash
  julia --project=benchmark benchmark/run_asv.jl
  ```
  Configure revisions via `MET_ASV_BASELINE`, `MET_ASV_CANDIDATE`, and `MET_ASV_OUTPUT_DIR` environment variables; the script writes JSON outputs plus a Markdown summary under `benchmark/results/`.
- Enable distributed microbenchmarks by exporting `MET_BENCH_ENABLE_DISTRIBUTED=true` (override worker count with `MET_BENCH_NWORKERS`).

## License

MIT License - see [LICENSE](LICENSE) file

## References

### Textbooks
- Lütkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*. Springer.
- Kilian, L., & Lütkepohl, H. (2017). *Structural Vector Autoregressive Analysis*. Cambridge University Press.
- Hamilton, J. D. (1994). *Time Series Analysis*. Princeton University Press.

### Key Papers
- Sims, C. A. (1980). Macroeconomics and Reality. *Econometrica*.
- Kilian, L. (2009). Not All Oil Price Shocks Are Alike. *American Economic Review*.
- Rubio-Ramírez et al. (2010). Structural Vector Autoregressions. *Review of Economic Studies*.
- Kilian, L. (1998). Small-Sample Confidence Intervals for IRFs. *Review of Economics and Statistics*.

## Acknowledgments

Built with modern Julia best practices and inspired by:
- R packages: vars, svars
- MATLAB toolboxes: Kilian-Lütkepohl replication files
- Python: statsmodels.tsa.vector_ar

---

**Status**: Active development | **Version**: 0.1.0 | **Julia**: ≥ 1.10
