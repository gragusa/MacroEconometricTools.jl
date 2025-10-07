# MacroEconometricTools.jl Documentation

Comprehensive documentation for macroeconomic time series analysis in Julia.

## Documentation Structure

### For Users

#### Getting Started
- **[Getting Started Tutorial](src/tutorials/getting_started.md)**: Introduction to VAR estimation, identification, and IRF computation using an oil market example
- **[Sign Restrictions Tutorial](src/tutorials/sign_restrictions.md)**: Advanced identification using sign restrictions

#### Mathematical Foundation
- **[Theory and Methods](src/mathematical/theory.md)**: Mathematical foundation of VAR models, identification schemes, and inference methods

### For Developers

- **[Technical Documentation](TECHNICAL.md)**: Implementation details, architecture, and extending the package

## Quick Links

### Examples by Topic

| Topic | Tutorial | Theory |
|-------|----------|--------|
| Basic VAR estimation | [Getting Started](src/tutorials/getting_started.md#estimating-a-var-model) | [VAR Model](src/mathematical/theory.md#the-reduced-form-var) |
| Cholesky identification | [Getting Started](src/tutorials/getting_started.md#cholesky-recursive-identification) | [Recursive ID](src/mathematical/theory.md#1-recursive-cholesky-identification) |
| Sign restrictions | [Sign Restrictions](src/tutorials/sign_restrictions.md) | [Sign ID](src/mathematical/theory.md#2-sign-restrictions) |
| Impulse responses | [Getting Started](src/tutorials/getting_started.md#impulse-response-functions) | [IRF Theory](src/mathematical/theory.md#impulse-response-functions) |
| Constraints | [Getting Started](src/tutorials/getting_started.md#constrained-var-estimation) | [Constrained Estimation](src/mathematical/theory.md#constrained-var-estimation) |
| Bootstrap inference | [Getting Started](src/tutorials/getting_started.md#impulse-response-functions) | [Bootstrap](src/mathematical/theory.md#bootstrap-inference) |
| **Parallel computing** | **[Parallel Computing](src/tutorials/parallel_computing.md)** | [Implementation](TECHNICAL.md#parallel-computing) |

### Examples by Application

| Application | Data | Methods |
|------------|------|---------|
| Oil market shocks (Kilian 2009) | [Oil data](src/tutorials/data/oil_data.jl) | Cholesky, Sign restrictions |
| Monetary policy | TBD | External instruments |
| Fiscal multipliers | TBD | Narrative sign restrictions |
| Technology shocks | TBD | Long-run restrictions |

## Features

### Implemented
- ✅ OLS VAR estimation
- ✅ Coefficient constraints (zero, fixed, block exogeneity)
- ✅ Cholesky identification
- ✅ Sign restrictions (partial implementation)
- ✅ Wild bootstrap inference
- ✅ Forecast error variance decomposition
- ✅ Historical decomposition
- ✅ Forecasting

### In Development
- 🚧 IV-SVAR (instrumental variable identification)
- 🚧 Local projections
- 🚧 Bayesian VAR
- 🚧 Panel VAR

### Planned
- 📋 VECM (Vector Error Correction Models)
- 📋 Time-varying parameter VAR
- 📋 Proxy SVAR
- 📋 Narrative restrictions

## Installation

```julia
using Pkg
Pkg.add("MacroEconometricTools")
```

## Minimal Example

```julia
using MacroEconometricTools

# Simulate data
Y = randn(200, 3)  # 200 observations, 3 variables

# Estimate VAR(4)
var = estimate(OLSVAR, Y, 4)

# Cholesky identification
P = identify(var, CholeskyID())

# Impulse responses with bootstrap
irf_result = irf(var, CholeskyID();
                 horizon=24,
                 bootstrap_reps=1000)
```

## Citation

If you use this package in your research, please cite:

```bibtex
@software{MacroEconometricTools,
  author = {Your Name},
  title = {MacroEconometricTools.jl: Structural Vector Autoregression in Julia},
  year = {2024},
  url = {https://github.com/yourusername/MacroEconometricTools.jl}
}
```

## Contributing

Contributions are welcome! Please see:
- [Technical Documentation](TECHNICAL.md#extending-the-package) for adding features
- [GitHub Issues](https://github.com/yourusername/MacroEconometricTools.jl/issues) for bug reports

## License

This package is licensed under the MIT License.

## References

### Textbooks
- Lütkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*
- Kilian, L., & Lütkepohl, H. (2017). *Structural Vector Autoregressive Analysis*
- Hamilton, J. D. (1994). *Time Series Analysis*

### Key Papers

**Identification:**
- Sims, C. A. (1980). Macroeconomics and Reality
- Rubio-Ramírez et al. (2010). Structural Vector Autoregressions
- Uhlig, H. (2005). What Are the Effects of Monetary Policy

**Applications:**
- Kilian, L. (2009). Not All Oil Price Shocks Are Alike
- Gertler, M., & Karadi, P. (2015). Monetary Policy Surprises
- Ramey, V. A. (2011). Identifying Government Spending Shocks

**Inference:**
- Kilian, L. (1998). Small-Sample Confidence Intervals
- Gonçalves, S., & Kilian, L. (2004). Bootstrapping Autoregressions

---

For questions or issues, please open an issue on [GitHub](https://github.com/yourusername/MacroEconometricTools.jl/issues).
