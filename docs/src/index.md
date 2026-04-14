# MacroEconometricTools.jl

A modern Julia package for macroeconometric analysis featuring Vector Autoregressions (VAR), structural identification, and impulse response analysis.

## Features

- **OLS VAR estimation** with flexible lag selection and coefficient constraints
- **IV-SVAR** (proxy-SVAR) identification with external instruments
- **Multiple identification schemes**: Cholesky, sign restrictions, IV, narrative restrictions
- **Comprehensive inference**: wild/block/standard bootstrap, Jentsch-Lunsford MBB for proxy-SVARs, MSW analytic confidence sets, Anderson-Rubin weak-instrument-robust sets
- **Forecast error variance decomposition** and **historical decomposition**
- **Forecasting** with uncertainty quantification
- **Distributed parallel computing** for bootstrap and sign restrictions

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/gragusa/MacroEconometricTools.jl")
```

## Quick Start

```julia
using MacroEconometricTools

# Estimate a VAR(4)
model = fit(OLSVAR, Y, 4; names=[:GDP, :Inflation, :Rate])

# Cholesky identification with bootstrap inference
result = irf(model, CholeskyID();
    horizon = 24,
    inference = WildBootstrap(1000),
    coverage = [0.68, 0.90, 0.95])

# IV-SVAR with external instrument
instrument = ExternalInstrument(proxy, 1)
iv_model = fit(IVSVAR, Y, 4; instrument=instrument)
iv_result = irf(iv_model, IVIdentification();
    horizon = 20,
    inference = ProxySVARMBB(2000; block_length=4))
```

## Tutorials

- **[Getting Started](tutorials/getting_started.md)**: VAR estimation, Cholesky identification, IRFs, FEVD
- **[Proxy SVAR (IV)](tutorials/svar_iv.md)**: External instrument identification, MBB, Anderson-Rubin, MSW inference
- **[Sign Restrictions](tutorials/sign_restrictions.md)**: Set identification with sign and narrative restrictions
- **[Parallel Computing](tutorials/parallel_computing.md)**: Distributed bootstrap and sign restriction search

## Mathematical Background

- **[Theory](mathematical/theory.md)**: Rigorous treatment of VAR estimation, identification, and inference
