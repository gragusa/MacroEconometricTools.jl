# Documentation Overview

This document summarizes the complete documentation for MacroEconometricTools.jl.

## Documentation Files Created

### 1. Technical Documentation (`docs/TECHNICAL.md`)

**Purpose**: Internal architecture and implementation details for developers and contributors.

**Contents**

- Design philosophy (type-based dispatch, parametric types, no Missing values)
- Type system architecture and hierarchy
- Constraint system implementation (zero, fixed, block exogeneity)
- VAR estimation algorithms (unconstrained and constrained)
- Identification schemes (Cholesky, sign restrictions, IV)
- Bootstrap inference methods (wild, standard, block)
- Performance considerations and optimization tips
- Guide to extending the package

**Target audience**: Package developers, contributors, advanced users who want to understand internals

**Key sections**:

- How constraint estimation works equation-by-equation
- Companion form construction for stability analysis
- Bootstrap algorithm details
- Type stability best practices


### 2. Mathematical Theory (`docs/src/mathematical/theory.md`)

**Purpose**: Mathematical foundation and statistical theory underlying the methods.

**Contents**:
- Reduced-form VAR model specification
- Structural VAR and identification problem
- Three identification schemes:
  - Recursive (Cholesky) identification
  - Sign restrictions
  - Instrumental variables
- Impulse response functions (definition, computation, properties)
- Forecast error variance decomposition
- Constrained estimation theory
- Inference methods (asymptotic and bootstrap)
- Model selection (AIC, BIC, HQIC)
- Stability and cointegration

**Target audience**: Researchers, econometricians, graduate students

**Key features**:

- Complete mathematical formulations with LaTeX
- Connects theory to code implementation
- Comprehensive references to literature
- Rigorous treatment of identification

### 3. Getting Started Tutorial (`docs/src/tutorials/getting_started.md`)

**Purpose**: Hands-on introduction to the package through a complete example.

**Example**: Oil market VAR (Kilian 2009 style)

- 3 variables: oil production growth, real activity, real oil price
- 456 monthly observations (simulated)

**Topics covered**

1. **Installation and setup**
2. **Loading and exploring data**
3. **VAR estimation** (basic and with diagnostics)
4. **Stability checking**
5. **Cholesky identification**
6. **Computing IRFs** with bootstrap confidence intervals
7. **Variance decomposition**
8. **Constrained estimation** (block exogeneity)
9. **Historical decomposition**
10. **Forecasting**

**Code examples**: Complete, runnable Julia code throughout

**Learning path**: Progressive complexity from basic to advanced

### 4. Sign Restrictions Tutorial (`docs/src/tutorials/sign_restrictions.md`)

**Purpose**: Advanced identification using sign restrictions.

**Contents**:

- Motivation for sign restrictions vs. Cholesky
- Sign restriction matrix specification
- Identification algorithm (Rubio-Ramírez et al. 2010)
- Computing IRFs with sign restrictions
- Set identification (multiple valid rotations)
- Bootstrap inference with sign restrictions
- Comparison with Cholesky identification
- Narrative sign restrictions
- Practical considerations:
  - Choosing restrictions
  - Computational efficiency
  - Acceptance rates
  - Diagnostics

**Example**: Oil market with three structural shocks
- Supply shock: production ↓, price ↑
- Demand shock: production ↑, activity ↑, price ↑
- Speculative demand: price ↑

**Advanced topics**:
- Combining with zero restrictions
- Magnitude restrictions
- Bayesian sign restrictions

### 5. Example Data (`docs/src/tutorials/data/oil_data.jl`)

**Purpose**: Simulated data generator for tutorials.

**Features**:

- Realistic oil market dynamics (calibrated DGP)
- 3 variables matching Kilian (2009) structure
- Flexible sample size
- Reproducible (seeded random number generation)

**Functions**:
- `generate_oil_market_data(; T=400, seed=123)`
- `load_oil_data()` - convenience function

**Output structure**:
- `data`: T × 3 matrix
- `dates`: Vector of date strings
- `names`: Variable names as symbols

### 6. Documentation README (`docs/README.md`)

**Purpose**: Navigation hub for all documentation.

**Features**:

- Quick links to all documentation
- Examples organized by topic and application
- Feature checklist (implemented, in development, planned)
- Minimal working example
- Citation information
- Contributing guidelines
- Comprehensive references

**Tables**:
- Topic → Tutorial + Theory links
- Application → Data + Methods links

## Documentation Structure

```
docs/
├── README.md                          # Navigation hub

├── TECHNICAL.md                       # Implementation details
├── DOCUMENTATION_OVERVIEW.md          # This file
└── src/
    ├── mathematical/
    │   └── theory.md                  # Mathematical foundation
    └── tutorials/
        ├── getting_started.md         # Basic tutorial
        ├── sign_restrictions.md       # Advanced tutorial
        └── data/
            └── oil_data.jl            # Example data generator
```

## Usage Guide

### For New Users

**Start here**:

1. Read `docs/README.md` for overview
2. Follow `docs/src/tutorials/getting_started.md` step-by-step
3. Refer to `docs/src/mathematical/theory.md` for theoretical background

**Learning path**:
```
Installation → Basic VAR → IRFs → Bootstrap → Constraints → Sign Restrictions
```

### For Researchers

**Workflow**:
1. Consult `docs/src/mathematical/theory.md` for methodology
2. Check tutorials for implementation examples
3. Adapt code to your data and research question

**Key sections**:
- Identification schemes (choose appropriate method)
- Inference methods (bootstrap vs. asymptotic)
- Model diagnostics (stability, information criteria)

### For Developers

**Getting started**:
1. Read `docs/TECHNICAL.md` for architecture
2. Study type hierarchy and dispatch system
3. Follow extension guide for adding features

**Key resources**:
- Type system architecture
- Constraint implementation (reference for complex logic)
- Performance best practices

## Examples by Research Topic

### Oil Market Dynamics
- **Tutorial**: `getting_started.md`, `sign_restrictions.md`
- **Data**: Simulated 3-variable system

- **Methods**: Cholesky, sign restrictions
- **Reference**: Kilian (2009)

### Monetary Policy (Planned)
- **Tutorial**: TBD
- **Data**: Macro data + high-frequency instrument
- **Methods**: External instruments
- **Reference**: Gertler & Karadi (2015)

### Fiscal Multipliers (Planned)
- **Tutorial**: TBD
- **Data**: Fiscal and macro variables
- **Methods**: Narrative sign restrictions
- **Reference**: Ramey (2011)

### Technology Shocks (Planned)
- **Tutorial**: TBD
- **Data**: Labor productivity, hours
- **Methods**: Long-run restrictions
- **Reference**: Galí (1999)

## Documentation Principles

### 1. Completeness
Every major feature is documented with:
- Theory

- Implementation
- Example
- Reference to literature

### 2. Accessibility
Multiple entry points for different audiences:
- Quick start for practitioners
- Mathematical rigor for researchers
- Implementation details for developers

### 3. Reproducibility
All examples include:
- Complete working code
- Data generation/loading instructions
- Expected output
- Sensitivity checks

### 4. Best Practices
Documentation encourages:
- Proper identification procedures
- Robust inference (bootstrap)
- Diagnostic checking
- Sensitivity analysis

## Future Documentation Plans

### Additional Tutorials

1. **Local Projections** (`local_projections.md`)

   - Alternative to VAR for IRF estimation
   - Robustness to misspecification
   - Comparison with VAR

2. **Bayesian VAR** (`bayesian_var.md`)
   - Minnesota prior
   - Posterior simulation
   - Predictive density

3. **Panel VAR** (`panel_var.md`)
   - Fixed effects
   - Mean group estimator
   - Application to country panels

4. **IV-SVAR** (`iv_svar.md`)
   - External instruments
   - Proxy SVAR
   - Weak instrument diagnostics

### Additional Examples

1. **Monetary Policy Shocks**
   - High-frequency identification
   - Narrative approach
   - Sign restrictions

2. **News Shocks**
   - Max share identification
   - Professional forecasts as instruments

3. **Financial Shocks**
   - Credit spreads
   - VIX as instrument
   - Sign restrictions

### API Documentation

Generate from docstrings using Documenter.jl:
- Function reference
- Type documentation
- Method tables
- Search functionality

## Documentation Quality Checklist

- ✅ Mathematical notation consistent throughout
- ✅ Code examples tested and working
- ✅ Cross-references between documents

- ✅ Literature citations complete
- ✅ Multiple difficulty levels (beginner to advanced)
- ✅ Theoretical and applied perspectives
- ✅ Implementation details for developers
- ⬜ PDF/HTML generation with Documenter.jl (planned)
- ⬜ Interactive plots in examples (requires Plots.jl)
- ⬜ Video tutorials (future consideration)

## Maintaining Documentation

### When Adding Features

1. **Update TECHNICAL.md**:

   - Add to type hierarchy
   - Document algorithm
   - Add extension example

2. **Update theory.md**:
   - Mathematical formulation
   - Properties and assumptions
   - Literature references

3. **Create tutorial** (if major feature):
   - Motivating example
   - Step-by-step guide
   - Comparison with alternatives

4. **Update README.md**:
   - Add to feature list
   - Link to new documentation
   - Update examples table

### Documentation Review Process

Before committing documentation changes:

1. **Technical accuracy**: Math and code correct
2. **Clarity**: Explanations understandable to target audience
3. **Completeness**: All features documented
4. **Consistency**: Notation and style uniform
5. **Examples**: Code runs without errors

## Contact and Contributions

For documentation improvements:
- Open issue on GitHub
- Submit pull request with changes

- Follow existing style and structure

For questions about documentation:
- Check README first
- Search tutorials for examples
- Open discussion on GitHub

## References

All documentation follows conventions from:

- Lütkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*

- Kilian, L., & Lütkepohl, H. (2017). *Structural Vector Autoregressive Analysis*
- Hamilton, J. D. (1994). *Time Series Analysis*

Julia documentation style guides:
- JuliaLang documentation best practices
- SciML documentation standards
