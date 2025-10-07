# Mathematical Foundation

## Vector Autoregression (VAR) Models

### The Reduced-Form VAR

A VAR(p) model relates a vector of $n$ variables to its own lags:

$$
Y_t = c + A_1 Y_{t-1} + A_2 Y_{t-2} + \cdots + A_p Y_{t-p} + u_t
$$

where:

- $Y_t$ is an $(n \times 1)$ vector of endogenous variables at time $t$
- $c$ is an $(n \times 1)$ vector of intercepts
- $A_i$ are $(n \times n)$ coefficient matrices for lag $i = 1, \ldots, p$
- $u_t$ is an $(n \times 1)$ vector of reduced-form errors with $E[u_t] = 0$ and $E[u_t u_t'] = \Sigma_u$

**Notation in code**:

- `Y` corresponds to $Y_t$
- `coefficients.intercept` corresponds to $c$
- `coefficients.lags[:,:,i]` corresponds to $A_i$
- `residuals` corresponds to $\hat{u}_t$
- `Σ` corresponds to $\Sigma_u$

### Companion Form Representation

The VAR(p) can be written as a VAR(1) system:

$$
Z_t = F Z_{t-1} + v_t
$$

where $Z_t = [Y_t', Y_{t-1}', \ldots, Y_{t-p+1}']'$ and

$$
F = \begin{bmatrix}
A_1 & A_2 & \cdots & A_{p-1} & A_p \\
I_n & 0 & \cdots & 0 & 0 \\
0 & I_n & \cdots & 0 & 0 \\
\vdots & \vdots & \ddots & \vdots & \vdots \\
0 & 0 & \cdots & I_n & 0
\end{bmatrix}
$$

**Properties**

- $F$ is $(np \times np)$
- Eigenvalues of $F$ determine stability
- Used for IRF computation and forecasting

**Stability condition**: All eigenvalues of $F$ must lie inside the unit circle.


## Structural VAR (SVAR)

### From Reduced Form to Structural Form

The reduced-form errors $u_t$ are linear combinations of structural shocks $\varepsilon_t$:

$$
u_t = P \varepsilon_t
$$

where:
- $\varepsilon_t$ is $(n \times 1)$ with $E[\varepsilon_t] = 0$, $E[\varepsilon_t \varepsilon_t'] = I_n$ (orthonormal structural shocks)
- $P$ is $(n \times n)$ impact matrix (contemporaneous effects)

**Covariance relationship**:
$$
\Sigma_u = E[u_t u_t'] = E[P \varepsilon_t \varepsilon_t' P'] = P P'
$$

**Identification problem**: Given $\Sigma_u$, recover $P$. There are infinitely many solutions since for any orthonormal matrix $Q$:
$$
\Sigma_u = P P' = (PQ)(PQ)' = \tilde{P}\tilde{P}'
$$

**Solution**: Impose $n(n-1)/2$ restrictions to achieve identification.


## Identification Schemes

### 1. Recursive (Cholesky) Identification

**Restriction**: $P$ is lower triangular.

$$
P = \begin{bmatrix}
p_{11} & 0 & \cdots & 0 \\
p_{21} & p_{22} & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
p_{n1} & p_{n2} & \cdots & p_{nn}
\end{bmatrix}
$$

**Interpretation**: Variable ordering matters. First variable responds contemporaneously only to its own shock, second variable responds to first two shocks, etc.

**Computation**: Cholesky decomposition of $\Sigma_u$:
$$
P = \text{chol}(\Sigma_u)
$$

**Example** (3-variable oil market VAR):
- Ordering: Oil production → Economic activity → Oil price
- Oil production responds only to oil supply shocks contemporaneously
- Economic activity responds to supply and demand shocks
- Oil price responds to all three shocks

### 2. Sign Restrictions

**Restrictions**: Impose signs on IRF responses for specific horizons.

**Algorithm**:

1. Compute any decomposition $\Sigma_u = \tilde{P}\tilde{P}'$ (e.g., Cholesky)
2. Draw random orthonormal matrix $Q$ (Haar measure on $SO(n)$)
3. Compute candidate: $P = \tilde{P}Q$
4. Compute IRFs from $P$
5. Accept if IRFs satisfy sign restrictions; otherwise, reject and redraw

**Example restrictions** (oil market):

| Shock | Oil prod. | Real activity | Oil price | Horizon |
|-------|-----------|---------------|-----------|---------|
| Supply | + | ? | - | 0-12 |
| Demand | + | + | + | 0-12 |
| Spec. | ? | ? | + | 0-12 |

**Advantages**:

- Does not require ordering assumptions
- Based on economic theory
- Robust to different decompositions

**Disadvantages**:

- May not uniquely identify shocks
- Requires many draws for complex restrictions

### 3. Instrumental Variable (IV) Identification

**Setup**: Use external instrument $z_t$ that is:
1. **Relevant**: Correlated with target structural shock $\varepsilon_{jt}$
2. **Exogenous**: Uncorrelated with other structural shocks $\varepsilon_{kt}$ for $k \neq j$

**Identification equation**:
$$
\frac{E[u_t z_t]}{E[z_t^2]} = p_j \frac{E[\varepsilon_{jt} z_t]}{E[z_t^2]}
$$

where $p_j$ is the $j$-th column of $P$.

**Two-stage approach**

1. Regress reduced-form residuals on instrument: $\hat{\phi} = (Z'Z)^{-1}Z'U$
2. Normalize: $p_j = \hat{\phi} / \|\hat{\phi}\|$

**Examples**:

- Monetary policy: High-frequency surprises around FOMC announcements (Gertler-Karadi)
- Fiscal policy: Narrative military spending shocks (Ramey-Shapiro)
- Oil supply: Political events in oil-producing countries (Kilian)

## Impulse Response Functions

### Definition

The impulse response function (IRF) measures the dynamic effect of a one-time structural shock on the system.

**Orthogonalized IRF** at horizon $h$:
$$
\text{IRF}(h) = \Phi_h P
$$

where $\Phi_h$ is the reduced-form multiplier:
$$
\Phi_h = \frac{\partial Y_{t+h}}{\partial u_t'} = F^h J
$$

with $J = [I_n, 0_{n \times n(p-1)}]$ selecting the first $n$ elements of $Z_t$.

**Element interpretation**: $\text{IRF}(h)_{ij}$ is the response of variable $i$ at horizon $h$ to a one-standard-deviation shock to structural shock $j$.

### Computation Algorithm

```
Input: Coefficient matrices A₁, ..., Aₚ, structural impact matrix P
Output: IRF(0), IRF(1), ..., IRF(H)

1. Build companion matrix F
2. Compute powers: F⁰, F¹, F², ..., Fᴴ
3. For each h:
   Φₕ = Fʰ[1:n, 1:n]  (extract top-left n×n block)
   IRF(h) = Φₕ P
```

**Properties**

- IRF(0) = P (contemporaneous impact)
- IRF(h) → 0 as h → ∞ (for stable VAR)

### Cumulative IRF

Cumulative response up to horizon $h$:
$$
\text{CIRF}(h) = \sum_{s=0}^h \text{IRF}(s)
$$

**Use cases**:
- Permanent effects
- Integrated variables (e.g., log prices vs. inflation)

## Forecast Error Variance Decomposition (FEVD)

### Definition

The FEVD measures the contribution of each structural shock to the forecast error variance of each variable.

**h-step ahead forecast error**:
$$
Y_{t+h} - E_t[Y_{t+h}] = \sum_{s=0}^{h-1} \Phi_s u_{t+h-s} = \sum_{s=0}^{h-1} \Phi_s P \varepsilon_{t+h-s}
$$

**Variance**:
$$
E[(Y_{t+h} - E_t[Y_{t+h}])(Y_{t+h} - E_t[Y_{t+h}])'] = \sum_{s=0}^{h-1} \Phi_s P P' \Phi_s'
$$

**Contribution of shock $j$ to variable $i$**:
$$
\text{FEVD}_{ij}(h) = \frac{\sum_{s=0}^{h-1} (\Phi_s p_j)_i^2}{\sum_{k=1}^n \sum_{s=0}^{h-1} (\Phi_s p_k)_i^2}
$$

where $p_j$ is the $j$-th column of $P$.

**Properties**

- $\sum_{j=1}^n \text{FEVD}_{ij}(h) = 1$ for each variable $i$ and horizon $h$
- Depends on identification scheme (not invariant to ordering for Cholesky)


## Constrained VAR Estimation

### Zero Constraints

Restrict specific coefficients to zero:
$$
(A_\ell)_{ij} = 0 \quad \text{for specified } (i,j,\ell)
$$

**Equation-by-equation estimation**:

For equation $i$:
$$
Y_{it} = X_t' \beta_i + u_{it}
$$

Let $\mathcal{F}_i \subset \{1, \ldots, np+1\}$ be the indices of free parameters in equation $i$.

**Restricted OLS**:
$$
\hat{\beta}_i(\mathcal{F}_i) = (X_{\mathcal{F}_i}' X_{\mathcal{F}_i})^{-1} X_{\mathcal{F}_i}' Y_i
$$

where $X_{\mathcal{F}_i}$ contains only columns indexed by $\mathcal{F}_i$.

**Constrained coefficients**: Set to zero.

### Fixed-Value Constraints

Fix specific coefficients to known values:
$$
(A_\ell)_{ij} = c_{ij\ell}
$$

**Estimation algorithm**

1. Partition parameters: $\beta = [\beta_{\text{free}}', \beta_{\text{fixed}}']'$
2. Transform dependent variable: $\tilde{Y} = Y - X_{\text{fixed}} \beta_{\text{fixed}}$
3. Estimate free parameters: $\hat{\beta}_{\text{free}} = (X_{\text{free}}' X_{\text{free}})^{-1} X_{\text{free}}' \tilde{Y}$
4. Combine: $\hat{\beta} = [\hat{\beta}_{\text{free}}', \beta_{\text{fixed}}']'$

### Block Exogeneity

Variables in set $\mathcal{A}$ do not Granger-cause variables in set $\mathcal{B}$:
$$
(A_\ell)_{ij} = 0 \quad \forall i \in \mathcal{B}, \, j \in \mathcal{A}, \, \ell = 1, \ldots, p
$$

**Example**: Small open economy

- $\mathcal{A}$: Foreign variables
- $\mathcal{B}$: Domestic variables
- Constraint: Foreign variables do not respond to domestic variables

**Test**: Likelihood ratio test or F-test for joint significance.

## Inference Methods

### Asymptotic Inference

Under standard regularity conditions, the OLS estimator is asymptotically normal:
$$
\sqrt{T}(\hat{\beta} - \beta_0) \xrightarrow{d} N(0, V)
$$

where $V$ depends on the structure of the errors.

**Delta method** for functions of parameters:
$$
\sqrt{T}(g(\hat{\beta}) - g(\beta_0)) \xrightarrow{d} N(0, \nabla g(\beta_0)' V \nabla g(\beta_0))
$$

**Application**: Confidence intervals for IRFs.

**Limitations**

- Approximation quality depends on sample size
- Ignores estimation uncertainty in $\Sigma_u$
- Not robust to small-sample bias

### Bootstrap Inference

**Wild bootstrap algorithm**:

1. Estimate VAR: $\hat{A}_1, \ldots, \hat{A}_p, \hat{\Sigma}_u$
2. Compute residuals: $\hat{u}_t$
3. For $b = 1, \ldots, B$:
   a. Draw Rademacher weights: $w_t^{(b)} \sim \{-1, +1\}$ with equal probability
   b. Bootstrap residuals: $u_t^{(b)} = w_t^{(b)} \hat{u}_t$
   c. Simulate data: $Y_t^{(b)} = \hat{c} + \sum_{j=1}^p \hat{A}_j Y_{t-j}^{(b)} + u_t^{(b)}$
   d. Re-estimate VAR on $Y^{(b)}$: $\hat{A}_1^{(b)}, \ldots, \hat{A}_p^{(b)}, \hat{\Sigma}_u^{(b)}$
   e. Re-identify: $\hat{P}^{(b)}$
   f. Compute IRF: $\widehat{\text{IRF}}^{(b)}(h)$

4. Confidence interval: $(\text{IRF}_\alpha(h), \text{IRF}_{1-\alpha}(h))$ as $\alpha$ and $(1-\alpha)$ percentiles of $\{\widehat{\text{IRF}}^{(b)}(h)\}_{b=1}^B$

**Advantages**

- Valid in finite samples
- Incorporates all sources of uncertainty
- Robust to non-normality (wild bootstrap)

**Variants**:

- **Standard bootstrap**: Resample residuals with replacement
- **Block bootstrap**: Resample blocks for time series dependence
- **Bias-corrected bootstrap**: Adjust for finite-sample bias


## Model Selection

### Information Criteria

**Akaike Information Criterion (AIC)**:
$$
\text{AIC}(p) = \log|\hat{\Sigma}_u(p)| + \frac{2pn^2}{T}
$$

**Bayesian Information Criterion (BIC)**:
$$
\text{BIC}(p) = \log|\hat{\Sigma}_u(p)| + \frac{pn^2 \log T}{T}
$$

**Hannan-Quinn Information Criterion (HQIC)**:
$$
\text{HQIC}(p) = \log|\hat{\Sigma}_u(p)| + \frac{2pn^2 \log \log T}{T}
$$

**Lag selection**: Choose $p$ that minimizes the criterion.

**Properties**

- AIC: Asymptotically efficient but not consistent
- BIC: Consistent (selects true lag length as $T \to \infty$)
- HQIC: Intermediate between AIC and BIC

### Likelihood Ratio Test

Test $H_0: p = p_0$ vs. $H_1: p = p_1$ where $p_1 > p_0$:
$$
\text{LR} = T(\log|\hat{\Sigma}_u(p_0)| - \log|\hat{\Sigma}_u(p_1)|) \xrightarrow{d} \chi^2_{n^2(p_1-p_0)}
$$

**Use**: Sequential testing from $p=0$ upward.

## Stability and Stationarity

### Stability Condition

A VAR(p) is stable if all eigenvalues of the companion matrix $F$ lie strictly inside the unit circle:
$$
|\lambda_i(F)| < 1 \quad \forall i
$$

**Implications**:
- IRFs decay to zero
- Unconditional moments exist
- Forecasts converge to unconditional mean

### Checking Stability

```julia
F = companion_form(model)
eigenvalues = eigvals(F)
is_stable = all(abs.(eigenvalues) .< 1)
```

### Cointegration

If some variables are integrated (unit roots), the VAR should be:

1. **Estimated in differences** (VAR in first differences), or
2. **Estimated in levels with cointegration** (Vector Error Correction Model, VECM)

**VECM representation**:
$$
\Delta Y_t = \alpha \beta' Y_{t-1} + \sum_{j=1}^{p-1} \Gamma_j \Delta Y_{t-j} + u_t
$$

where $\beta$ are cointegrating vectors and $\alpha$ are adjustment coefficients.

*Note: VECM not yet implemented in this package.*


## References

### Textbooks

- Lütkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*. Springer.
- Hamilton, J. D. (1994). *Time Series Analysis*. Princeton University Press.
- Kilian, L., & Lütkepohl, H. (2017). *Structural Vector Autoregressive Analysis*. Cambridge University Press.

### Identification

- Sims, C. A. (1980). Macroeconomics and Reality. *Econometrica*, 48(1), 1-48.
- Rubio-Ramírez, J. F., Waggoner, D. F., & Zha, T. (2010). Structural Vector Autoregressions: Theory of Identification and Algorithms for Inference. *Review of Economic Studies*, 77(2), 665-696.
- Stock, J. H., & Watson, M. W. (2012). Disentangling the Channels of the 2007-2009 Recession. *Brookings Papers on Economic Activity*, 2012(1), 81-135.

### Inference

- Kilian, L. (1998). Small-Sample Confidence Intervals for Impulse Response Functions. *Review of Economics and Statistics*, 80(2), 218-230.
- Gonçalves, S., & Kilian, L. (2004). Bootstrapping Autoregressions with Conditional Heteroskedasticity of Unknown Form. *Journal of Econometrics*, 123(1), 89-120.
