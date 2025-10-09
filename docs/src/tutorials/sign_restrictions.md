# Sign Restrictions for Structural Identification

This tutorial demonstrates how to identify structural shocks using sign restrictions, following the methodology of Uhlig (2005) and Rubio-Ramírez et al. (2010).

## Motivation

Cholesky identification requires ordering assumptions that may not be economically justified. Sign restrictions offer an alternative by imposing theory-based restrictions on IRF signs.

## Example: Oil Market with Three Shocks

Following Kilian & Murphy (2012), we identify three structural shocks in the oil market:

1. **Oil supply shock**: Reduces production, increases price
2. **Aggregate demand shock**: Increases production, activity, and price
3. **Oil-specific demand shock**: Increases price, reduces production (later)

### Sign Restriction Matrix

```julia
using MacroEconometricTools

# Load data
include("data/oil_data.jl")
data = load_oil_data()
Y = data.data
names = data.names  # [:oil_prod_growth, :real_activity, :log_oil_price]

# Estimate VAR
p = 24
var_model = fit(OLSVAR, Y, p; names=names)
```

Define sign restrictions (rows = variables, columns = shocks):

```julia
#                      Supply  Demand  Spec.Demand
# oil_prod_growth        -       +        ?
# real_activity          ?       +        ?
# log_oil_price          +       +        +

# Create restriction matrix
# Values: +1 (positive), -1 (negative), 0 or NaN (unrestricted)
restrictions = [
    -1   1   0;   # oil production
     0   1   0;   # real activity
     1   1   1    # oil price
]

# Identification scheme
horizon_restrict = 12  # Restrictions apply for first 12 months
id_sign = SignRestriction(restrictions, horizon_restrict)
```

### Identification Algorithm

```julia
# Identify structural shocks - find ONE rotation matrix
# This may take a while - trying many rotations
P_sign = rotation_matrix(var_model, id_sign; max_draws=10000, verbose=true)

println("Found rotation satisfying sign restrictions")
println("Impact matrix:")
display(round.(P_sign, digits=3))
```

Expected output:
```
Found rotation satisfying sign restrictions (attempt 3847)
Impact matrix:
 -0.987   0.042   0.156
  0.123   1.986   0.345
  0.511   0.087   0.026
```

**Interpretation**: Columns are structural shocks (supply, demand, spec. demand).

### Set Identification: Multiple Valid Rotations

Sign restrictions are **set-identified** - many rotation matrices satisfy the restrictions.
To account for this, compute IRFs for multiple draws:

```julia
# Compute IRFs with multiple rotation draws
irf_result = irf(var_model, id_sign;
                n_draws=1000,       # Number of valid rotations to find
                max_attempts=10000, # Max attempts per draw
                horizon=48)         # Returns SignRestrictedIRFResult

# Access the results
irf_result.irf_median          # Median IRF across draws
irf_result.irf_draws           # All IRF draws: (1000, 49, 3, 3)
irf_result.rotation_matrices   # All 1000 rotation matrices
irf_result.lower               # Pointwise quantile bands
irf_result.upper
```

### Understanding the Algorithm

The algorithm works as follows:

1. **Initial decomposition**: Compute Cholesky $\Sigma_u = \tilde{P}\tilde{P}'$

2. **Random rotation**: Draw orthonormal matrix $Q \in SO(n)$ from Haar distribution
   ```julia
   Q = random_orthonormal_matrix(n)
   ```

3. **Candidate impact matrix**: $P = \tilde{P}Q$
   - Note: $PP' = \tilde{P}QQ'\tilde{P}' = \tilde{P}\tilde{P}' = \Sigma_u$ ✓

4. **Compute candidate IRFs**: $\text{IRF}(h) = \Phi_h P$

5. **Check signs**: For each horizon $h \leq H$ and each restricted element $(i,j)$:
   ```julia
   if restrictions[i,j] == 1
       satisfies = all(IRF[1:H, i, j] .> 0)
   elseif restrictions[i,j] == -1
       satisfies = all(IRF[1:H, i, j] .< 0)
   else
       satisfies = true  # unrestricted
   end
   ```

6. **Accept or reject**: If all restrictions satisfied, accept $P$; otherwise, draw new $Q$

### Plotting Sign-Restricted IRFs

The package provides three visualization modes for set-identified IRFs:

```julia
using Plots

# 1. Quantiles only (median + bands) - DEFAULT
plot(irf_result; plot_type=:quantiles,
                vars=:all,
                shocks=:all)

# 2. All IRF paths (shows set identification uncertainty)
plot(irf_result; plot_type=:paths,
                path_alpha=0.02,    # Transparency for paths
                path_color=:gray)

# 3. Both paths and quantiles
plot(irf_result; plot_type=:both,
                path_alpha=0.015,
                median_color=:black)
```

**Interpretation**:
- `:quantiles` mode: Shows median IRF and pointwise confidence bands
- `:paths` mode: Each gray line is one valid IRF (shows full set identification)
- `:both` mode: Combines both visualizations

## Computing IRFs with Sign Restrictions

### Single Rotation (Point Estimate)

```julia
# Get one rotation matrix, then compute IRFs
P = rotation_matrix(var_model, id_sign; max_draws=10000)
irf_point = irf(var_model, CholeskyID(); horizon=48, inference=:none)
# Note: This uses only ONE rotation, ignoring set identification
```

### Multiple Rotations (Set Identification)
irf_sign = irf(var_model, id_sign;
               horizon=48,
               inference=:none)  # No inference for now

# Plot IRF of oil price to supply shock
h_range = 0:48
response = irf_sign.irf[:, 3, 1]  # oil price, supply shock

println("Oil price response to negative supply shock:")
for h in [0, 6, 12, 24]
    println("  h=$h: ", round(response[h+1], digits=4))
end
```

Expected signs:
```
Oil price response to negative supply shock:
  h=0: 0.0511   # Positive (supply ↓ → price ↑)
  h=6: 0.0387
  h=12: 0.0245
  h=24: 0.0089
```

### Set Identification

Unlike Cholesky, sign restrictions typically don't uniquely identify shocks. Multiple rotations may satisfy the restrictions.

```julia
# Collect multiple valid rotations
n_rotations = 100
valid_rotations = []
valid_irfs = []

attempts = 0
max_attempts = 100000

while length(valid_rotations) < n_rotations && attempts < max_attempts
    attempts += 1

    # Draw random rotation
    Q = random_orthonormal_matrix(3)
    P_candidate = cholesky(vcov(var_model)).L * Q

    # Compute IRFs
    irf_candidate = compute_irf_point(var_model, P_candidate, horizon_restrict)

    # Check restrictions
    if check_sign_restrictions(irf_candidate, restrictions, horizon_restrict)
        push!(valid_rotations, P_candidate)
        irf_full = compute_irf_point(var_model, P_candidate, 48)
        push!(valid_irfs, irf_full)
    end
end

println("Found $n_rotations valid rotations in $attempts attempts")
```

### Identified Set of IRFs

```julia
# Stack IRFs
irf_array = cat(valid_irfs..., dims=4)  # (H, n_vars, n_shocks, n_rotations)

# Compute median and percentiles
irf_median = median(irf_array, dims=4)[:,:,:,1]
irf_16 = [percentile(irf_array[h,i,j,:], 16) for h in 1:49, i in 1:3, j in 1:3]
irf_84 = [percentile(irf_array[h,i,j,:], 84) for h in 1:49, i in 1:3, j in 1:3]

# Display identified set for oil price response to supply shock
println("\nIdentified set (68% of valid rotations):")
println("Horizon | Median | [16th, 84th] percentile")
for h in [0, 6, 12, 24]
    println(@sprintf("%7d | %6.4f | [%6.4f, %6.4f]",
            h, irf_median[h+1,3,1], irf_16[h+1,3,1], irf_84[h+1,3,1]))
end
```

**Interpretation**: The range shows the identified set - all IRFs consistent with the sign restrictions.

## Bootstrap Inference with Sign Restrictions

Combining sign restrictions with bootstrap accounts for:

1. **Set identification**: Multiple rotations satisfy restrictions
2. **Parameter uncertainty**: Bootstrap over VAR parameters

### Two-Stage Bootstrap

```julia
function bootstrap_sign_restrictions(model, id::SignRestriction;
                                    horizon=48,
                                    bootstrap_reps=500,
                                    rotations_per_draw=10,
                                    method=:wild)

    n_vars = n_vars(model)
    irf_storage = zeros(horizon+1, n_vars, n_vars, bootstrap_reps * rotations_per_draw)

    idx = 1
    for b in 1:bootstrap_reps
        # 1. Bootstrap VAR parameters
        ε_boot = resample_residuals(model.residuals, method)
        Y_boot = simulate_var(model, ε_boot, Y[1:p,:])
        model_boot = fit(typeof(model.spec), Y_boot, n_lags(model))

        # 2. Find valid rotations for bootstrap model
        Σ_boot = vcov(model_boot)
        P_chol = cholesky(Σ_boot).L

        attempts = 0
        found = 0
        while found < rotations_per_draw && attempts < 1000
            attempts += 1
            Q = random_orthonormal_matrix(n_vars)
            P_cand = P_chol * Q
            irf_cand = compute_irf_point(model_boot, P_cand, id.horizon)

            if check_sign_restrictions(irf_cand, id.restrictions, id.horizon)
                # Compute full horizon IRF
                P_valid = P_cand
                irf_full = compute_irf_point(model_boot, P_valid, horizon)
                irf_storage[:,:,:,idx] = irf_full
                idx += 1
                found += 1
            end
        end

        if found == 0
            @warn "Could not find valid rotation for bootstrap draw $b"
        end
    end

    return irf_storage[:,:,:,1:(idx-1)]
end
```

### Computing Confidence Bands

```julia
# Run bootstrap
irf_boot = bootstrap_sign_restrictions(var_model, id_sign;
                                       horizon=48,
                                       bootstrap_reps=500,
                                       rotations_per_draw=10,
                                       method=:wild)

# Compute percentiles
irf_median = median(irf_boot, dims=4)[:,:,:,1]
irf_05 = [percentile(irf_boot[h,i,j,:], 5) for h in 1:49, i in 1:3, j in 1:3]
irf_95 = [percentile(irf_boot[h,i,j,:], 95) for h in 1:49, i in 1:3, j in 1:3]

# Display results
println("Oil price response to supply shock (with 90% confidence bands):")
println("Horizon | Median | 90% CI")
for h in [0, 3, 6, 12, 18, 24]
    println(@sprintf("%7d | %6.4f | [%6.4f, %6.4f]",
            h, irf_median[h+1,3,1], irf_05[h+1,3,1], irf_95[h+1,3,1]))
end
```

**Interpretation**: Confidence intervals now account for both set identification and parameter uncertainty.


## Comparison: Cholesky vs Sign Restrictions

### Same Data, Different Identification

```julia
# Cholesky identification (production → activity → price)
id_chol = CholeskyID()
irf_chol = irf(var_model, id_chol; horizon=48, inference=:none)

# Sign restrictions
irf_sign = irf(var_model, id_sign; horizon=48, inference=:none)

# Compare first shock (interpreted as supply shock)
println("Impact of first shock on oil price:")
println("  Cholesky: ", round(irf_chol.irf[1,3,1], digits=4))
println("  Sign restrictions (median): ", round(irf_median[1,3,1], digits=4))
```

### Narrative Sign Restrictions

Combine sign restrictions with narrative approach (Antolín-Díaz & Rubio-Ramírez, 2018):

```julia
# Additional restriction: Largest oil price increase in sample
# should be attributed to supply shock

# Find date of largest oil price change
price_changes = diff(Y[:,3])
max_change_idx = argmax(price_changes)
max_change_date = dates[max_change_idx+1]

println("Largest oil price increase: $max_change_date")
println("Magnitude: ", round(price_changes[max_change_idx], digits=3))

# Impose that this was mainly a supply shock
# (This would require extending the identification algorithm)
```


## Advanced Topics

### Combining with Other Restrictions

#### Zero Restrictions

Some coefficients known to be zero:

```julia
# Example: Oil production weakly exogenous
constraints = [ZeroConstraint(:oil_prod_growth, [:real_activity, :log_oil_price])]

var_constrained = fit(OLSVAR, Y, p; names=names, constraints=constraints)

# Then apply sign restrictions
id_combined = SignRestriction(restrictions, 12)
P_combined = identify(var_constrained, id_combined; max_draws=10000)
```

#### Magnitude Restrictions

Restrict relative magnitudes of responses:

```julia
# Example: Oil price responds more to demand than supply shock at h=0
# This requires custom identification algorithm
```

### Bayesian Sign Restrictions

Instead of the frequentist "accept/reject" approach, use Bayesian posterior:

$$
p(\theta | Y, \text{sign restrictions}) \propto p(Y | \theta) p(\theta) \mathbb{1}[\text{restrictions hold}]
$$

where $\mathbb{1}[\cdot]$ is an indicator function.

**Algorithm**

1. Draw VAR parameters from posterior: $A, \Sigma | Y$
2. Draw rotation $Q$ uniformly on $SO(n)$
3. Check sign restrictions
4. Keep draw if restrictions satisfied

*Note: Bayesian VAR not yet fully implemented in this package.*


## Practical Considerations

### Choosing Restrictions

**Good practices**:
1. Base on economic theory, not data
2. Use minimal set of restrictions
3. Report sensitivity to restriction choices
4. Consider identified set width

**Warning signs**:
- Too many valid rotations → weak identification
- Too few valid rotations → over-identified or misspecified

### Computational Efficiency

For large systems, finding valid rotations can be slow:

```julia
# Monitor acceptance rate
function identify_with_monitoring(model, id; max_draws=10000)
    accepts = 0
    attempts = 0

    for draw in 1:max_draws
        attempts += 1
        # ... standard algorithm ...

        if satisfies_restrictions
            accepts += 1
            acceptance_rate = accepts / attempts
            if accepts % 100 == 0
                println("Found $accepts valid draws, ",
                       "acceptance rate: $(round(acceptance_rate*100, digits=2))%")
            end
        end
    end
end
```

**Typical acceptance rates**:
- 3 variables, loose restrictions: 10-30%
- 5 variables, moderate restrictions: 1-5%
- 7+ variables, strict restrictions: <1%

### Diagnostics

Check if restrictions are too strong:

```julia
# Distribution of first eigenvalue of valid rotations
eigenvals_valid = [maximum(abs.(eigvals(companion_form(get_lags(model, P)))))
                   for P in valid_rotations]

# Should be similar to unrestricted
eigenvals_unrestricted = bootstrap_eigenvalues(var_model, 1000)

# Compare distributions
println("Max eigenvalue:")
println("  Valid rotations: ", round(mean(eigenvals_valid), digits=3),
        " (", round(std(eigenvals_valid), digits=3), ")")
println("  Unrestricted: ", round(mean(eigenvals_unrestricted), digits=3),
        " (", round(std(eigenvals_unrestricted), digits=3), ")")
```


## References

- Uhlig, H. (2005). What Are the Effects of Monetary Policy on Output? Results from an Agnostic Identification Procedure. *Journal of Monetary Economics*, 52(2), 381-419.

- Rubio-Ramírez, J. F., Waggoner, D. F., & Zha, T. (2010). Structural Vector Autoregressions: Theory of Identification and Algorithms for Inference. *Review of Economic Studies*, 77(2), 665-696.

- Kilian, L., & Murphy, D. P. (2012). Why Agnostic Sign Restrictions Are Not Enough: Understanding the Dynamics of Oil Market VAR Models. *Journal of the European Economic Association*, 10(5), 1166-1188.

- Antolín-Díaz, J., & Rubio-Ramírez, J. F. (2018). Narrative Sign Restrictions for SVARs. *American Economic Review*, 108(10), 2802-2829.

- Arias, J. E., Rubio-Ramírez, J. F., & Waggoner, D. F. (2018). Inference Based on Structural Vector Autoregressions Identified with Sign and Zero Restrictions: Theory and Applications. *Econometrica*, 86(2), 685-720.
