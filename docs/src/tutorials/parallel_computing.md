# Parallel Computing with MacroEconometricTools.jl

This tutorial demonstrates how to use distributed computing to speed up computationally intensive tasks like bootstrap inference and sign restriction identification.

## Why Distributed Computing?

Two operations in MacroEconometricTools.jl are **embarrassingly parallel**:

1. **Bootstrap replications**: Each bootstrap draw is independent
2. **Sign restriction search**: Each rotation draw is independent

Using multiple processes can provide near-linear speedup.

---

## Setup

### Adding Worker Processes

```julia
using Distributed

# Add 4 worker processes
addprocs(4)

# Check workers
println("Number of workers: ", nworkers())
println("Worker IDs: ", workers())
```

Output:
```
Number of workers: 4
Worker IDs: [2, 3, 4, 5]
```

### Loading Package on Workers

**Important**: The package must be loaded on all workers using `@everywhere`.

```julia
@everywhere using MacroEconometricTools
@everywhere using LinearAlgebra, Random
```

### Verify Setup

```julia
# Test that package is available on all workers
@everywhere println("Worker ", myid(), " ready")
```

---

## Parallel Bootstrap

### Basic Usage

```julia
using MacroEconometricTools

# Load data
include("data/oil_data.jl")
data = load_oil_data()
Y = data.data
names = data.names

# Estimate VAR
p = 24
var_model = estimate(OLSVAR, Y, p; names=names)

# Identification
id = CholeskyID()

# Serial bootstrap (baseline)
@time irf_serial = bootstrap_irf(var_model, id, 24, 1000;
                                 parallel=:none)

# Distributed bootstrap (parallel)
@time irf_parallel = bootstrap_irf(var_model, id, 24, 1000;
                                   parallel=:distributed)
```

### Expected Speedup

With 4 workers:
```
Serial:      ~45.2 seconds
Parallel:    ~12.8 seconds
Speedup:     3.5× (close to ideal 4×)
```

**Note**: Speedup depends on problem size, number of workers, and overhead.

### Performance Tips

#### 1. **Use enough bootstrap replications**

```julia
# Too few reps: overhead dominates
bootstrap_irf(var_model, id, 24, 100; parallel=:distributed)  # Not worth it

# Optimal: enough work per worker
bootstrap_irf(var_model, id, 24, 2000; parallel=:distributed)  # Good
```

**Rule of thumb**: At least 50-100 replications per worker.

#### 2. **Consider model complexity**

```julia
# Simple model: may not benefit much
var_small = estimate(OLSVAR, Y[:,1:2], 4)  # 2 vars, 4 lags

# Complex model: benefits more
var_large = estimate(OLSVAR, Y, 24)  # 3 vars, 24 lags
```

Larger models have longer estimation time per replication → more benefit from parallelization.

### Advanced: Controlling Workers

```julia
# Use specific number of workers
using Distributed

# Remove existing workers
while nworkers() > 1
    rmprocs(workers()[end])
end

# Add optimal number (typically = # CPU cores)
n_cores = 8
addprocs(n_cores)
@everywhere using MacroEconometricTools

# Run bootstrap
irf_result = bootstrap_irf(var_model, id, 48, 5000;
                          parallel=:distributed)
```

---

## Parallel Sign Restrictions

Sign restriction identification can be very slow for complex restrictions. Parallel search provides substantial speedup.

### Basic Usage

```julia
# Define sign restrictions
restrictions = [
    -1   1   0;   # Supply shock: prod ↓, activity ?, price ↑
     0   1   0;   # Demand shock: prod ?, activity ↑, price ?
     1   1   1    # Spec. demand: prod ↑, activity ↑, price ↑
]

id_sign = SignRestriction(restrictions, 12)

# Serial search
@time P_serial = identify(var_model, id_sign;
                         max_draws=50000,
                         parallel=:none,
                         verbose=true)

# Distributed search
@time P_parallel = identify(var_model, id_sign;
                           max_draws=50000,
                           parallel=:distributed,
                           verbose=true)
```

### Expected Output

Serial:
```
Completed 10000 draws, no valid rotation found yet...
Completed 20000 draws, no valid rotation found yet...
Found valid rotation at attempt 23847
 18.432 seconds
```

Parallel (4 workers):
```
Searching for valid rotation using 4 workers (12500 draws each)...
Worker 2 found valid rotation at attempt 5821
  5.243 seconds
```

**Speedup**: ~3.5× (close to 4× with 4 workers)

### When to Use Parallel

#### Good Cases:
- **Strict restrictions** (low acceptance rate < 5%)
- **Many variables** (n ≥ 5)
- **Long horizon** for restrictions (h ≥ 12)

#### Less beneficial:
- **Loose restrictions** (high acceptance rate > 20%)
- **Few variables** (n ≤ 3)
- **Short horizon** (h ≤ 4)

### Monitoring Search

```julia
# Enable verbose mode to track progress
P = identify(var_model, id_sign;
            max_draws=100000,
            parallel=:distributed,
            verbose=true)
```

Output shows which worker finds the solution first:
```
Searching for valid rotation using 4 workers (25000 draws each)...
Worker 3 found valid rotation at attempt 15234
```

---

## Combined: Parallel Bootstrap + Sign Restrictions

The most computationally intensive case: bootstrap with sign restrictions.

### Setup

```julia
using Distributed

# Use maximum cores
addprocs(Sys.CPU_THREADS)
@everywhere using MacroEconometricTools

println("Using ", nworkers(), " workers")
```

### High-Level Interface

```julia
# Estimate VAR
var_model = estimate(OLSVAR, Y, 24; names=names)

# Sign restrictions
id_sign = SignRestriction(restrictions, 12)

# Compute IRFs with bootstrap (automatically uses parallel)
irf_result = irf(var_model, id_sign;
                horizon=48,
                inference=:bootstrap,
                bootstrap_reps=2000,
                parallel=:distributed)  # Use distributed for both bootstrap AND sign search
```

**Note**: The `irf()` function will automatically use distributed computing for both the bootstrap loop AND the sign restriction search within each replication.

### Performance Example

Problem: 3 variables, 24 lags, 48-period horizon, 2000 bootstrap reps

```
Serial (1 core):     ~3.5 hours
Parallel (8 cores):  ~32 minutes
Speedup:             ~6.5×
```

**Not perfect 8× because**:
- Communication overhead
- Load balancing
- Some serial components

---

## Best Practices

### 1. **Match workers to cores**

```julia
# Get number of CPU threads
n_threads = Sys.CPU_THREADS

# Add workers (leave 1-2 cores for OS)
addprocs(n_threads - 2)
```

### 2. **Use distributed for large jobs**

```julia
# Small job: not worth overhead
irf(var, id; horizon=12, bootstrap_reps=100)  # Use parallel=:none

# Large job: worth it
irf(var, id; horizon=48, bootstrap_reps=5000; parallel=:distributed)
```

### 3. **Monitor memory usage**

Each worker needs its own copy of data:

```julia
# Check memory per worker
@everywhere using InteractiveUtils
@everywhere varinfo()
```

### 4. **Graceful shutdown**

```julia
# Remove workers when done
rmprocs(workers())
```

---

## Troubleshooting

### "No worker processes available"

```julia
# Error: No worker processes available. Falling back to serial.

# Solution:
using Distributed
addprocs(4)
@everywhere using MacroEconometricTools
```

### "Package not found on worker"

```julia
# Error: UndefVarError: MacroEconometricTools not defined

# Solution: Use @everywhere
@everywhere using MacroEconometricTools
```

### Slow startup

If loading the package on workers is slow:

```julia
# Precompile once
using MacroEconometricTools

# Then add workers
addprocs(4)

# Load (will use precompiled version)
@everywhere using MacroEconometricTools
```

### Out of memory

Too many workers or too large model:

```julia
# Reduce number of workers
rmprocs(workers()[end-2:end])  # Remove last 3 workers

# Or batch the bootstrap
n_batches = 5
batch_size = div(total_reps, n_batches)

for batch in 1:n_batches
    irf_batch = bootstrap_irf(var, id, horizon, batch_size;
                              parallel=:distributed)
    # Save or process batch
end
```

---

## Benchmarking

Compare serial vs. parallel for your specific problem:

```julia
using BenchmarkTools

# Your VAR model
var = estimate(OLSVAR, Y, 12)
id = CholeskyID()

# Benchmark serial
@btime bootstrap_irf($var, $id, 24, 500; parallel=:none)

# Benchmark parallel
@btime bootstrap_irf($var, $id, 24, 500; parallel=:distributed)
```

### Expected Results

| Configuration | Time | Speedup |
|--------------|------|---------|
| Serial (1 core) | 45.2 s | 1.0× |
| Parallel (2 cores) | 24.1 s | 1.9× |
| Parallel (4 cores) | 12.8 s | 3.5× |
| Parallel (8 cores) | 7.2 s | 6.3× |

**Diminishing returns** beyond ~4-8 cores for typical VAR problems.

---

## Summary

### When to Use Parallel

✅ **Use distributed for**:
- Bootstrap with ≥1000 replications
- Sign restrictions with low acceptance rates
- Large VARs (many variables or lags)
- Long IRF horizons

❌ **Don't use for**:
- Small jobs (< 100 bootstrap reps)
- Simple calculations (Cholesky ID + no bootstrap)
- Memory-constrained systems

### Setup Checklist

1. ✅ Add workers: `addprocs(n)`
2. ✅ Load package everywhere: `@everywhere using MacroEconometricTools`
3. ✅ Verify: `nworkers() > 1`
4. ✅ Run with `parallel=:distributed`
5. ✅ Clean up: `rmprocs(workers())`

### Performance Tips

- Match workers to physical cores
- Use ≥50 bootstrap reps per worker
- Monitor memory usage
- Benchmark your specific problem

---

## Additional Resources

- [Julia Distributed Computing Documentation](https://docs.julialang.org/en/v1/manual/distributed-computing/)
- [Parallel Bootstrap Theory](../mathematical/theory.md#bootstrap-inference)
- [Technical Implementation](../../TECHNICAL.md#parallel-computing)

---

## Example Scripts

### Complete Parallel Bootstrap Example

```julia
# parallel_bootstrap_example.jl

using Distributed
addprocs(4)

@everywhere using MacroEconometricTools
@everywhere using LinearAlgebra, Random

# Load data
include("data/oil_data.jl")
data = load_oil_data()
Y = data.data
names = data.names

# Estimate VAR
println("Estimating VAR...")
var_model = estimate(OLSVAR, Y, 24; names=names)

# Identification
id = CholeskyID()

# Parallel bootstrap
println("\nRunning parallel bootstrap on $(nworkers()) workers...")
@time irf_result = irf(var_model, id;
                       horizon=48,
                       inference=:bootstrap,
                       bootstrap_reps=5000,
                       parallel=:distributed,
                       coverage=[0.68, 0.90, 0.95])

println("\nBootstrap complete!")
println("IRF dimensions: ", size(irf_result.irf))

# Cleanup
rmprocs(workers())
```

### Complete Sign Restriction Example

```julia
# parallel_sign_restrictions.jl

using Distributed
addprocs(8)  # Use more workers for sign restrictions

@everywhere using MacroEconometricTools

# Load data and estimate VAR
include("data/oil_data.jl")
data = load_oil_data()
var_model = estimate(OLSVAR, data.data, 24; names=data.names)

# Define sign restrictions
restrictions = [
    -1   1   0;
     0   1   0;
     1   1   1
]

id_sign = SignRestriction(restrictions, 12)

# Parallel search
println("Searching for valid rotation on $(nworkers()) workers...")
@time P = identify(var_model, id_sign;
                  max_draws=100000,
                  parallel=:distributed,
                  verbose=true)

println("\nFound valid rotation!")
display(round.(P, digits=3))

# Cleanup
rmprocs(workers())
```
