# Benchmarks for MacroEconometricTools.jl

This directory contains benchmarks for tracking performance of the package over time.

## Quick Start

### Benchmark Current State Only

To benchmark just the current working directory state and create a markdown table:

```bash
julia --project=benchmark benchmark/run_benchmark_current.jl
```

This will:
- Run all benchmarks defined in `benchmarks.jl`
- Save results to `benchmark/results/`
- Create a markdown table at `benchmark/results/benchmark_current_TIMESTAMP.md`

### Compare Against main Branch

To compare current state against the main branch:

```bash
julia --project=benchmark benchmark/run_asv.jl
```

By default this compares:
- **Baseline**: `origin/main`
- **Candidate**: Current working directory (dirty state)

## Configuration

### Environment Variables for `run_asv.jl`

- `MET_ASV_OUTPUT_DIR`: Output directory (default: `benchmark/results`)
- `MET_ASV_BASELINE`: Baseline revision (default: `origin/main`)
- `MET_ASV_CANDIDATE`: Candidate revision (default: `dirty`)
- `MET_ASV_FILTER`: Comma-separated benchmark name patterns to run
- `MET_ASV_TUNE`: Set to `true` for more accurate results (slower)
- `MET_ASV_LOAD_SAMPLES`: Number of load time samples (default: 5)

### Examples

**Benchmark only IRF-related tests:**
```bash
MET_ASV_FILTER="irf" julia --project=benchmark benchmark/run_asv.jl
```

**Compare two specific commits:**
```bash
MET_ASV_BASELINE="v0.1.0" MET_ASV_CANDIDATE="HEAD" \
  julia --project=benchmark benchmark/run_asv.jl
```

**Enable tuning for more accurate results (slower):**
```bash
MET_ASV_TUNE=true julia --project=benchmark benchmark/run_benchmark_current.jl
```

## Benchmark Configuration

### What's Benchmarked

The benchmarks in `benchmarks.jl` test:

1. **VAR Estimation** (`estimation` group)
   - OLS VAR with 5 lags

2. **IRF Computation** (`irf` group)
   - Point IRF computation
   - Bootstrap IRF computation

3. **Bootstrap** (`bootstrap` group)
   - Serial wild bootstrap
   - Distributed wild bootstrap (if workers available)

4. **Sign Restrictions** (`sign_restrictions` group)
   - Serial identification
   - Distributed identification (if workers available)

### Controlling Benchmark Parameters

Set these environment variables before running:

- `MET_BENCH_SEED`: Random seed (default: 20240612)
- `MET_BENCH_REPS`: Bootstrap replications (default: 250)
- `MET_BENCH_HORIZON`: IRF horizon (default: 20)
- `MET_BENCH_IRF_REPS`: IRF bootstrap reps (default: 100)
- `MET_BENCH_SIGN_DRAWS`: Max sign restriction draws (default: 10000)
- `MET_BENCH_ENABLE_DISTRIBUTED`: Enable distributed tests (default: false)
- `MET_BENCH_NWORKERS`: Number of workers for distributed (default: CPU_THREADS)

**Example: Run heavier benchmarks**
```bash
MET_BENCH_REPS=500 MET_BENCH_IRF_REPS=200 \
  julia --project=benchmark benchmark/run_benchmark_current.jl
```

**Example: Enable distributed benchmarks**
```bash
MET_BENCH_ENABLE_DISTRIBUTED=true MET_BENCH_NWORKERS=4 \
  julia --project=benchmark benchmark/run_benchmark_current.jl
```

## Output

Results are saved to `benchmark/results/` directory:

- `benchmark_current_TIMESTAMP.md`: Markdown table for current state only
- `summary_TIMESTAMP.md`: Markdown comparison table (from `run_asv.jl`)
- Raw JSON results files (used by AirspeedVelocity.jl)

## Reading Results

The markdown tables show:

- **Benchmark name**: What's being measured
- **Time**: Median execution time (with min-max range)
- **Memory**: Peak memory allocation
- **Allocs**: Number of allocations
- **Speedup**: Ratio vs baseline (in `run_asv.jl` output only)

### Interpreting Speedup

In comparison tables:
- `1.0x` = No change
- `< 1.0x` = Regression (slower)
- `> 1.0x` = Improvement (faster)

Example:
```
| benchmark  | time (candidate) | vs. baseline |
|------------|------------------|--------------|
| estimate   | 1.2 ms          | 1.5x faster  |
| bootstrap  | 850 ms          | 0.8x slower  |
```

## Tips

1. **Warmup**: First run is slower due to compilation. Run benchmarks twice for accurate results.

2. **Consistency**: Close other applications to reduce system noise affecting results.

3. **Tuning**: Enable tuning (`MET_ASV_TUNE=true`) for publication-quality results, but expect 5-10x longer runtime.

4. **Distributed**: Distributed benchmarks require `Distributed` package and are more variable. Run multiple times.

5. **Version Control**: Commit changes before benchmarking for accurate revision tracking.

## Integration with CI

To track performance regressions in CI, run:

```bash
julia --project=benchmark -e '
using AirspeedVelocity
# Compare PR against main
results = AirspeedVelocity.benchmark(
    ["origin/main", "HEAD"];
    output_dir="benchmark/results"
)
# Fail if >20% regression in any benchmark
check_performance_regression(results, threshold=1.2)
'
```

## Troubleshooting

**Error: "Distributed package not loaded"**
- Distributed benchmarks are skipped if `Distributed` is not available
- This is normal and not an error

**Error: "Package MacroEconometricTools not found"**
- Make sure you're in the package root directory
- Try running `using Pkg; Pkg.develop(path=".")` first

**Very slow benchmarks**
- Reduce `MET_BENCH_REPS` and `MET_BENCH_IRF_REPS`
- Disable distributed benchmarks with `MET_BENCH_ENABLE_DISTRIBUTED=false`

**High variance in results**
- Enable tuning: `MET_ASV_TUNE=true`
- Close background applications
- Run on a dedicated machine
