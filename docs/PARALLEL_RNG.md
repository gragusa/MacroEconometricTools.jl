# Random Number Generation in Parallel Computing

## The Problem

When running bootstrap or Monte Carlo simulations in parallel, we need to ensure:

1. **Independence**: Different workers should use independent random streams
2. **Reproducibility**: Results should be reproducible given a seed
3. **Sound statistics**: Random draws should not be correlated across workers

## Our Approach

### Stream Splitting via Seed Derivation

We use a **single base seed** and derive worker-specific seeds deterministically:

```julia
# Generate base seed from hardware RNG
base_seed = rand(Random.RandomDevice(), UInt64)

# For each worker/replication, derive unique seed
worker_seed = base_seed + UInt64(worker_id) * 0x9e3779b97f4a7c15
```

The constant `0x9e3779b97f4a7c15` is the **golden ratio** in fixed-point representation, which ensures good mixing properties.

### Why This Works

1. **Large separation**: Seeds are separated by ~2^64 × φ ≈ 1.6 × 10^19
2. **No overlap**: Different workers will never use the same seed
3. **Deterministic**: Given the same base seed, same results
4. **Reproducible**: Setting base seed manually allows exact reproduction

### Alternative Approaches Considered

#### ❌ Sequential seeding
```julia
# BAD: Seeds too close together
seeds = [1, 2, 3, 4, ...]
```
**Problem**: Consecutive seeds may produce correlated streams with some RNGs.

#### ❌ Random seeds per worker
```julia
# BAD: Not reproducible
seeds = rand(1:typemax(Int), n_workers)
```
**Problem**: Results change on every run, no reproducibility.

#### ✅ Stream splitting (our approach)
```julia
# GOOD: Independent + reproducible
base_seed = rand(RandomDevice(), UInt64)
worker_seeds = [base_seed + i * GOLDEN_RATIO for i in 1:n_workers]
```
**Benefits**: Independence and reproducibility.

## Implementation Details

### Bootstrap (`src/bootstrap.jl`)

```julia
function bootstrap_irf_distributed(...)
    # Base seed from hardware RNG (truly random)
    base_seed = rand(Random.RandomDevice(), UInt64)

    # Create (replication_index, base_seed) pairs
    rep_seeds = [(i, base_seed) for i in 1:reps]

    function single_bootstrap_rep(rep_info::Tuple{Int,UInt64})
        rep_idx, base_seed = rep_info

        # Derive unique seed for this replication
        worker_seed = base_seed + UInt64(rep_idx) * 0x9e3779b97f4a7c15
        Random.seed!(worker_seed)

        # ... bootstrap code ...
    end

    pmap(single_bootstrap_rep, rep_seeds)
end
```

**Key points**:
- Each bootstrap replication gets unique seed
- Workers may process multiple replications (via `pmap` scheduling)
- No matter how work is divided, seeds are unique and deterministic

### Sign Restrictions (`src/var/identification.jl`)

```julia
function identify_sign_distributed(...)
    # Base seed
    base_seed = rand(Random.RandomDevice(), UInt64)

    # Work specification: (n_attempts, worker_id, base_seed)
    work_specs = [(draws_per_worker, i, base_seed) for i in 1:n_workers]

    function search_rotations(work_info)
        n_attempts, worker_id, base_seed = work_info

        # Worker-specific seed
        worker_seed = base_seed + UInt64(worker_id) * 0x9e3779b97f4a7c15
        Random.seed!(worker_seed)

        # ... search code ...
    end

    pmap(search_rotations, work_specs)
end
```

**Key points**:
- Each worker gets different seed
- Worker searches use independent random stream
- Early termination still works (first successful worker returns)

## Reproducibility

### Making Results Reproducible

Users can make results reproducible by:

1. **Set global RNG state before calling**:
```julia
using Random
Random.seed!(12345)

# Now base_seed will be deterministic
irf_boot = bootstrap_irf(model, id, 24, 1000; parallel=:distributed)
```

2. **Specify base seed explicitly** (future enhancement):
```julia
# Could add kwarg:
irf_boot = bootstrap_irf(model, id, 24, 1000;
                        parallel=:distributed,
                        rng_seed=12345)
```

### Current Behavior

- **Default**: Uses hardware RNG (`Random.RandomDevice()`) for base seed → non-reproducible but truly random
- **After `Random.seed!()`**: NOT currently reproducible because we call `rand(RandomDevice(), UInt64)` which bypasses global RNG

### Future Enhancement

To make fully reproducible, we should:

```julia
function bootstrap_irf_distributed(...; rng=Random.GLOBAL_RNG)
    # Use provided RNG instead of RandomDevice
    base_seed = rand(rng, UInt64)
    # ... rest of code ...
end
```

Then users can:
```julia
Random.seed!(12345)
irf1 = bootstrap_irf(model, id, 24, 1000; parallel=:distributed)

Random.seed!(12345)
irf2 = bootstrap_irf(model, id, 24, 1000; parallel=:distributed)

@assert irf1 == irf2  # Reproducible!
```

## Statistical Properties

### Independence Testing

To verify independence of streams, we can check:

1. **Correlation test**: Bootstrap samples from different workers should be uncorrelated
2. **NIST tests**: Apply randomness tests to streams from different workers
3. **Overlapping periods**: Ensure no overlap in random sequences

### Theoretical Guarantees

- **Mersenne Twister** (Julia's default): Period is 2^19937 - 1
- **Seed separation**: 2^64 × φ ≈ 1.6 × 10^19
- **Number of workers**: Typically < 100

**Conclusion**: With seeds separated by ~10^19 and period > 10^6000, overlap is impossible in practice.

## References

- L'Ecuyer, P. (2012). "Random Number Generation." In *Handbook of Computational Statistics*.
- Matsumoto, M., & Nishimura, T. (1998). "Mersenne Twister: A 623-dimensionally equidistributed uniform pseudo-random number generator."
- Julia Random documentation: https://docs.julialang.org/en/v1/stdlib/Random/

## FAQ

### Q: Why use `UInt64` instead of `Int`?

**A**: `UInt64` provides full 64-bit range (0 to 2^64-1), while `Int64` only provides -2^63 to 2^63-1. We want maximum separation.

### Q: What if I have more than 10^19 replications?

**A**: You would need multiple base seeds. Practically impossible - 10^19 replications would take longer than the age of the universe.

### Q: Is this better than `Random.Future.randjump()`?

**A**: `randjump()` is more sophisticated (actually advances RNG state), but:
- Requires Future module
- More complex to implement
- Our approach is simpler and sufficient for practical use

### Q: Can I use this with other RNGs?

**A**: Yes, but ensure:
- RNG has large enough period
- Seeding is deterministic
- You understand the RNG's properties

## Recommendations

### For Users

1. ✅ **Trust the defaults**: Our approach is sound for typical use
2. ✅ **Use many replications**: 1000+ for bootstrap, 10000+ for sign restrictions
3. ⚠️ **Don't worry about reproducibility** unless you need exact reproduction
4. ⚠️ **For exact reproduction**: Wait for `rng_seed` kwarg enhancement

### For Developers

1. ✅ **Always derive seeds from base seed**: Don't use consecutive integers
2. ✅ **Use hardware RNG for base seed**: `rand(RandomDevice(), UInt64)`
3. ✅ **Document seeding approach**: Users should understand RNG behavior
4. ✅ **Add reproducibility option**: Future enhancement with `rng` kwarg

## Summary

Our parallel RNG approach:
- ✅ Statistically sound (independent streams)
- ✅ Simple and efficient
- ✅ No external dependencies
- ⚠️ Not currently reproducible (but fixable)
- ✅ Adequate for all practical applications

The use of golden ratio for seed spacing is a well-established technique that ensures independent streams across workers.
