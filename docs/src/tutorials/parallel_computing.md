# Parallel Computing

The only operation that can run on multiple processes is the rotation search for sign-restriction identification. Each candidate rotation is drawn independently, so the search can be split across Distributed.jl workers.

Bootstrap inference runs serially:

```julia
result = irf(var_model, CholeskyID(); horizon = 24, inference = WildBootstrap(1000))
```

`NarrativeRestriction` supports only `parallel = :none`.

## Setup

Add worker processes and load the package on every worker:

```julia
using Distributed

addprocs(4)
@everywhere using MacroEconometricTools

println("Workers: ", workers())
```

The package must be loaded with `@everywhere`; otherwise workers raise `UndefVarError: MacroEconometricTools not defined`. Calling `using MacroEconometricTools` on the main process before `addprocs` avoids precompiling on each worker.

## Sign Restrictions

```julia
using MacroEconometricTools

var_model = fit(OLSVAR, Y, 24; names)

# Rows: variables. Columns: shocks. Restrictions apply at horizons 0:12.
restrictions = [
    -1  1  0
     0  1  0
     1  1  1
]
id_sign = SignRestriction(restrictions, 12)
```

Draw a single rotation matrix, serially or with the search distributed over the workers:

```julia
P = rotation_matrix(var_model, id_sign; max_draws = 50_000, parallel = :none)
P = rotation_matrix(var_model, id_sign; max_draws = 50_000, parallel = :distributed)
```

With `verbose = true`, the search prints progress. `rng` sets the random number generator.

To compute set-identified impulse responses, pass `parallel` to `irf`. It applies to the rotation search only:

```julia
result = irf(var_model, id_sign;
             horizon = 48,
             n_draws = 1000,
             max_attempts = 50_000,
             parallel = :distributed)
```

Here `max_attempts` is the maximum number of rotation draws per accepted draw.

Distribution has overhead, so it helps when the acceptance rate is low (strict restrictions, many variables, long restriction horizons). With loose restrictions a valid rotation is usually found within a few draws and the serial search is faster.

## Cleanup

```julia
rmprocs(workers())
```

## References

- [Julia Distributed Computing documentation](https://docs.julialang.org/en/v1/manual/distributed-computing/)
