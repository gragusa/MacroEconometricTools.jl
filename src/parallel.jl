# ============================================================================
# Parallel Computing Utilities
# ============================================================================

"""
    setup_workers(n_workers::Int)

Add worker processes for parallel computation.

# Arguments
- `n_workers::Int`: Number of workers to add

# Returns
- Vector of worker process IDs

# Examples
```julia
using Distributed
workers = setup_workers(4)  # Add 4 worker processes
```
"""
function setup_workers(n_workers::Int)
    using Distributed

    current_workers = nworkers()
    if current_workers < n_workers
        addprocs(n_workers - current_workers)
    end

    # Load package on all workers
    @everywhere using MacroEconometricTools

    return workers()
end

"""
    check_parallel_available()

Check if Distributed package is loaded and workers are available.

# Returns
- `(available::Bool, n_workers::Int)`
"""
function check_parallel_available()
    if isdefined(Main, :Distributed)
        using Distributed
        return (true, nworkers())
    else
        return (false, 1)
    end
end

"""
    pmap_batched(f, collection; batch_size=nothing)

Parallel map with batching for better load balancing.

# Arguments
- `f`: Function to apply
- `collection`: Collection to map over
- `batch_size::Union{Int,Nothing}`: Batch size (default: auto)

# Returns
- Vector of results
"""
function pmap_batched(f, collection; batch_size=nothing)
    using Distributed

    n = length(collection)
    n_w = nworkers()

    if isnothing(batch_size)
        # Auto batch size: aim for ~10 batches per worker
        batch_size = max(1, div(n, 10 * n_w))
    end

    # Create batches
    batches = [collection[i:min(i+batch_size-1, n)]
               for i in 1:batch_size:n]

    # Process batches in parallel
    results = pmap(batches) do batch
        [f(item) for item in batch]
    end

    # Flatten results
    return vcat(results...)
end
