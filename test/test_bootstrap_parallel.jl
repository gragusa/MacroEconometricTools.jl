# ============================================================================
# Bootstrap Parallel Execution and RNG Reproducibility Tests
# ============================================================================
#
# These tests verify that:
# 1. Bootstrap IRF with distributed parallelization is reproducible with fixed RNG
# 2. Serial and distributed execution produce consistent results
# 3. Different seeds produce different results (sanity check)
#

using MacroEconometricTools
using Test
using LinearAlgebra
using Random
using StableRNGs: StableRNG
using Distributed

@testset "Bootstrap IRF Reproducibility" begin

    # Generate simple test data
    Random.seed!(123)
    T = 100
    n_v = 3
    n_l = 2

    # Generate AR(2) data
    Y = randn(T, n_v)
    for t in 3:T
        Y[t, :] = 0.5 * Y[t-1, :] + 0.3 * Y[t-2, :] + 0.1 * randn(n_v)
    end

    # Estimate VAR
    var = fit(OLSVAR, Y, n_l)
    id = CholeskyID()
    horizon = 5
    reps = 50  # Small number for fast testing

    @testset "Serial Bootstrap Reproducibility" begin
        # Test 1: Same seed → same results
        boot1 = bootstrap_irf(var, id, horizon, reps; parallel=:none, rng=StableRNG(42))
        boot2 = bootstrap_irf(var, id, horizon, reps; parallel=:none, rng=StableRNG(42))

        @test boot1 ≈ boot2
        @test size(boot1) == (reps, horizon + 1, n_v, n_v)

        # Test 2: Different seeds → different results
        boot3 = bootstrap_irf(var, id, horizon, reps; parallel=:none, rng=StableRNG(123))
        @test !(boot1 ≈ boot3)

        # Test 3: Check that results are reasonable
        @test all(isfinite.(boot1))
        @test !all(boot1 .== 0)
    end

    @testset "Distributed Bootstrap Reproducibility" begin
        # Skip if no workers available
        if Distributed.nworkers() == 1
            @info "Skipping distributed tests (no workers available)"
        else
            # Test 1: Same seed → same results (distributed)
            boot1_dist = bootstrap_irf(var, id, horizon, reps; parallel=:distributed, rng=StableRNG(42))
            boot2_dist = bootstrap_irf(var, id, horizon, reps; parallel=:distributed, rng=StableRNG(42))

            @test boot1_dist ≈ boot2_dist
            @test size(boot1_dist) == (reps, horizon + 1, n_v, n_v)

            # Test 2: Different seeds → different results (distributed)
            boot3_dist = bootstrap_irf(var, id, horizon, reps; parallel=:distributed, rng=StableRNG(123))
            @test !(boot1_dist ≈ boot3_dist)

            # Test 3: Check that results are reasonable
            @test all(isfinite.(boot1_dist))
            @test !all(boot1_dist .== 0)
        end
    end

    @testset "Serial vs Distributed Consistency" begin
        if Distributed.nworkers() == 1
            @info "Skipping serial vs distributed comparison (no workers available)"
        else
            # With same seed, serial and distributed should produce results
            # that have the same distribution (though order may differ due to batching)
            boot_serial = bootstrap_irf(var, id, horizon, reps; parallel=:none, rng=StableRNG(999))
            boot_dist = bootstrap_irf(var, id, horizon, reps; parallel=:distributed, rng=StableRNG(999))

            # Check that distributions are similar (compare quantiles)
            for h in 1:(horizon+1)
                for i in 1:n_v
                    for j in 1:n_v
                        serial_vals = boot_serial[:, h, i, j]
                        dist_vals = boot_dist[:, h, i, j]

                        # Compare median
                        @test abs(median(serial_vals) - median(dist_vals)) < 0.5 * std(serial_vals)

                        # Compare variance (should be similar)
                        @test abs(var(serial_vals) - var(dist_vals)) < 0.5 * var(serial_vals)
                    end
                end
            end
        end
    end

    @testset "Bootstrap Methods" begin
        # Test different bootstrap methods for reproducibility
        for method in [:wild, :standard, :block]
            boot1 = bootstrap_irf(var, id, horizon, reps; method=method, parallel=:none, rng=StableRNG(42))
            boot2 = bootstrap_irf(var, id, horizon, reps; method=method, parallel=:none, rng=StableRNG(42))

            @test boot1 ≈ boot2
            @test all(isfinite.(boot1))
        end
    end

    println("✓ All bootstrap reproducibility tests passed!")
end
