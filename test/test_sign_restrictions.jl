# ============================================================================
# Sign Restriction Identification and RNG Reproducibility Tests
# ============================================================================
#
# These tests verify that:
# 1. Sign restriction identification is reproducible with fixed RNG
# 2. Serial and distributed search produce reproducible results
# 3. Found matrices satisfy the imposed restrictions
# 4. P*P' ≈ Σ holds for identified matrices
#

using MacroEconometricTools
using Test
using LinearAlgebra
using Random
using StableRNGs: StableRNG
using Distributed

@testset "Sign Restriction Identification" begin

    # Generate simple test data
    Random.seed!(456)
    T = 100
    n_v = 3
    n_l = 2

    # Generate AR(2) data
    Y = randn(T, n_v)
    for t in 3:T
        Y[t, :] = 0.5 * Y[t-1, :] + 0.3 * Y[t-2, :] + 0.1 * randn(n_v)
    end

    # Estimate VAR
    var = estimate(OLSVAR, Y, n_l)

    @testset "Serial Sign Restriction Reproducibility" begin
        # Simple sign restriction: first shock has positive impact on first variable
        restrictions = zeros(Int, n_v, n_v)
        restrictions[1, 1] = 1  # Positive impact

        id = SignRestriction(restrictions, 0)

        # Test 1: Same seed → same results (use rotation_matrix directly for kwargs)
        P1 = rotation_matrix(var, id; max_draws=1000, parallel=:none, rng=StableRNG(42))
        P2 = rotation_matrix(var, id; max_draws=1000, parallel=:none, rng=StableRNG(42))

        @test P1 ≈ P2
        @test size(P1) == (n_v, n_v)

        # Test 2: Different seeds → different results (set identification)
        P3 = rotation_matrix(var, id; max_draws=1000, parallel=:none, rng=StableRNG(123))
        @test !(P1 ≈ P3)

        # Test 3: Verify restrictions are satisfied
        @test P1[1, 1] > 0
        @test P3[1, 1] > 0

        # Test 4: Verify P*P' ≈ Σ
        Σ = Matrix(vcov(var))
        @test P1 * P1' ≈ Σ rtol=1e-10
        @test P3 * P3' ≈ Σ rtol=1e-10
    end

    @testset "Distributed Sign Restriction Reproducibility" begin
        if Distributed.nworkers() == 1
            @info "Skipping distributed sign restriction tests (no workers available)"
        else
            # Simple sign restriction
            restrictions = zeros(Int, n_v, n_v)
            restrictions[1, 1] = 1  # Positive

            id = SignRestriction(restrictions, 0)

            # Test 1: Same seed → same results (distributed)
            P1_dist = rotation_matrix(var, id; max_draws=1000, parallel=:distributed, rng=StableRNG(42))
            P2_dist = rotation_matrix(var, id; max_draws=1000, parallel=:distributed, rng=StableRNG(42))

            @test P1_dist ≈ P2_dist
            @test size(P1_dist) == (n_v, n_v)

            # Test 2: Verify restrictions satisfied
            @test P1_dist[1, 1] > 0

            # Test 3: Verify P*P' ≈ Σ
            Σ = Matrix(vcov(var))
            @test P1_dist * P1_dist' ≈ Σ rtol=1e-10
        end
    end

    @testset "Multiple Sign Restrictions" begin
        # More complex restrictions
        restrictions = zeros(Int, n_v, n_v)
        restrictions[1, 1] = 1   # Shock 1: positive on var 1
        restrictions[2, 1] = -1  # Shock 1: negative on var 2
        restrictions[2, 2] = 1   # Shock 2: positive on var 2

        id = SignRestriction(restrictions, 0)

        # Find rotation satisfying restrictions
        P = rotation_matrix(var, id; max_draws=5000, parallel=:none, rng=StableRNG(999))

        # Verify all restrictions
        @test P[1, 1] > 0
        @test P[2, 1] < 0
        @test P[2, 2] > 0

        # Verify P*P' ≈ Σ
        Σ = Matrix(vcov(var))
        @test P * P' ≈ Σ rtol=1e-10
    end

    # Note: Sign restrictions on IRF horizons > 0 would require passing parameters
    # to the identify function during IRF computation, which is not currently supported
    # in the API. For now we test only impact restrictions.
    @testset "Sign Restrictions on IRFs (horizon > 0) - Basic" begin
        # Restrictions on IRF responses (not just impact)
        restrictions = zeros(Int, n_v, n_v)
        restrictions[1, 1] = 1  # Positive response for horizons 0-2

        id = SignRestriction(restrictions, 2)

        # Find rotation satisfying restrictions on IRFs
        P = rotation_matrix(var, id; max_draws=5000, parallel=:none, rng=StableRNG(777))

        # Verify P*P' ≈ Σ
        Σ = Matrix(vcov(var))
        @test P * P' ≈ Σ rtol=1e-10

        # Basic check: impact should satisfy restrictions
        @test P[1, 1] > 0
    end

    @testset "IRF Computation with Sign Restrictions" begin
        # Test that IRF computation with sign restrictions works and is reproducible
        restrictions = zeros(Int, n_v, n_v)
        restrictions[1, 1] = 1

        id = SignRestriction(restrictions, 0)

        horizon = 10
        n_draws = 50  # Small number for testing

        # Test with sign restriction-specific method (returns SignRestrictedIRFResult)
        irf_result = irf(var, id; horizon=horizon, n_draws=n_draws, rng=StableRNG(333))

        @test irf_result isa SignRestrictedIRFResult
        @test size(irf_result.irf_median) == (horizon+1, n_v, n_v)
        @test size(irf_result.irf_draws) == (n_draws, horizon+1, n_v, n_v)
        @test irf_result.irf_median[1, 1, 1] > 0  # Median should satisfy restriction
        @test all(irf_result.irf_draws[:, 1, 1, 1] .> 0)  # All draws should satisfy restriction

        # Test reproducibility
        irf_result2 = irf(var, id; horizon=horizon, n_draws=n_draws, rng=StableRNG(333))
        @test irf_result.irf_median ≈ irf_result2.irf_median
        @test irf_result.irf_draws ≈ irf_result2.irf_draws
    end

    println("✓ All sign restriction tests passed!")
end
