# ============================================================================
# bootstrap_irf(model, identification, inference; horizon, normalization, rng)
# ============================================================================

using MacroEconometricTools
using Test
using Random
using StableRNGs: StableRNG

@testset "bootstrap_irf entry point" begin
    rng = StableRNG(123)
    T, n_v, n_l = 100, 3, 2
    Y = randn(rng, T, n_v)
    for t in 3:T
        Y[t, :] = 0.5 * Y[t - 1, :] + 0.3 * Y[t - 2, :] + 0.1 * randn(rng, n_v)
    end
    model = fit(OLSVAR, Y, n_l)
    id = CholeskyID()
    horizon, reps = 5, 30
    methods = (WildBootstrap(reps), Bootstrap(reps), BlockBootstrap(reps; block_length = 6))

    @testset "Reproducibility: $(nameof(typeof(inf)))" for inf in methods
        boot1 = bootstrap_irf(model, id, inf; horizon, rng = StableRNG(42))
        boot2 = bootstrap_irf(model, id, inf; horizon, rng = StableRNG(42))
        boot3 = bootstrap_irf(model, id, inf; horizon, rng = StableRNG(43))
        @test boot1 == boot2
        @test boot1 != boot3
        @test size(boot1) == (reps, n_v, n_v, horizon + 1)
        @test all(isfinite, boot1)
    end

    @testset "Same draws as irf: $(nameof(typeof(inf)))" for inf in (
            WildBootstrap(reps; save_draws = true), Bootstrap(reps; save_draws = true),
            BlockBootstrap(reps; block_length = 6, save_draws = true),
        )
        for id_ in (id, CholeskyID([3, 1, 2]))
            draws = bootstrap_irf(model, id_, inf; horizon, rng = StableRNG(7))
            result = irf(model, id_; horizon, inference = inf, rng = StableRNG(7))
            @test Array(get_draws(result)) == draws
        end
    end

    @testset "normalization" begin
        draws = bootstrap_irf(
            model, id, WildBootstrap(reps); horizon, normalization = UnitEffect(),
            rng = StableRNG(1)
        )
        @test all(draws[:, j, j, 1] ≈ ones(reps) for j in 1:n_v)
    end

    @testset "Argument errors" begin
        @test_throws "horizon must be positive" bootstrap_irf(
            model, id, WildBootstrap(reps); horizon = 0
        )
        @test_throws "bootstrap_irf supports WildBootstrap, Bootstrap, and BlockBootstrap (got Analytic)" bootstrap_irf(
            model, id, Analytic()
        )
        @test_throws "(got ProxySVARMBB)" bootstrap_irf(model, id, ProxySVARMBB(10))
        Z = randn(StableRNG(2), size(residuals(model), 1))
        for inf in methods
            @test_throws "$(nameof(typeof(inf))) does not resample the instrument" bootstrap_irf(
                model, IVIdentification(Z, 1), inf; horizon
            )
        end
        # The bootstrap has no parallel option.
        @test_throws MethodError bootstrap_irf(
            model, id, WildBootstrap(reps); horizon, parallel = :distributed
        )
    end

    @testset "Deprecated forms forward to the entry point" begin
        new_wild = bootstrap_irf(model, id, WildBootstrap(reps); horizon, rng = StableRNG(5))
        new_std = bootstrap_irf(model, id, Bootstrap(reps); horizon, rng = StableRNG(5))
        new_block = bootstrap_irf(
            model, id, BlockBootstrap(reps; block_length = 6); horizon, rng = StableRNG(5)
        )
        @test (
            @test_deprecated bootstrap_irf(
                model, id, horizon, reps; rng = StableRNG(5)
            )
        ) == new_wild
        @test (
            @test_deprecated bootstrap_irf(
                model, id, horizon, reps; method = :standard, rng = StableRNG(5)
            )
        ) == new_std
        @test (
            @test_deprecated bootstrap_irf(
                model, id, horizon, reps; method = :block, block_length = 6, rng = StableRNG(5)
            )
        ) == new_block
        @test (
            @test_deprecated bootstrap_irf_wild(
                model, id, horizon, reps, StableRNG(5)
            )
        ) == new_wild
        @test (
            @test_deprecated bootstrap_irf_standard(
                model, id, horizon, reps, StableRNG(5)
            )
        ) == new_std
        @test (
            @test_deprecated bootstrap_irf_block(
                model, id, horizon, reps, 6, StableRNG(5)
            )
        ) == new_block
        @test_throws "method must be :wild, :standard, or :block" bootstrap_irf(
            model, id, horizon, reps; method = :jackknife
        )
    end
end
