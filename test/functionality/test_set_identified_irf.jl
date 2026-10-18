using Test
using MacroEconometricTools
using StableRNGs

@testset "Set-identified irf: shared implementation and validation" begin
    rng = StableRNG(2024)
    Y = randn(rng, 200, 2)
    for t in 2:200
        Y[t, :] .+= 0.5 .* Y[t - 1, :]
    end
    model = fit(OLSVAR, Y, 1; names = [:a, :b])
    R = [1 0; 0 0]
    sign_id = SignRestriction(R, 0)
    narrative_id = NarrativeRestriction(R, NarrativeShockRestriction{Int}[]; horizon = 0)

    @testset "Sign and narrative paths agree without narrative constraints" begin
        kw = (; horizon = 4, n_draws = 20, coverage = [0.9, 0.68], scale = 2, cumulate = [:a])
        s = irf(model, sign_id; rng = StableRNG(7), kw...)
        n = irf(model, narrative_id; rng = StableRNG(7), kw...)
        @test s.rotation_matrices == n.rotation_matrices
        @test s.irf_draws == n.irf_draws
        @test s.lower == n.lower && s.upper == n.upper
        @test s.coverage == n.coverage == [0.68, 0.9]
        @test n.identification === narrative_id
        @test n.metadata.cumulated_vars == [:a]
    end

    @testset "Validation shared with the generic irf" begin
        for id in (sign_id, narrative_id)
            @test_throws "horizon must be positive" irf(model, id; horizon = 0, n_draws = 2)
            @test_throws "coverage levels must be in (0, 1)" irf(
                model, id; coverage = [0.9, 1.0], n_draws = 2
            )
            @test_throws "n_draws must be positive" irf(model, id; n_draws = 0)
            @test_throws "max_attempts must be positive" irf(
                model, id; n_draws = 2, max_attempts = 0
            )
        end
        @test_throws "horizon must be positive" irf(model, CholeskyID(); horizon = 0)
    end

    @testset "parallel" begin
        @test_throws "not supported for NarrativeRestriction" irf(
            model, narrative_id; n_draws = 2, parallel = :distributed
        )
        @test_throws "parallel must be :none or :distributed" irf(
            model, sign_id; n_draws = 2, parallel = :threads
        )
    end
end
