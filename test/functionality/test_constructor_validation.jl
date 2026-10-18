using Test
using MacroEconometricTools
using MacroEconometricTools: bootstrap_irf
using StableRNGs
using Dates
using Random

@testset "Identification constructors validate their arguments" begin
    @testset "CholeskyID" begin
        @test CholeskyID().ordering === nothing
        @test CholeskyID(nothing).ordering === nothing
        @test CholeskyID([2, 1]).ordering == [2, 1]
        @test CholeskyID(Int32[2, 1]).ordering isa Vector{Int}
        @test CholeskyID([:b, :a]).ordering == [:b, :a]
        @test_throws "must be a permutation of 1:2, got [1, 1]" CholeskyID([1, 1])
        @test_throws "must be a permutation of 1:2, got [0, 1]" CholeskyID([0, 1])
        @test_throws "must be a permutation of 1:2, got [1, 3]" CholeskyID([1, 3])
        @test_throws "repeated variable names" CholeskyID([:a, :a])
        @test_throws "must not be empty" CholeskyID(Int[])
        @test_throws "must not be empty" CholeskyID(Symbol[])
        @test_throws "CholeskyID ordering must be `nothing`" CholeskyID(["a", "b"])
        @test_throws "CholeskyID ordering must be `nothing`" CholeskyID(1)
    end

    @testset "SignRestriction" begin
        id = SignRestriction([1 0; -1 1])
        @test id.restrictions == [1 0; -1 1] && id.horizon == 0
        @test SignRestriction([(+) 0; (-) (+)], 3).restrictions == [1 0; -1 1]
        @test SignRestriction(Int32[1 0; 0 -1], 2).restrictions isa Matrix{Int}
        R = [1 0; 0 1]
        @test SignRestriction(view(R, :, :)).restrictions == R
        @test SignRestriction(R).restrictions !== R
        @test_throws "values must be -1, 0, or +1, got 2" SignRestriction([2 0; 0 0], 0)
        @test_throws "Invalid sign restriction element" SignRestriction([1.0 0; 0 0])
        @test_throws "must be square (variables × shocks), got size (2, 3)" SignRestriction(
            [1 0 0; 0 1 0]
        )
        @test_throws "horizon must be ≥ 0, got -1" SignRestriction([1 0; 0 1], -1)
    end

    @testset "NarrativeRestriction" begin
        shocks = [NarrativeShockRestriction(1, 10, 1)]
        R = [1 0; 0 0]
        for nr in (
                NarrativeRestriction(R, shocks, 2),
                NarrativeRestriction(R, shocks; horizon = 2),
            )
            @test nr isa NarrativeRestriction{Int}
            @test nr.sign_restrictions == R && nr.horizon == 2
            @test nr.narrative_shocks == shocks
        end
        @test NarrativeRestriction([(+) 0; 0 0], shocks).sign_restrictions == R
        @test NarrativeRestriction(shocks, 3).sign_restrictions == zeros(Int, 3, 3)

        # Every construction path validates.
        @test_throws "values must be -1, 0, or +1, got 2" NarrativeRestriction(
            [2 0; 0 0], shocks, 0
        )
        @test_throws "values must be -1, 0, or +1, got 2" NarrativeRestriction{Int}(
            [2 0; 0 0], shocks, 0
        )
        @test_throws "must be square" NarrativeRestriction([1 0 0; 0 1 0], shocks)
        @test_throws "horizon must be ≥ 0, got -1" NarrativeRestriction(R, shocks, -1)
        @test_throws "horizon must be ≥ 0, got -1" NarrativeRestriction(
            shocks, 2; horizon = -1
        )
        far = [NarrativeShockRestriction(3, 10, 1)]
        @test_throws "narrative restriction on shock 3, but the model has 2 shocks" NarrativeRestriction(
            R, far
        )
        @test_throws "narrative restriction on shock 3" NarrativeRestriction(far, 2)

        # Vectors with an abstract element type.
        empty = NarrativeRestriction(R, NarrativeShockRestriction[])
        @test isempty(empty.narrative_shocks)
        @test empty.sign_restrictions == R
        mixed = NarrativeShockRestriction[NarrativeShockRestriction(1, Date(2001, 1, 1), 1)]
        @test NarrativeRestriction(R, mixed) isa NarrativeRestriction{Date}
    end

    @testset "Empty narrative vector behaves like a sign restriction" begin
        rng = StableRNG(11)
        Y = randn(rng, 120, 2)
        model = fit(OLSVAR, Y, 1)
        R = [1 0; 0 0]
        s = irf(model, SignRestriction(R); horizon = 3, n_draws = 5, rng = StableRNG(3))
        n = irf(
            model, NarrativeRestriction(R, NarrativeShockRestriction[]);
            horizon = 3, n_draws = 5, rng = StableRNG(3)
        )
        @test s.irf_draws == n.irf_draws
    end
end

@testset "Bad input raises ArgumentError" begin
    rng = StableRNG(5)
    Y = randn(rng, 80, 2)
    for t in 2:80
        Y[t, :] .+= 0.4 .* Y[t - 1, :]
    end
    model = fit(OLSVAR, Y, 1; names = [:a, :b])
    n_res = size(model.residuals, 1)

    @testset "Set-identified irf rejects inference" begin
        sign_id = SignRestriction([1 0; 0 0])
        narr_id = NarrativeRestriction([1 0; 0 0], NarrativeShockRestriction{Int}[])
        for id in (sign_id, narr_id)
            @test_throws "is set-identifying and takes no `inference` keyword" irf(
                model, id; inference = WildBootstrap(10), n_draws = 2
            )
        end
    end

    @testset "Unknown shock name in rescale" begin
        r = irf(
            model, CholeskyID(); horizon = 4,
            inference = WildBootstrap(20; save_draws = true), rng = StableRNG(1)
        )
        @test_throws ArgumentError rescale(r, :c => 2.0)
        @test_throws "Shock :c not found" rescale(r, :c => 2.0)
    end

    @testset "Cumulation without draws" begin
        @test_throws ArgumentError irf(model, CholeskyID(); horizon = 4, cumulate = [:a])
        @test_throws "Cumulation requires bootstrap draws" irf(
            model, CholeskyID(); horizon = 4, cumulate = [:a]
        )
    end

    @testset "Narrative date not in the model dates" begin
        d = collect(Date(2000, 1, 1):Month(1):(Date(2000, 1, 1) + Month(79)))
        dmodel = fit(OLSVAR, Y, 1; names = [:a, :b], dates = d)
        id = NarrativeRestriction(
            [1 0; 0 0], [NarrativeShockRestriction(1, Date(1990, 1, 1), 1)]
        )
        @test_throws ArgumentError irf(dmodel, id; horizon = 3, n_draws = 2)
        @test_throws "date 1990-01-01 not found in the model dates (2000-02-01 to" irf(
            dmodel, id; horizon = 3, n_draws = 2
        )
    end

    @testset "Block length longer than the sample" begin
        @test_throws "block_length = $(n_res + 1) must lie in 1:$n_res" irf(
            model, CholeskyID(); horizon = 4,
            inference = BlockBootstrap(5; block_length = n_res + 1), rng = StableRNG(1)
        )
        @test_throws "block_length = $(n_res + 1) must lie in 1:$n_res" bootstrap_irf(
            model, CholeskyID(), BlockBootstrap(5; block_length = n_res + 1);
            horizon = 4, rng = StableRNG(1)
        )
        Z = randn(StableRNG(2), n_res)
        @test_throws "block_length = $(n_res + 1) must lie in 1:$n_res" irf(
            model, IVIdentification(Z, 1); horizon = 4,
            inference = ProxySVARMBB(5; block_length = n_res + 1), rng = StableRNG(1)
        )
    end
end
