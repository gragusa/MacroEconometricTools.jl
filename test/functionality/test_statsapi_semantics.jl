using MacroEconometricTools
using MacroEconometricTools: StatsBase
using StableRNGs: StableRNG
using Test

@testset "StatsAPI and Base semantics" begin
    Y = randn(StableRNG(11), 120, 3)
    model = fit(OLSVAR, Y, 2)

    @testset "generics are the StatsAPI functions" begin
        @test coef === StatsBase.coef
        @test vcov === StatsBase.vcov
        @test loglikelihood === StatsBase.loglikelihood
        @test aic === StatsBase.aic
        @test bic === StatsBase.bic
    end

    @testset "VARModel is not a collection" begin
        @test_throws MethodError size(model)
        @test (StatsBase.nobs(model), n_lags(model), n_vars(model)) == (118, 2, 3)
    end

    @testset "deprecated names" begin
        ll = @test_deprecated log_likelihood(model)
        @test ll == loglikelihood(model)
        P = [2.0 0.0 0.0; 1.0 0.5 0.0; 0.3 0.2 4.0]
        Pn = @test_deprecated MacroEconometricTools.normalize(P, UnitEffect())
        @test Pn == normalize_impact(P, UnitEffect())
        Q = copy(P)
        @test_deprecated MacroEconometricTools.normalize!(Q, UnitEffect())
        @test Q == Pn
        # The deprecated names are not exported, so `normalize` from LinearAlgebra
        # stays unambiguous under `using LinearAlgebra, MacroEconometricTools`.
        @test !Base.isexported(MacroEconometricTools, :normalize)
        @test !Base.isexported(MacroEconometricTools, :normalize!)
    end

    @testset "normalize_impact" begin
        P = [2.0 0.0; 1.0 0.5]
        @test normalize_impact(P, UnitStd()) == P
        @test normalize_impact(P, UnitEffect()) == [1.0 0.0; 0.5 1.0]
        @test P == [2.0 0.0; 1.0 0.5]
        Q = copy(P)
        @test normalize_impact!(Q, UnitEffect()) === Q
        @test Q == [1.0 0.0; 0.5 1.0]
        @test normalize_impact(Float32.(P), UnitEffect()) == Float32[1 0; 0.5 1]
        V = view(copy(P), :, :)
        normalize_impact!(V, UnitEffect())
        @test V == [1.0 0.0; 0.5 1.0]
    end

    @testset "specification equality and hashing" begin
        cases = [
            (ZeroConstraint(:a, [:b], [1, 2]), ZeroConstraint(:a, [:b], [1, 2]), ZeroConstraint(:a, [:b], [1])),
            (FixedConstraint(:a, :b, 1, 0.5), FixedConstraint(:a, :b, 1, 0.5), FixedConstraint(:a, :b, 1, 0.25)),
            (BlockExogeneity([:a], [:b, :c]), BlockExogeneity([:a], [:b, :c]), BlockExogeneity([:a], [:c])),
            (CholeskyID([2, 1, 3]), CholeskyID([2, 1, 3]), CholeskyID([1, 2, 3])),
            (CholeskyID([:y, :x]), CholeskyID([:y, :x]), CholeskyID([:x, :y])),
            (SignRestriction([1 0; 0 -1], 3), SignRestriction([1 0; 0 -1], 3), SignRestriction([1 0; 0 -1], 2)),
            (NarrativeShockRestriction(1, 50, 1), NarrativeShockRestriction(1, 50, 1), NarrativeShockRestriction(1, 50, -1)),
            (
                NarrativeRestriction([1 0; 0 1], [NarrativeShockRestriction(1, 50, 1)], 2),
                NarrativeRestriction([1 0; 0 1], [NarrativeShockRestriction(1, 50, 1)], 2),
                NarrativeRestriction([1 0; 0 1], [NarrativeShockRestriction(1, 51, 1)], 2),
            ),
        ]
        for (a, b, c) in cases
            @test a == b
            @test isequal(a, b)
            @test hash(a) == hash(b)
            @test a != c
            @test !isequal(a, c)
        end
        @test CholeskyID() == CholeskyID()
        @test CholeskyID([1, 2]) != CholeskyID([:a, :b])
        @test ZeroConstraint(:a, [:b]) != BlockExogeneity([:a], [:b])
        @test length(Set([CholeskyID([2, 1]), CholeskyID([2, 1]), CholeskyID([1, 2])])) == 2
    end
end
