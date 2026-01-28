# Tests for Hub Types (BayesianIRFResult, LocalProjectionIRFResult, NarrativeRestriction)
# These are the shared types for the hub-and-spoke architecture

using MacroEconometricTools
using Test
using LinearAlgebra
using Statistics
using AxisArrays

@testset "Hub Types (IRF Results)" begin
    @testset "BayesianIRFResult construction" begin
        n_draws = 100
        n_vars = 3
        n_shocks = 3
        n_horizons = 11

        # Create mock IRF data as AxisArray
        data = AxisArray(
            randn(n_draws, n_vars, n_shocks, n_horizons),
            Axis{:draw}(1:n_draws),
            Axis{:variable}([:GDP, :INF, :RATE]),
            Axis{:shock}([:GDP, :INF, :RATE]),
            Axis{:horizon}(0:10)
        )

        # Create lower/upper bounds
        coverage = [0.68, 0.90]
        lower = [AxisArray(
                     randn(n_vars, n_shocks, n_horizons),
                     Axis{:variable}([:GDP, :INF, :RATE]),
                     Axis{:shock}([:GDP, :INF, :RATE]),
                     Axis{:horizon}(0:10)
                 ) for _ in coverage]
        upper = [AxisArray(
                     randn(n_vars, n_shocks, n_horizons),
                     Axis{:variable}([:GDP, :INF, :RATE]),
                     Axis{:shock}([:GDP, :INF, :RATE]),
                     Axis{:horizon}(0:10)
                 ) for _ in coverage]

        id = CholeskyID()
        metadata = (max_horizon = 10, n_draws = n_draws)

        # Construct BayesianIRFResult
        irf_result = BayesianIRFResult{Float64, typeof(data), typeof(id)}(
            data, lower, upper, coverage, id, metadata
        )

        @test irf_result isa AbstractIRFResult{Float64}
        @test irf_result.data isa AxisArray
        @test size(irf_result.data) == (n_draws, n_vars, n_shocks, n_horizons)
    end

    @testset "BayesianIRFResult accessor functions" begin
        num_draws = 50
        num_vars = 2
        num_shocks = 2
        num_horizons = 6

        data = AxisArray(
            randn(num_draws, num_vars, num_shocks, num_horizons),
            Axis{:draw}(1:num_draws),
            Axis{:variable}([:Y1, :Y2]),
            Axis{:shock}([:S1, :S2]),
            Axis{:horizon}(0:5)
        )

        coverage = [0.68, 0.90]
        lower = [AxisArray(zeros(num_vars, num_shocks, num_horizons),
                     Axis{:variable}([:Y1, :Y2]),
                     Axis{:shock}([:S1, :S2]),
                     Axis{:horizon}(0:5)) for _ in coverage]
        upper = [AxisArray(ones(num_vars, num_shocks, num_horizons),
                     Axis{:variable}([:Y1, :Y2]),
                     Axis{:shock}([:S1, :S2]),
                     Axis{:horizon}(0:5)) for _ in coverage]

        id = CholeskyID()
        metadata = (max_horizon = 5, n_draws = num_draws)

        irf_result = BayesianIRFResult{Float64, typeof(data), typeof(id)}(
            data, lower, upper, coverage, id, metadata
        )

        # Test accessors
        @test has_draws(irf_result) == true
        @test n_draws(irf_result) == 50
        @test length(coverages(irf_result)) == 2
        @test length(lowerbounds(irf_result)) == 2
        @test length(upperbounds(irf_result)) == 2

        # Test point_estimate (median)
        pe = point_estimate(irf_result)
        @test pe isa AxisArray
        @test size(pe) == (num_vars, num_shocks, num_horizons)

        # Test mean_estimate
        me = mean_estimate(irf_result)
        @test me isa AxisArray
        @test size(me) == (num_vars, num_shocks, num_horizons)

        # Test get_draws
        draws = get_draws(irf_result)
        @test draws === irf_result.data
    end

    @testset "LocalProjectionIRFResult construction" begin
        n_response = 1
        n_shock = 1
        n_horizons = 13

        data = AxisArray(
            randn(n_response, n_shock, n_horizons),
            Axis{:response}([:y]),
            Axis{:shock}([:x]),
            Axis{:horizon}(0:12)
        )

        stderr = AxisArray(
            abs.(randn(n_response, n_shock, n_horizons)) * 0.1,
            Axis{:response}([:y]),
            Axis{:shock}([:x]),
            Axis{:horizon}(0:12)
        )

        coverage = [0.68, 0.95]
        lower = [AxisArray(Array(data) .- 1.0,
                     Axis{:response}([:y]),
                     Axis{:shock}([:x]),
                     Axis{:horizon}(0:12)) for _ in coverage]
        upper = [AxisArray(Array(data) .+ 1.0,
                     Axis{:response}([:y]),
                     Axis{:shock}([:x]),
                     Axis{:horizon}(0:12)) for _ in coverage]

        metadata = (horizon = 12, term = :x)

        irf_result = LocalProjectionIRFResult{Float64, typeof(data)}(
            data, stderr, lower, upper, coverage, metadata
        )

        @test irf_result isa AbstractIRFResult{Float64}
        @test irf_result.data isa AxisArray
        @test size(irf_result.stderr) == (n_response, n_shock, n_horizons)
    end

    @testset "LocalProjectionIRFResult accessor functions" begin
        n_horizons = 6

        data = AxisArray(
            randn(1, 1, n_horizons),
            Axis{:response}([:y]),
            Axis{:shock}([:x]),
            Axis{:horizon}(0:5)
        )

        stderr = AxisArray(
            abs.(randn(1, 1, n_horizons)) * 0.1,
            Axis{:response}([:y]),
            Axis{:shock}([:x]),
            Axis{:horizon}(0:5)
        )

        coverage = [0.90]
        lower = [AxisArray(Array(data) .- 1.96 .* Array(stderr),
            Axis{:response}([:y]),
            Axis{:shock}([:x]),
            Axis{:horizon}(0:5))]
        upper = [AxisArray(Array(data) .+ 1.96 .* Array(stderr),
            Axis{:response}([:y]),
            Axis{:shock}([:x]),
            Axis{:horizon}(0:5))]

        metadata = (horizon = 5, term = :x)

        irf_result = LocalProjectionIRFResult{Float64, typeof(data)}(
            data, stderr, lower, upper, coverage, metadata
        )

        # Test accessors
        @test has_draws(irf_result) == false
        @test n_draws(irf_result) == 0
        @test length(coverages(irf_result)) == 1

        # Test point_estimate (returns data directly for LP)
        pe = point_estimate(irf_result)
        @test pe === irf_result.data
    end
end

@testset "Narrative Identification Types" begin
    @testset "NarrativeShockRestriction construction" begin
        # Integer date
        nr1 = NarrativeShockRestriction(1, 50, 1)
        @test nr1.shock == 1
        @test nr1.date == 50
        @test nr1.sign == 1

        # Negative sign
        nr2 = NarrativeShockRestriction(2, 100, -1)
        @test nr2.shock == 2
        @test nr2.date == 100
        @test nr2.sign == -1
    end

    @testset "NarrativeRestriction construction from explicit restrictions" begin
        sign_restrictions = [1 0;
                             -1 0]

        narrative_shocks = [
            NarrativeShockRestriction(1, 50, 1),
            NarrativeShockRestriction(1, 75, -1)
        ]

        # Default horizon
        id = NarrativeRestriction(sign_restrictions, narrative_shocks)
        @test id.sign_restrictions == sign_restrictions
        @test length(id.narrative_shocks) == 2
        @test id.horizon == 0

        # Custom horizon
        id2 = NarrativeRestriction(sign_restrictions, narrative_shocks, 4)
        @test id2.horizon == 4
    end

    @testset "NarrativeRestriction with only narrative shocks" begin
        narrative_shocks = [
            NarrativeShockRestriction(1, 50, 1),
            NarrativeShockRestriction(2, 75, -1)
        ]

        # Should auto-create zero sign_restrictions matrix
        id = NarrativeRestriction(narrative_shocks, 3)  # 3 variables
        @test id.sign_restrictions == zeros(Int, 3, 3)
        @test length(id.narrative_shocks) == 2
    end

    @testset "NarrativeRestriction type hierarchy" begin
        sign_restrictions = [1 0; 0 0]
        narrative_shocks = [NarrativeShockRestriction(1, 50, 1)]

        id = NarrativeRestriction(sign_restrictions, narrative_shocks)

        @test id isa AbstractIdentification
    end
end

@testset "IRF Result Type Hierarchy" begin
    @testset "All IRF types are AbstractIRFResult" begin
        # BayesianIRFResult
        data_b = AxisArray(randn(10, 2, 2, 5),
            Axis{:draw}(1:10),
            Axis{:variable}([:A, :B]),
            Axis{:shock}([:A, :B]),
            Axis{:horizon}(0:4))
        lower_b = [AxisArray(randn(2, 2, 5),
            Axis{:variable}([:A, :B]),
            Axis{:shock}([:A, :B]),
            Axis{:horizon}(0:4))]
        upper_b = [AxisArray(randn(2, 2, 5),
            Axis{:variable}([:A, :B]),
            Axis{:shock}([:A, :B]),
            Axis{:horizon}(0:4))]
        irf_b = BayesianIRFResult{Float64, typeof(data_b), CholeskyID}(
            data_b, lower_b, upper_b, [0.90], CholeskyID(), (;))

        @test irf_b isa AbstractIRFResult
        @test irf_b isa AbstractIRFResult{Float64}

        # LocalProjectionIRFResult
        data_l = AxisArray(randn(1, 1, 5),
            Axis{:response}([:y]),
            Axis{:shock}([:x]),
            Axis{:horizon}(0:4))
        stderr_l = AxisArray(abs.(randn(1, 1, 5)),
            Axis{:response}([:y]),
            Axis{:shock}([:x]),
            Axis{:horizon}(0:4))
        lower_l = [AxisArray(Array(data_l) .- 1.0,
            Axis{:response}([:y]),
            Axis{:shock}([:x]),
            Axis{:horizon}(0:4))]
        upper_l = [AxisArray(Array(data_l) .+ 1.0,
            Axis{:response}([:y]),
            Axis{:shock}([:x]),
            Axis{:horizon}(0:4))]
        irf_l = LocalProjectionIRFResult{Float64, typeof(data_l)}(
            data_l, stderr_l, lower_l, upper_l, [0.90], (;))

        @test irf_l isa AbstractIRFResult
        @test irf_l isa AbstractIRFResult{Float64}
    end
end

println("✓ All hub types tests passed!")
